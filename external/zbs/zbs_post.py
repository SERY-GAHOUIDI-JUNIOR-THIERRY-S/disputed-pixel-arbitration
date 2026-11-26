"""
ZBS (An et al., CVPR 2023) under our protocol, part 2 of 2: instance-level background model + foreground selection,
then scoring with our metric. Input: the cached Detic detections produced by zbs_detect.py (Kaggle GPU).

The logic follows the authors' released main.py (image-sequence mode) line by line, with the authors' own SORT
tracker (sort_zbs.py) and helper functions (IoU, IoF, foreground_selection, cal_maxIoU, copied verbatim below):
  - initialisation frames (= our training window) feed SORT with detections at score >= conf;
  - at the first scored frame the instance-level background model is built once from the tracks;
  - scored frames keep detections at score >= conf + delta_conf; an instance is foreground if its class has no
    background instance, or if it passes the foreground-selection rule.
Parameter set (fixed before seeing any result): the paper's universal parameters, which are also the defaults of
the released code: tau_conf = 0.6 (= 0.4 + 0.2 in the background stage), tau_move = 0.5, tau_fore = 0.8, IoF rule,
SORT max_age 20, min_hits 5, IoU 0.4. Groups 'p1'..'p3' of the authors' CDnet script are available for sensitivity.
Only change: instance masks are combined with a logical OR (the released code adds them into a uint8 image).

  python3 zbs_post.py --dets <dir with *.pkl.gz> --zbs <ZBS repo> --out zbs_runs [--group paper]
"""
import os, sys, re, json, glob, gzip, pickle, argparse
from collections import defaultdict
import numpy as np, cv2, torch
from scipy.signal import medfilt

GROUPS = {
    # name: conf (background stage), delta_conf, fore_threshold, move_threshold, noiof, max_age, min_hits, iou_threshold
    'paper': dict(conf=0.4, delta=0.2, fore=0.8, move=0.5, noiof=False, max_age=20, min_hits=5, iou=0.4),
    'p1': dict(conf=0.3, delta=0.2, fore=0.3, move=0.5, noiof=True, max_age=100, min_hits=5, iou=0.3),
    'p2': dict(conf=0.4, delta=0.1, fore=0.3, move=0.5, noiof=True, max_age=100, min_hits=5, iou=0.3),
    'p3': dict(conf=0.4, delta=0.2, fore=1.0, move=0.0, noiof=False, max_age=20, min_hits=5, iou=0.4),
}


# ---------------- verbatim from ZBS main.py ----------------
def IoU(box1, box2):
    box_1 = box1.tolist()
    box_2 = box2.tolist()
    ax = max(box_1[0], box_2[0])
    ay = max(box_1[1], box_2[1])
    bx = min(box_1[2], box_2[2])
    by = min(box_1[3], box_2[3])
    area_N = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_M = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h
    return area_X / (area_N + area_M - area_X)


def IoF(box1, box2):
    box_1 = box1.tolist()
    box_2 = box2.tolist()
    ax = max(box_1[0], box_2[0])
    ay = max(box_1[1], box_2[1])
    bx = min(box_1[2], box_2[2])
    by = min(box_1[3], box_2[3])
    area_N = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h
    return area_X / area_N


def cal_maxIoU(det_box, target_boxes):
    max_IoU = 0
    for box in target_boxes:
        max_IoU = max(max_IoU, IoU(det_box, box))
    return max_IoU


def foreground_selection(rec_box, bg_boxes, fore_threshold):
    IoU_list = []
    for bg_box in bg_boxes:
        IoU_list.append(IoU(rec_box, bg_box))
    if np.max(IoU_list) < fore_threshold and IoF(rec_box, bg_boxes[np.argmax(IoU_list)]) < 0.5 + fore_threshold / 2:
        return True
    return False
# -----------------------------------------------------------


def decode(fr, keep):
    from pycocotools import mask as mask_util
    if not keep.any():
        return np.zeros((0, fr['h'], fr['w']), bool)
    return np.stack([mask_util.decode(fr['rle'][i]).astype(bool) for i in np.flatnonzero(keep)])


def zbs_sequence(frames, P, Sort):
    """Yields (frame_number, mask) for every scored frame, following ZBS main.py (image mode)."""
    mot_tracker = Sort(max_age=P['max_age'], min_hits=P['min_hits'], iou_threshold=P['iou'])
    total_trks = np.array([]).reshape(0, 7)
    moved_ids = set()
    cls_bg = defaultdict(list)
    gt_flag = True
    conf = P['conf']
    for fr in frames:
        keep = fr['scores'] >= conf
        if fr['role'] == 'init':
            np_box = fr['boxes'][keep].astype(np.float64)
            np_score = fr['scores'][keep].astype(np.float64).reshape(-1, 1)
            np_cls = fr['classes'][keep].astype(np.float64).reshape(-1, 1)
            dets = np.concatenate((np_box, np_score, np_cls), axis=1)
            trackers = mot_tracker.update(dets)
            total_trks = np.concatenate((total_trks, trackers), axis=0)
            if P['noiof']:
                all_trk_ids = set(trackers[:, 4].astype('uint16'))
                sure_ids = all_trk_ids & moved_ids
                unsure_ids = all_trk_ids - sure_ids
                for trk_id in unsure_ids:
                    trks = total_trks[total_trks[:, 4] == trk_id, :4]
                    delta_trk = np.max(trks, axis=0) - np.min(trks, axis=0)
                    delta_trk[[0, 2]] /= fr['h']          # as released: x extent divided by img.shape[0]
                    delta_trk[[1, 3]] /= fr['w']
                    if sum(delta_trk > P['move']):
                        moved_ids.add(trk_id)
            continue
        if gt_flag:
            total_trk_ids = set(total_trks[:, 4].astype('uint16'))
            if P['noiof']:
                for id in total_trk_ids - moved_ids:
                    trks = total_trks[total_trks[:, 4] == id]
                    bbox = np.average(trks[:, :4], axis=0)
                    for cls in set(trks[:, -1]):
                        cls_bg[int(cls)].append(bbox)
            else:
                for id in total_trk_ids:
                    trks = total_trks[total_trks[:, 4] == id]
                    arg_bbox = np.average(trks[:, :4], axis=0)
                    IoU_trk = np.round([IoU(bbox, arg_bbox) for bbox in trks[:, :4]], 2)
                    kernel = len(IoU_trk) // 9 if (len(IoU_trk) // 9) % 2 else len(IoU_trk) // 9 + 1
                    IoU_trk = medfilt(IoU_trk, kernel)
                    if np.min(IoU_trk) > 1 - P['move']:
                        for cls in set(trks[:, -1]):
                            cls_bg[int(cls)].append(arg_bbox)
            gt_flag = False
            conf = P['conf'] + P['delta']      # as released: the detector is rebuilt with the higher threshold
            # AFTER this frame was detected, so the first scored frame still uses the lower threshold (keep unchanged)
        idx = np.flatnonzero(keep)
        masks = decode(fr, keep)
        boxes = torch.from_numpy(fr['boxes'][keep].astype(np.float32))
        classes = fr['classes'][keep].astype(int)
        out = np.zeros((fr['h'], fr['w']), bool)
        for j in range(len(idx)):
            c = int(classes[j])
            if cls_bg[c] == []:
                out |= masks[j]
            else:
                bg_boxes = torch.Tensor(np.array(cls_bg[c]))
                if P['noiof']:
                    if cal_maxIoU(boxes[j], bg_boxes) < P['fore']:
                        out |= masks[j]
                elif foreground_selection(boxes[j], bg_boxes, P['fore']):
                    out |= masks[j]
        yield fr['n'], out, dict(n_bg=sum(len(v) for v in cls_bg.values()))


def counts(pm, fg, valid):
    return [int((pm & fg & valid).sum()), int((pm & ~fg & valid).sum()), int((~pm & fg & valid).sum()),
            int((~pm & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())]


def score_sequence(ds, seq, gen, roots):
    """Same scoring as bgs_baselines.py (identical frames, GT handling and counts)."""
    sys.path.insert(0, roots['code'])
    if ds == 'limu':
        d = os.path.join(roots['limu'], seq)
        gts = {int(re.findall(r'(\d+)_gt', os.path.basename(p))[-1]): p for p in glob.glob(os.path.join(d, '*_gt.bmp'))}
        rows = []
        for n, m, _ in gen:
            g = cv2.imread(gts[n], 0) > 127
            pm = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            rows.append([n] + counts(pm, g, np.ones_like(g)))
        return rows
    if ds == 'wallflower':
        import wallflower_eval as W
        d = os.path.join(roots['wf'], seq)
        gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
        gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
        res = list(gen); assert len(res) == 1
        s, dd = W.score(res[0][1].astype(np.uint8) * 255, gt, seq == 'MovedObject')
        return dict(score=s, P=dd['prec'], R=dd['rec'], tp=dd['tp'], fp=dd['fp'], fn=dd['fn'], tn=dd['tn'])
    import lasiesta_cpb_eval as LCE
    gtd = os.path.join(roots['lasiesta'], seq + '-GT')
    num = lambda p: int(re.findall(r'(\d+)', os.path.basename(p))[-1])
    gts = {num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
    rows = []
    for n, m, _ in gen:
        fg, valid = LCE.read_gt(gts.get(n, ''))
        if fg is not None:
            rows.append([n] + counts(m, fg, valid))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', required=True); ap.add_argument('--zbs', required=True)
    ap.add_argument('--out', default='zbs_runs'); ap.add_argument('--group', default='paper')
    ap.add_argument('--code', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'code')); ap.add_argument('--limu', default='data/limu')
    ap.add_argument('--wf', default='data/wallflower'); ap.add_argument('--lasiesta', default='data/lasiesta')
    ap.add_argument('--save_masks', default='')
    a = ap.parse_args()
    sys.path.insert(0, a.zbs)
    from sort_zbs import Sort
    roots = dict(code=a.code, limu=a.limu, wf=a.wf, lasiesta=a.lasiesta)
    P = GROUPS[a.group]; meth = 'ZBS' if a.group == 'paper' else f'ZBS-{a.group}'
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.dets, '*.pkl.gz'))):
        ds, seq = os.path.basename(f)[:-7].split('__')
        outp = os.path.join(a.out, f'{meth}__{ds}__{seq}.json')
        if os.path.exists(outp):
            continue
        D = pickle.load(gzip.open(f))
        res = score_sequence(ds, seq, zbs_sequence(D['frames'], P, Sort), roots)
        json.dump(dict(method=meth, dataset=ds, sequence=seq, params=P, result=res), open(outp + '.tmp', 'w'))
        os.replace(outp + '.tmp', outp)
        print(meth, ds, seq, 'ok', flush=True)


if __name__ == '__main__':
    main()
