"""
ZBS (An et al., CVPR 2023) under our protocol, part 1 of 2: all-instance detection with the authors' detector.

Runs the exact Detic model that ZBS uses (Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size, LVIS vocabulary,
ONE_CLASS_PER_PROPOSAL, as in ZBS main.py/setup_cfg) on the frames our protocol needs:
  - the background-initialisation window (= the training window of every other method), and
  - every scored frame.
Detections are cached at score >= 0.30 (the lowest threshold of any ZBS parameter group); a higher ZBS threshold
is applied later by filtering, which is equivalent because detectron2 filters by score before per-class NMS.
Part 2 (zbs_post.py, run locally) applies the ZBS instance-level background model and foreground selection
with the authors' own code (SORT tracker and selection rules) and scores the masks with our metric.

Output: <out>/<dataset>__<seq>.pkl.gz  = dict(meta=..., frames=[dict(n, role, h, w, boxes, scores, classes, rle)])
Resumable: finished sequences are skipped; files are written atomically.
"""
import os, sys, re, json, time, gzip, pickle, argparse, traceback

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')     # Detic checkpoints predate torch>=2.6 weights_only
import numpy as np, cv2, torch

CONFIG = 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
CACHE_CONF = 0.30
IMG_EXT = ('.jpg', '.jpeg', '.bmp', '.png')
LIMU_SEQS = ['CameraParameter', 'Intersection', 'LightSwitch']
WF_SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']
ORDER = ['lasiesta', 'limu', 'wallflower']


def num(name):
    return int(re.findall(r'(\d+)', os.path.basename(name))[-1])


def index_frames(d):
    """frame number -> path for the input images of one sequence folder (ground-truth files excluded)."""
    out = {}
    for f in os.listdir(d):
        fl = f.lower()
        if not fl.endswith(IMG_EXT) or 'gt' in fl or fl.startswith('hand_segmented') or not re.search(r'\d', f):
            continue
        out[num(f)] = os.path.join(d, f)
    return out


def _has(d, pred):
    try:
        return any(pred(f) for f in os.listdir(d))
    except OSError:
        return False


CHECKS = {
    'limu': lambda d: _has(d, lambda f: f.lower().endswith('.jpg') and f.lower().startswith('image')),
    'wallflower': lambda d: os.path.exists(os.path.join(d, 'b00000.bmp')),
    'lasiesta': lambda d: _has(d, lambda f: f.lower().endswith(('.bmp', '.png')) and f.startswith(os.path.basename(d) + '-')),
}


def find_sequences(base, manifest):
    """Search <base> recursively for every sequence folder of the manifest. Returns {(ds, seq): dir}."""
    wanted = {(ds, s) for ds in manifest for s in manifest[ds]}
    found = {}
    for root, dirs, _ in os.walk(base, followlinks=True):
        for ds, s in wanted:
            if (ds, s) not in found and s in dirs and CHECKS[ds](os.path.join(root, s)):
                found[(ds, s)] = os.path.join(root, s)
        if len(found) == len(wanted):
            break
    return found


def frame_plan(ds, spec):
    """[(frame_number, role)] in temporal order. role: 'init' (background window) or 'score'."""
    a, b = spec['init']
    plan = [(n, 'init') for n in range(a, b + 1)]
    if ds == 'lasiesta':
        plan += [(n, 'score') for n in range(spec['score'][0], spec['score'][1] + 1)]
    else:
        plan += [(n, 'score') for n in spec['score']]
    return plan


def verify(manifest, dirs):
    """Every needed frame must exist; LASIESTA sequences must have exactly L frames (same copy as ours)."""
    problems, total = [], 0
    for ds in ORDER:
        for s, spec in manifest[ds].items():
            if (ds, s) not in dirs:
                problems.append(f'{ds}/{s}: folder not found'); continue
            idx = index_frames(dirs[(ds, s)])
            plan = frame_plan(ds, spec); total += len(plan)
            miss = [n for n, _ in plan if n not in idx]
            if miss:
                problems.append(f'{ds}/{s}: {len(miss)} needed frames missing (e.g. {miss[:3]})')
            if ds == 'lasiesta' and len(idx) != spec['L']:
                problems.append(f'{ds}/{s}: {len(idx)} frames found, expected {spec["L"]}')
    return problems, total


def build_predictor(zbs_root, weights, device):
    """Same configuration as ZBS main.py setup_cfg() + VisualizationDemo (vocabulary lvis), threshold CACHE_CONF."""
    os.chdir(zbs_root)                                   # metadata paths in Detic are relative to the repo root
    for p in (zbs_root, os.path.join(zbs_root, 'third_party/CenterNet2')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    from centernet.config import add_centernet_config
    from detic.config import add_detic_config
    from detic.modeling.utils import reset_cls_test
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    add_centernet_config(cfg); add_detic_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.merge_from_list(['MODEL.WEIGHTS', weights or ''])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CACHE_CONF
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CACHE_CONF
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CACHE_CONF
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    pred = DefaultPredictor(cfg)
    meta = MetadataCatalog.get('lvis_v1_val')
    reset_cls_test(pred.model, 'datasets/metadata/lvis_v1_clip_a+cname.npy', len(meta.thing_classes))
    return pred, cfg, list(meta.thing_classes)


def check_weights(model, weights):
    """Fraction of model parameters whose name and shape are present in the checkpoint (should be ~1.0)."""
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    ck = ck.get('model', ck)
    sd = model.state_dict()
    ok = sum(1 for k, v in sd.items() if k in ck and tuple(np.shape(ck[k])) == tuple(v.shape))
    return ok / max(1, len(sd)), len(sd), len(ck)


def encode(inst):
    from pycocotools import mask as mask_util
    m = inst.pred_masks.numpy().astype(np.uint8)          # (k, H, W), already pasted at the input resolution
    rle = [mask_util.encode(np.asfortranarray(x)) for x in m]
    return dict(boxes=inst.pred_boxes.tensor.numpy().astype(np.float32),
                scores=inst.scores.numpy().astype(np.float32),
                classes=inst.pred_classes.numpy().astype(np.int16), rle=rle)


@torch.no_grad()
def detect(pred, img):
    return pred(img)['instances'].to('cpu')


def run_sequence(pred, ds, s, d, spec, out, meta, deadline=None, log=print, max_frames=None):
    dst = os.path.join(out, f'{ds}__{s}.pkl.gz')
    if os.path.exists(dst):
        log(f'  {ds}/{s}: already done, skipped'); return 'skipped'
    idx = index_frames(d); plan = frame_plan(ds, spec)
    if max_frames:
        plan = plan[:max_frames]
    frames, t0 = [], time.time()
    for i, (n, role) in enumerate(plan):
        if deadline and time.time() > deadline:
            log(f'  {ds}/{s}: time budget reached at frame {i}/{len(plan)}; sequence not saved'); return 'timeout'
        img = cv2.imread(idx[n], cv2.IMREAD_COLOR)        # BGR, as detectron2 read_image(format="BGR") in ZBS
        if img is None:
            raise IOError(f'cannot read {idx[n]}')
        inst = detect(pred, img)
        frames.append(dict(n=n, role=role, h=img.shape[0], w=img.shape[1], **encode(inst)))
        if (i + 1) % 100 == 0 or i + 1 == len(plan):
            el = time.time() - t0
            log(f'  {ds}/{s}: {i + 1}/{len(plan)} frames  {(i + 1) / el:.2f} fps  ({len(inst)} det. in last frame)')
    tmp = dst + '.tmp'
    with gzip.open(tmp, 'wb', compresslevel=4) as f:
        pickle.dump(dict(meta=dict(meta, dataset=ds, seq=s, spec=spec, frames=len(frames)), frames=frames), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    return 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zbs', required=True); ap.add_argument('--weights', default='')
    ap.add_argument('--data', required=True); ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--only', default=''); ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--hours', type=float, default=11.0)
    a = ap.parse_args()
    man = json.load(open(a.manifest)); os.makedirs(a.out, exist_ok=True)
    dirs = find_sequences(a.data, man)
    pred, cfg, classes = build_predictor(a.zbs, a.weights, a.device)
    meta = dict(config=CONFIG, cache_conf=CACHE_CONF, torch=torch.__version__)
    deadline = time.time() + a.hours * 3600
    for ds in ORDER:
        for s, spec in man[ds].items():
            if a.only and s not in a.only.split(','):
                continue
            run_sequence(pred, ds, s, dirs[(ds, s)], spec, a.out, meta, deadline, max_frames=a.max_frames or None)


if __name__ == '__main__':
    main()
