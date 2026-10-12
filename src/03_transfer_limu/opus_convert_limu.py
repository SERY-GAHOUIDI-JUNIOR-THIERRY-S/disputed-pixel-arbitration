#!/usr/bin/env python3
"""
================================================================================
LIMU -> CDnet-layout CONVERTER
================================================================================
Normalizes the LIMU "Detection of Moving Objects" scenes into the exact CDnet2014
layout your whole pipeline already expects, so cpb_train_extra.py / the framework /
opus_transfer_eval.py all run UNCHANGED.

RAW layout (as you have it), per scene S in {CameraParameter, Intersection, LightSwitch}:
    <root>/limu/<S>/<S>/  image00000.jpg ...              (source frames, .jpg)
                          imageNNNNN_gt.bmp ...            (GT masks, .bmp, ~every 15th)
    (Intersection uses image_%04d.jpg and image_%04d_gt.bmp)

WRITES (CDnet convention, 1-based, 6-digit):
    <root>/limu/<S>/input/inNNNNNN.jpg          (ALL source frames, sequential)
    <root>/limu/<S>/groundtruth/gtNNNNNN.png    (255=FG, 0=BG) -- written ONLY for
                                                 the labeled frames (~1 per 15).
                                                 Unlabeled frames get NO gt file, so
                                                 load_gt() returns None and they are
                                                 never scored / never fed to guidance.
    <root>/limu/<S>/temporalROI.txt             ("401 <lastframe>")

PROTOCOL (Liang paper, LIMU):
    * first 400 frames  -> CPB training  (they become the pre-ROI range, so the
      existing pre_roi_nums() picks them up automatically);
    * ROI / evaluation  -> frames 401..end, scored ONLY on frames with a real GT
      mask.  The CPB model in this reproduction is trained once (the .pkl) and
      applied per-frame independently -- there is no online temporal state at test
      time -- so unlabeled frames need not be processed at all.  Keeping GT sparse
      (no filler masks) means precompute/eval touch only the ~800 labeled frames,
      which is both correct (matches the paper) and ~15x faster on CPU.

GT MAPPING: LIMU says "white pixels denote foreground" -> pixel >=128 becomes FG
(255), everything else BG (0).  The summary prints the true global histogram so we
can confirm there is no third ("don't care") value before trusting the numbers.

USAGE
    python opus_convert_limu.py                       # auto-find root, all 3 scenes
    python opus_convert_limu.py --dataset_root data/cdnet2014
    python opus_convert_limu.py --link                # hardlink inputs (fast, no copy)
    python opus_convert_limu.py --force               # overwrite an existing conversion
    python opus_convert_limu.py --train_frames 400    # override training-frame budget
================================================================================
"""
import argparse, os, re, shutil, glob
import numpy as np
import cv2

SCENES = ['CameraParameter', 'Intersection', 'LightSwitch']
IMG_EXT = ('.jpg', '.jpeg')
OUTSIDE_ROI = 85          # CDnet "outside ROI" -> excluded by frame_counts
FG_VAL, BG_VAL = 255, 0


def find_root(explicit):
    if explicit:
        return explicit
    for c in ('data/cdnet2014', 'dataset/cdnet2014', 'cdnet2014',
              'data/CDnet2014', 'CDnet2014'):
        if os.path.isdir(os.path.join(c, 'limu')):
            return c
    # last resort: any folder that has a 'limu' subdir
    for c in ('data', 'dataset', '.'):
        if os.path.isdir(os.path.join(c, 'limu')):
            return c
    return None


def inner_data_dir(scene_dir):
    """Descend into the doubled folder: return the dir that actually holds the RAW
    jpgs.  Ignores our own output folders (input/groundtruth) so a re-run with
    --force never mistakes a previously-created input/ for the raw data, and
    prefers the scene-named subfolder when several qualify."""
    SKIP = {'input', 'groundtruth'}
    try:
        entries = os.listdir(scene_dir)
    except Exception:
        return scene_dir
    # raw jpgs directly here?
    if any(f.lower().endswith(IMG_EXT) for f in entries):
        return scene_dir
    subs = [d for d in entries
            if os.path.isdir(os.path.join(scene_dir, d)) and d.lower() not in SKIP]
    base = os.path.basename(scene_dir.rstrip('/\\')).lower()
    # try the scene-named subfolder first, then the rest
    subs.sort(key=lambda d: (d.lower() != base, d.lower()))
    for d in subs:
        p = os.path.join(scene_dir, d)
        try:
            if any(f.lower().endswith(IMG_EXT) for f in os.listdir(p)):
                return p
        except Exception:
            pass
    return scene_dir


def idx_of(name):
    """Trailing integer of a source name (image00042.jpg / image_0042.jpg)."""
    stem = os.path.splitext(name)[0]
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else None


def gt_idx_of(name):
    """Integer from a GT name imageNNNNN_gt.bmp / image_NNNN_gt.bmp."""
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'_gt$', '', stem, flags=re.IGNORECASE)
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else None


def convert_scene(root, scene, args):
    scene_dir = os.path.join(root, 'limu', scene)
    if not os.path.isdir(scene_dir):
        print(f'  [skip] {scene}: folder not found ({scene_dir})'); return
    data_dir = inner_data_dir(scene_dir)

    files = os.listdir(data_dir)
    srcs = sorted((idx_of(f), f) for f in files
                  if f.lower().endswith(IMG_EXT) and idx_of(f) is not None)
    gts = {}
    for f in files:
        if f.lower().endswith('_gt.bmp') or (f.lower().endswith('.bmp')
                                             and '_gt' in f.lower()):
            gi = gt_idx_of(f)
            if gi is not None:
                gts[gi] = f
    if not srcs:
        print(f'  [skip] {scene}: no source jpgs under {data_dir}'); return

    out_in = os.path.join(scene_dir, 'input')
    out_gt = os.path.join(scene_dir, 'groundtruth')
    roi_path = os.path.join(scene_dir, 'temporalROI.txt')
    if os.path.exists(out_in) and not args.force:
        print(f'  [skip] {scene}: already converted (use --force to redo)'); return
    for d in (out_in, out_gt):
        if os.path.exists(d) and args.force:
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # map source frames to sequential 1-based CDnet numbers, in source-index order
    srcs.sort(key=lambda t: t[0])
    total = len(srcs)

    gt_hist = {}         # global pixel histogram across labeled masks
    n_gt = 0             # labeled frames written
    n_gt_fg = 0          # labeled frames that actually contain foreground
    n_gt_in_roi = 0      # labeled frames that fall in the eval range (>400)
    size = None
    train_n = args.train_frames

    for pos, (sidx, sname) in enumerate(srcs):
        num = pos + 1                                 # 1-based CDnet number
        src_path = os.path.join(data_dir, sname)
        if not os.path.exists(src_path):
            continue
        dst_in = os.path.join(out_in, f'in{num:06d}.jpg')
        if args.link:
            try:
                if os.path.exists(dst_in):
                    os.remove(dst_in)
                os.link(src_path, dst_in)
            except Exception:
                shutil.copy2(src_path, dst_in)
        else:
            shutil.copy2(src_path, dst_in)

        # frame size (for resizing any odd-sized mask); read once
        if size is None:
            im = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            size = im.shape if im is not None else (240, 320)

        # GT is SPARSE: write a mask ONLY for labeled frames. Unlabeled frames get
        # no gt file at all -> load_gt() returns None -> skipped by precompute/eval.
        if sidx in gts:
            m = cv2.imread(os.path.join(data_dir, gts[sidx]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != size:
                m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            for v, c in zip(*[x.tolist() for x in np.unique(m, return_counts=True)]):
                gt_hist[v] = gt_hist.get(v, 0) + c
            out = np.where(m >= 128, FG_VAL, BG_VAL).astype(np.uint8)
            cv2.imwrite(os.path.join(out_gt, f'gt{num:06d}.png'), out)
            n_gt += 1
            if (out == FG_VAL).any():
                n_gt_fg += 1
            if num > train_n:
                n_gt_in_roi += 1

    roi_start = min(train_n + 1, total)
    with open(roi_path, 'w') as fh:
        fh.write(f'{roi_start} {total}')

    print(f'\n  == {scene} ==')
    print(f'     source frames  : {total}   (size {size[0]}x{size[1]})')
    print(f'     GT masks found : {n_gt}   ({n_gt_fg} contain foreground)')
    print(f'     training range : in000001..in{train_n:06d}   (pre-ROI, CPB train)')
    print(f'     eval  range    : in{roi_start:06d}..in{total:06d}   '
          f'({n_gt_in_roi} scored GT frames in ROI)')
    print(f'     GT pixel histogram (raw values across all masks): '
          f'{sorted(gt_hist.items())}')
    if any(v not in (0, 255) for v in gt_hist):
        print(f'     [NOTE] GT has values other than 0/255 -> mapped as >=128 => FG. '
              f'Confirm none of those are a "don\'t care" label.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', default=None,
                    help='the cdnet2014 folder (auto-detected if omitted)')
    ap.add_argument('--train_frames', type=int, default=400,
                    help='frames used for CPB training (paper: 400 for LIMU)')
    ap.add_argument('--link', action='store_true',
                    help='hardlink input frames instead of copying (fast, no extra disk)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing input/ groundtruth/')
    args = ap.parse_args()

    root = find_root(args.dataset_root)
    if not root:
        raise SystemExit('Could not find a cdnet2014 folder containing a "limu" '
                         'subdir. Pass --dataset_root explicitly.')
    print('=' * 78)
    print(f'  LIMU -> CDnet converter   root={root}   train_frames={args.train_frames}'
          f'   mode={"hardlink" if args.link else "copy"}')
    print('=' * 78)
    for s in SCENES:
        convert_scene(root, s, args)
    print('\n' + '=' * 78)
    print('  DONE. Each scene now has input/ , groundtruth/ , temporalROI.txt')
    print('  Next:')
    print('    1) CPB training:')
    print('         python cpb_train_extra.py --dataset limu --output_dir models/cpb_limu')
    print('    2) then guidance precompute + zero-shot transfer eval '
          '(opus_transfer_eval.py)')
    print('=' * 78 + '\n')


if __name__ == '__main__':
    main()
