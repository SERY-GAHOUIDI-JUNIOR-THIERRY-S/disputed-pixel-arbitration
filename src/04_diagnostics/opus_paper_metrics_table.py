#!/usr/bin/env python3
"""
================================================================================
PAPER METRICS TABLE — CPB / Guidance / Framework in THREE aggregations
================================================================================
Produces, on the SAME 60-frame cache used by the DRL arbiter (so every row is
apples-to-apples), the reproduction baselines in all three CDnet aggregations:

    * pooled        micro-average over ALL scenes' pixel counts   (== Liang CDnet "Overall")
    * scene-mean    macro over scenes (equal weight per scene)
    * category-mean macro over the 11 categories of the per-scene mean (official CDnet)

Rows: CPB-alone (stage-1), Guidance-alone (the cached mask), Framework (stage-3,
all-64 + paper stage-2).  The arbiter / MLP rows come from your CV run; this fills
the three baseline rows so the whole table is consistent.

PREREQ: apply the k_stage2 paper patch to opus_final_hod_framework.py first, so
F.HoDPipeline runs the locked all-64 + paper communication.

USAGE
  # DeepLab guidance (drl_cache):
  python opus_paper_metrics_table.py --models_dir models/cpb_numba_rgb \
      --cache_dir drl_cache --channel_mode rgb --out paper_metrics_deeplab.json
  # PSPNet guidance (drl_cache_pspnet):
  python opus_paper_metrics_table.py --models_dir models/cpb_numba_rgb \
      --cache_dir drl_cache_pspnet --channel_mode rgb --out paper_metrics_pspnet.json
================================================================================
"""
import argparse, os, glob, pickle, json, time
import numpy as np
import cv2
import opus_final_hod_framework as F


def f1_from_counts(c):
    return F.metrics_from_counts(c[0], c[1], c[2], c[3])['f1']


def add(a, b):
    return [a[i] + b[i] for i in range(4)]


def aggregate(scene_counts, scene_cat):
    """scene_counts: {sid: (tp,fp,fn,tn)} ; scene_cat: {sid: category}.
    Returns dict with pooled, scene_mean, category_mean."""
    # pooled
    tot = [0, 0, 0, 0]
    for c in scene_counts.values():
        tot = add(tot, c)
    pooled = f1_from_counts(tot)
    # scene-mean
    sf = [f1_from_counts(c) for c in scene_counts.values()]
    scene_mean = float(np.mean(sf)) if sf else 0.0
    # category-mean = mean over categories of (mean of per-scene F1 within category)
    by_cat = {}
    for sid, c in scene_counts.items():
        by_cat.setdefault(scene_cat[sid], []).append(f1_from_counts(c))
    cat_vals = {cat: float(np.mean(v)) for cat, v in by_cat.items()}
    category_mean = float(np.mean(list(cat_vals.values()))) if cat_vals else 0.0
    return {'pooled': pooled, 'scene_mean': scene_mean,
            'category_mean': category_mean, 'per_cat': cat_vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--channel_mode', default='rgb', choices=['auto', 'gray', 'rgb'])
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--out', default='paper_metrics.json')
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet2014 not found — use --dataset_root')

    caches = sorted(glob.glob(os.path.join(args.cache_dir, '*.npz')))
    if not caches:
        raise SystemExit(f'no .npz caches in {args.cache_dir}')

    cpb = {}; guid = {}; fw = {}; cat_of = {}
    t0 = time.time()
    for ci, cpath in enumerate(caches):
        stem = os.path.basename(cpath)[:-4]           # cat__scene
        if '__' not in stem:
            continue
        cat, scene = stem.split('__', 1)
        mp = os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')
        sp = os.path.join(root, cat, scene)
        if not (os.path.exists(mp) and os.path.isdir(sp)):
            print(f'  [skip] {cat}/{scene} (missing model or scene)'); continue
        model = pickle.load(open(mp, 'rb'))
        dat = np.load(cpath)
        nums = [int(n) for n in dat['nums']]; gmask = dat['guid']
        idir = os.path.join(sp, 'input'); gdir = os.path.join(sp, 'groundtruth')

        pipe = F.HoDPipeline(model, 'stage3', args.floor_min, args.af, args.sigema,
                             args.bth_pixels, args.ref, True, args.channel_mode)
        # SEPARATE pipe for pure, unsupervised CPB: last_s1 after a guidance-fed run is
        # contaminated by the guidance, so CPB-alone MUST come from a guidance-free pass.
        pipe_cpb = F.HoDPipeline(model, 'stage3', args.floor_min, args.af, args.sigema,
                                 args.bth_pixels, args.ref, True, args.channel_mode)
        cc = [0, 0, 0, 0]; gc = [0, 0, 0, 0]; fc = [0, 0, 0, 0]; used = 0
        for i, n in enumerate(nums):
            bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
            if bgr is None or gt is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gm = gmask[i]
            if gm.shape != gt.shape:
                gm = cv2.resize(gm, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            fpred = pipe.run(rgb, gm)                  # framework (stage-3, all64+paper)
            pipe_cpb.run(rgb, np.zeros(gt.shape, np.uint8))  # guidance-free pass
            s1 = pipe_cpb.last_s1                       # CPB-alone (unsupervised, no guidance)
            if fpred.shape != gt.shape:
                fpred = cv2.resize(fpred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            if s1 is not None and s1.shape != gt.shape:
                s1 = cv2.resize(s1, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            gpred = (gm > 127).astype(np.uint8) * 255
            fc = add(fc, list(F.frame_counts(fpred, gt)))
            cc = add(cc, list(F.frame_counts(s1, gt)))
            gc = add(gc, list(F.frame_counts(gpred, gt)))
            used += 1
        if used == 0:
            continue
        sid = f'{cat}/{scene}'
        cpb[sid] = cc; guid[sid] = gc; fw[sid] = fc; cat_of[sid] = cat
        print(f'  [{ci+1}/{len(caches)}] {sid:32s} frames={used}  '
              f'CPB={f1_from_counts(cc):.3f}  guid={f1_from_counts(gc):.3f}  fw={f1_from_counts(fc):.3f}'
              f'  ({time.time()-t0:.0f}s)')

    A = {'CPB (RGB, all64)': aggregate(cpb, cat_of),
         'Guidance': aggregate(guid, cat_of),
         'Framework (all64+paper)': aggregate(fw, cat_of)}

    print('\n' + '=' * 78)
    print(f'  PAPER METRICS  (cache={args.cache_dir}, {len(cat_of)} scenes, same frames as arbiter)')
    print('=' * 78)
    print(f"  {'method':<28}{'category_mean':>15}{'scene_mean':>13}{'pooled':>10}")
    print('  ' + '-' * 66)
    for k, v in A.items():
        print(f"  {k:<28}{v['category_mean']:>15.4f}{v['scene_mean']:>13.4f}{v['pooled']:>10.4f}")
    print('  ' + '-' * 66)
    print("  Liang CPB              (reported)     0.7070*         -           0.6330")
    print("  Liang CPB+DeepLabV3+   (reported)     0.7710*         -           0.7530")
    print("   * Liang category-mean is COMPUTED from their Table 2 cells; they report only pooled 'Overall'.")

    print(f"\n  Per-category (Framework all64+paper):")
    for cat in sorted(A['Framework (all64+paper)']['per_cat']):
        print(f"    {cat:<26}{A['Framework (all64+paper)']['per_cat'][cat]:.4f}")

    out = {'cache_dir': args.cache_dir, 'models_dir': args.models_dir,
           'n_scenes': len(cat_of), 'channel_mode': args.channel_mode,
           'summary': A,
           'per_scene_f1': {sid: {'cat': cat_of[sid], 'cpb': f1_from_counts(cpb[sid]),
                                  'guid': f1_from_counts(guid[sid]), 'fw': f1_from_counts(fw[sid])}
                            for sid in cat_of},
           'liang_reference': {'cpb': {'category_mean': 0.7070, 'pooled': 0.6330},
                               'cpb_deeplab': {'category_mean': 0.7710, 'pooled': 0.7530}}}
    json.dump(out, open(args.out, 'w'), indent=2)
    print(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()