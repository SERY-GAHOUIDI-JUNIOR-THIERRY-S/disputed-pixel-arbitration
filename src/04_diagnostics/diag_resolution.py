#!/usr/bin/env python3
"""
================================================================================
RESOLUTION DIAGNOSTIC — why is cache-CPB 0.513 but full-ROI-CPB 0.428?
================================================================================
The span check proved the 60-frame cache is UNIFORM across the ROI, so the gap is
NOT temporal sampling. The two remaining suspects are (a) evaluation RESOLUTION
and (b) a CPB code-path difference. This isolates (a): it runs CPB on the EXACT
SAME cached frames and scores each frame BOTH ways --

   native : prediction resized UP to groundtruth size, scored at GT resolution
            (official CDnet protocol; this is what opus_paper_metrics_table did -> 0.513)
   fit    : groundtruth resized DOWN to the model's fit size, scored there
            (what liang_v7 does at fit=HxW -> 0.428, if resolution is the cause)

If  native ~= 0.513  and  fit ~= 0.428  -> resolution is the ENTIRE story.
Then 0.513 (native) is the CORRECT, CDnet-official number and the 60-frame cache
is a valid protocol -- no full-ROI run needed.

USAGE
  python diag_resolution.py --models_dir models/cpb_numba_rgb --cache_dir drl_cache --channel_mode rgb
================================================================================
"""
import argparse, os, glob, pickle, json, time
import numpy as np
import cv2
import opus_final_hod_framework as F

CATS = ['baseline', 'cameraJitter', 'badWeather', 'dynamicBackground', 'intermittentObjectMotion',
        'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal', 'turbulence']


def f1c(c):
    return F.metrics_from_counts(c[0], c[1], c[2], c[3])['f1']


def add(a, b):
    return [a[i] + b[i] for i in range(4)]


def catmean(scene_counts, cat_of):
    by = {}
    for sid, c in scene_counts.items():
        by.setdefault(cat_of[sid], []).append(f1c(c))
    cv = {cat: float(np.mean(v)) for cat, v in by.items()}
    return cv, (float(np.mean(list(cv.values()))) if cv else 0.0)


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
    ap.add_argument('--out', default='diag_resolution.json')
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet2014 not found - use --dataset_root')
    caches = sorted(glob.glob(os.path.join(args.cache_dir, '*.npz')))
    if not caches:
        raise SystemExit(f'no caches in {args.cache_dir}')

    native = {}; fit = {}; cat_of = {}
    t0 = time.time()
    for ci, cpath in enumerate(caches):
        stem = os.path.basename(cpath)[:-4]
        if '__' not in stem:
            continue
        cat, scene = stem.split('__', 1)
        mp = os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')
        sp = os.path.join(root, cat, scene)
        if not (os.path.exists(mp) and os.path.isdir(sp)):
            print(f'  [skip] {cat}/{scene}'); continue
        model = pickle.load(open(mp, 'rb'))
        dat = np.load(cpath); nums = [int(n) for n in dat['nums']]
        idir = os.path.join(sp, 'input'); gdir = os.path.join(sp, 'groundtruth')
        pipe = F.HoDPipeline(model, 'stage3', args.floor_min, args.af, args.sigema,
                             args.bth_pixels, args.ref, True, args.channel_mode)
        cn = [0, 0, 0, 0]; cf = [0, 0, 0, 0]; used = 0
        for n in nums:
            bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
            if bgr is None or gt is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pipe.run(rgb, np.zeros(gt.shape, np.uint8))
            s1 = pipe.last_s1
            if s1 is None:
                continue
            # NATIVE: resize prediction up to GT
            s1_nat = s1 if s1.shape == gt.shape else cv2.resize(s1, (gt.shape[1], gt.shape[0]),
                                                                interpolation=cv2.INTER_NEAREST)
            cn = add(cn, list(F.frame_counts(s1_nat, gt)))
            # FIT: resize GT down to prediction (model fit) size
            gt_fit = gt if s1.shape == gt.shape else cv2.resize(gt, (s1.shape[1], s1.shape[0]),
                                                                interpolation=cv2.INTER_NEAREST)
            cf = add(cf, list(F.frame_counts(s1, gt_fit)))
            used += 1
        if used == 0:
            continue
        sid = f'{cat}/{scene}'
        native[sid] = cn; fit[sid] = cf; cat_of[sid] = cat
        print(f'  [{ci+1}/{len(caches)}] {sid:34s} n={used:3d}  native={f1c(cn):.3f}  fit={f1c(cf):.3f}'
              f'  d={f1c(cn)-f1c(cf):+.3f}  ({time.time()-t0:.0f}s)')

    ncv, ncm = catmean(native, cat_of)
    fcv, fcm = catmean(fit, cat_of)
    print(f"\n{'#'*70}\n  RESOLUTION DIAGNOSTIC  ({len(cat_of)} scenes, identical frames)\n{'#'*70}")
    print(f"  category-mean CPB, scored NATIVE (CDnet official) = {ncm:.4f}   (expect ~0.513)")
    print(f"  category-mean CPB, scored FIT   (downscaled)      = {fcm:.4f}   (expect ~0.428)")
    print(f"  full-ROI liang_v7 (fit) reference                 = 0.4284")
    print(f"\n  {'category':<26}{'native':>10}{'fit':>10}{'delta':>10}")
    for c in CATS:
        if c in ncv:
            print(f"  {c:<26}{ncv[c]:>10.4f}{fcv.get(c,0):>10.4f}{ncv[c]-fcv.get(c,0):>+10.4f}")
    print(f"\n  READ: if fit ~= 0.428 and native ~= 0.513, resolution is the whole gap.")
    print(f"  Then NATIVE is the correct CDnet metric and the 60-frame cache is valid.")

    json.dump({'native_catmean': ncm, 'fit_catmean': fcm, 'native_per_cat': ncv, 'fit_per_cat': fcv,
               'per_scene': {sid: {'native': f1c(native[sid]), 'fit': f1c(fit[sid])} for sid in cat_of}},
              open(args.out, 'w'), indent=2)
    print(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()