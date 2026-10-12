#!/usr/bin/env python3
"""
================================================================================
PROTOCOL VALIDATION — is uniform temporal sub-sampling faithful to full ROI?
================================================================================
You already have the EXHAUSTIVE CPB score (full ROI): category-mean 0.4284.
This script computes CPB on UNIFORMLY-SPACED frames at several sample sizes and
shows how fast it converges to that exhaustive value.  CPB needs no guidance, so
this is cheap.  If uniform-N CPB ~= 0.4284 within tolerance, the SAME sampling is
a validated, unbiased estimator for the expensive quantities (guidance,
framework, arbiter) too -- which is what licenses the whole sampled protocol.

Key point vs your old cache: frames are chosen by np.linspace over the FULL
temporalROI (uniform), NOT the first-N frames (which sit before CPB drifts and
therefore over-state the score).

USAGE
  python validate_protocol.py --models_dir models/cpb_numba_rgb --channel_mode rgb --grid 30,60,100,150
  # faster first look:
  python validate_protocol.py --models_dir models/cpb_numba_rgb --channel_mode rgb --grid 60,100
================================================================================
"""
import argparse, os, glob, re, pickle, json, time
import numpy as np
import cv2
import opus_final_hod_framework as F

# ---- exhaustive full-ROI CPB (RGB, all-64) reference, from your all-frames run ----
FULL_CAT = {
    'baseline': 0.5558, 'cameraJitter': 0.5832, 'badWeather': 0.3786,
    'dynamicBackground': 0.4942, 'intermittentObjectMotion': 0.4481,
    'lowFramerate': 0.3257, 'nightVideos': 0.2920, 'PTZ': 0.0683,
    'shadow': 0.4768, 'thermal': 0.6283, 'turbulence': 0.4617,
}
FULL_OVERALL = 0.4284
FULL_SCENE = {
    'baseline/highway': 0.1105, 'baseline/office': 0.8836, 'baseline/pedestrians': 0.8734, 'baseline/PETS2006': 0.3559,
    'cameraJitter/badminton': 0.6649, 'cameraJitter/boulevard': 0.8437, 'cameraJitter/sidewalk': 0.2790, 'cameraJitter/traffic': 0.5453,
    'badWeather/blizzard': 0.3348, 'badWeather/skating': 0.3469, 'badWeather/snowFall': 0.5382, 'badWeather/wetSnow': 0.2944,
    'dynamicBackground/boats': 0.4908, 'dynamicBackground/canoe': 0.7724, 'dynamicBackground/fall': 0.1746,
    'dynamicBackground/fountain01': 0.1760, 'dynamicBackground/fountain02': 0.6479, 'dynamicBackground/overpass': 0.7037,
    'intermittentObjectMotion/abandonedBox': 0.5329, 'intermittentObjectMotion/parking': 0.3205, 'intermittentObjectMotion/sofa': 0.5604,
    'intermittentObjectMotion/streetLight': 0.8514, 'intermittentObjectMotion/tramstop': 0.3265, 'intermittentObjectMotion/winterDriveway': 0.0967,
    'lowFramerate/port_0_17fps': 0.0041, 'lowFramerate/tramCrossroad_1fps': 0.5763, 'lowFramerate/tunnelExit_0_35fps': 0.3487, 'lowFramerate/turnpike_0_5fps': 0.3736,
    'nightVideos/bridgeEntry': 0.1023, 'nightVideos/busyBoulvard': 0.2680, 'nightVideos/fluidHighway': 0.3083,
    'nightVideos/streetCornerAtNight': 0.3544, 'nightVideos/tramStation': 0.3835, 'nightVideos/winterStreet': 0.3357,
    'PTZ/continuousPan': 0.0325, 'PTZ/intermittentPan': 0.0501, 'PTZ/twoPositionPTZCam': 0.1755, 'PTZ/zoomInZoomOut': 0.0152,
    'shadow/backdoor': 0.5908, 'shadow/bungalows': 0.4927, 'shadow/busStation': 0.7030, 'shadow/copyMachine': 0.4028,
    'shadow/cubicle': 0.1209, 'shadow/peopleInShade': 0.5507,
    'thermal/corridor': 0.7811, 'thermal/diningRoom': 0.8035, 'thermal/lakeSide': 0.3554, 'thermal/library': 0.9166, 'thermal/park': 0.2848,
    'turbulence/turbulence0': 0.0396, 'turbulence/turbulence1': 0.5726, 'turbulence/turbulence2': 0.7567, 'turbulence/turbulence3': 0.4779,
}
CATS = ['baseline', 'cameraJitter', 'badWeather', 'dynamicBackground', 'intermittentObjectMotion',
        'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal', 'turbulence']


def f1_counts(c):
    return F.metrics_from_counts(c[0], c[1], c[2], c[3])['f1']


def gt_frame_numbers(gdir):
    """All groundtruth frame indices present, sorted."""
    nums = []
    for p in glob.glob(os.path.join(gdir, '*')):
        m = re.search(r'(\d+)', os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def roi_range(sp, gdir):
    """temporalROI [start,end] if present, else min/max of gt files."""
    tr = os.path.join(sp, 'temporalROI.txt')
    if os.path.exists(tr):
        try:
            s, e = [int(x) for x in open(tr).read().split()[:2]]
            return s, e
        except Exception:
            pass
    g = gt_frame_numbers(gdir)
    return (g[0], g[-1]) if g else (None, None)


def uniform_nums(s, e, N):
    return sorted(set(int(round(x)) for x in np.linspace(s, e, N)))


def cpb_f1_for_scene(model, sp, nums, channel_mode, af, sigema, bth, ref, floor):
    """Exactly the CPB path used by opus_paper_metrics_table.py: stage3 pipe,
    zero guidance, read pipe.last_s1 (pure CPB stage-1)."""
    idir = os.path.join(sp, 'input'); gdir = os.path.join(sp, 'groundtruth')
    pipe = F.HoDPipeline(model, 'stage3', floor, af, sigema, bth, ref, True, channel_mode)
    cc = [0, 0, 0, 0]; used = 0
    for n in nums:
        bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        zero = np.zeros(gt.shape, np.uint8)
        pipe.run(rgb, zero)
        s1 = pipe.last_s1
        if s1 is None:
            continue
        if s1.shape != gt.shape:
            s1 = cv2.resize(s1, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        cc = [cc[i] + F.frame_counts(s1, gt)[i] for i in range(4)]
        used += 1
    return cc, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--channel_mode', default='rgb', choices=['auto', 'gray', 'rgb'])
    ap.add_argument('--grid', default='30,60,100,150')
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--out', default='protocol_validation.json')
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet2014 not found - use --dataset_root')
    Ns = [int(x) for x in args.grid.split(',') if x.strip()]

    # discover scenes = the ones we have full-ROI reference for (all 53)
    scenes = list(FULL_SCENE.keys())
    results = {N: {} for N in Ns}   # N -> {sid: (f1, used)}
    t0 = time.time()
    for N in Ns:
        print(f"\n{'='*70}\n  UNIFORM N={N} frames/scene\n{'='*70}")
        for sid in scenes:
            cat, scene = sid.split('/', 1)
            sp = os.path.join(root, cat, scene)
            gdir = os.path.join(sp, 'groundtruth')
            mp = os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')
            if not (os.path.isdir(sp) and os.path.exists(mp)):
                print(f'   [skip] {sid} (missing)'); continue
            s, e = roi_range(sp, gdir)
            if s is None:
                print(f'   [skip] {sid} (no ROI)'); continue
            nums = uniform_nums(s, e, N)
            model = pickle.load(open(mp, 'rb'))
            cc, used = cpb_f1_for_scene(model, sp, nums, args.channel_mode,
                                        args.af, args.sigema, args.bth_pixels, args.ref, args.floor_min)
            f1 = f1_counts(cc) if used else 0.0
            results[N][sid] = (f1, used)
            ref = FULL_SCENE[sid]
            print(f'   {sid:42s} N={used:3d}  uni={f1:.3f}  full={ref:.3f}  d={f1-ref:+.3f}  ({time.time()-t0:.0f}s)')

    # ---- summary table ----
    print(f"\n{'#'*72}\n  CONVERGENCE TO EXHAUSTIVE FULL-ROI CPB (0.4284 category-mean)\n{'#'*72}")
    hdr = f"  {'metric':<26}" + ''.join(f"{('N='+str(N)):>11}" for N in Ns) + f"{'FULL-ROI':>11}"
    print(hdr); print('  ' + '-' * (len(hdr) - 2))

    def catmean(N):
        by = {}
        for sid, (f1, u) in results[N].items():
            by.setdefault(sid.split('/')[0], []).append(f1)
        cv = {c: float(np.mean(v)) for c, v in by.items()}
        return cv, float(np.mean(list(cv.values()))) if cv else 0.0

    catmeans = {N: catmean(N) for N in Ns}
    # overall row
    row = f"  {'category-mean F1':<26}" + ''.join(f"{catmeans[N][1]:>11.4f}" for N in Ns) + f"{FULL_OVERALL:>11.4f}"
    print(row)
    # abs error to full overall
    row = f"  {'|delta| vs full-ROI':<26}" + ''.join(f"{abs(catmeans[N][1]-FULL_OVERALL):>11.4f}" for N in Ns) + f"{0.0:>11.4f}"
    print(row)
    # per-scene MAE
    def mae(N):
        errs = [abs(f1 - FULL_SCENE[sid]) for sid, (f1, u) in results[N].items()]
        return float(np.mean(errs)) if errs else 0.0
    row = f"  {'per-scene MAE':<26}" + ''.join(f"{mae(N):>11.4f}" for N in Ns) + f"{0.0:>11.4f}"
    print(row)

    print(f"\n  per-category category-mean F1:")
    print(f"  {'category':<26}" + ''.join(f"{('N='+str(N)):>11}" for N in Ns) + f"{'FULL':>11}")
    for c in CATS:
        vals = ''.join(f"{catmeans[N][0].get(c, float('nan')):>11.4f}" for N in Ns)
        print(f"  {c:<26}{vals}{FULL_CAT[c]:>11.4f}")

    print(f"\n  VERDICT GUIDE: pick the smallest N whose |delta vs full-ROI| <= ~0.02")
    print(f"  and per-scene MAE <= ~0.04. That N is your validated protocol size.")

    json.dump({'grid': Ns, 'full_overall': FULL_OVERALL,
               'catmean': {N: catmeans[N][1] for N in Ns},
               'per_scene': {N: {sid: results[N][sid][0] for sid in results[N]} for N in Ns}},
              open(args.out, 'w'), indent=2)
    print(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()