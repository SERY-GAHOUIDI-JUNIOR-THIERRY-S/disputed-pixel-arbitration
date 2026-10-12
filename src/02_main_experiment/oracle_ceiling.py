#!/usr/bin/env python3
"""
ORACLE CEILING on disputed pixels — the Table-3 oracle row, computed with the
SAME env, dispute set, and scoring as the DRL CV so it is directly comparable.

On each disputed pixel the oracle picks the primitive side (CPB or guidance) that
matches ground truth (exactly one does, since the two branches disagree there);
agreed pixels keep the framework's own resolution. This is the achievable upper
bound of ANY arbiter that only touches disputed pixels.

Run (RGB, DeepLab cache):
  python oracle_ceiling.py --models_dir models/cpb_numba_rgb --cache_dir drl_cache --out oracle_deeplab.json
  python oracle_ceiling.py --models_dir models/cpb_numba_rgb --cache_dir drl_cache_pspnet --out oracle_pspnet.json
"""
import argparse, json
import numpy as np
import opus_final_project_metrics as T


def agg(scene_counts, cat_of):
    tot = [0, 0, 0, 0]
    for c in scene_counts.values():
        tot = [tot[k] + c[k] for k in range(4)]
    pooled = T.ENV.f1_from_counts(*tot)
    sf = [T.ENV.f1_from_counts(*c) for c in scene_counts.values()]
    scene_mean = float(np.mean(sf)) if sf else 0.0
    by = {}
    for sid, c in scene_counts.items():
        by.setdefault(cat_of[sid], []).append(T.ENV.f1_from_counts(*c))
    cat = {k: float(np.mean(v)) for k, v in by.items()}
    return {'pooled': pooled, 'scene_mean': scene_mean,
            'category_mean': float(np.mean(list(cat.values()))) if cat else 0.0,
            'per_cat': cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--out', default='oracle_ceiling.json')
    args = ap.parse_args()
    args.use_patch = False; args.patch_k = 4

    root = T.find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels,
                   ref=args.ref, sigema=args.sigema)
    ids = [s for v in T.make_split().values() for s in v]          # all 53 scenes
    print(f'building {len(ids)} scenes (in-order CPB pass) ...')
    scenes = T.build_scenes(ids, args, root, envargs)

    oc = {}; bc = {}; cat_of = {}
    for si, sc in enumerate(scenes):
        cat = sc.sid.split('/')[0]
        o = [0, 0, 0, 0]; b = [0, 0, 0, 0]
        for i in range(len(sc.nums)):
            e = sc.env(i); r, c = e.dispute_ij(); base = e.baseline_mask()
            if len(r) == 0:
                omask = base
            else:
                cpb_lab = e.s1_default[r, c] > 0
                gt = e.gtfg_fit[r, c]
                actions = np.where(cpb_lab == gt, 0, 1)   # 0=CPB, 1=guidance (GT-matching side)
                omask = T._segment_gate3(e, actions, r, c)
            o = [o[k] + v for k, v in enumerate(e.counts(omask))]
            b = [b[k] + v for k, v in enumerate(e.counts(base))]
        oc[sc.sid] = o; bc[sc.sid] = b; cat_of[sc.sid] = cat
        print(f'  [{si+1}/{len(scenes)}] {sc.sid:34s} base={T.ENV.f1_from_counts(*b):.3f}  '
              f'oracle={T.ENV.f1_from_counts(*o):.3f}')

    O = agg(oc, cat_of); B = agg(bc, cat_of)
    print('\n' + '=' * 66)
    print(f'  ORACLE CEILING  (cache={args.cache_dir}, {len(cat_of)} scenes)')
    print('=' * 66)
    print(f"  {'aggregation':<20}{'framework':>12}{'oracle':>12}{'head-room':>12}")
    for key in ('pooled', 'scene_mean', 'category_mean'):
        print(f"  {key:<20}{B[key]:>12.4f}{O[key]:>12.4f}{O[key]-B[key]:>+12.4f}")
    print('=' * 66)
    json.dump({'framework': B, 'oracle': O,
               'per_scene': {sid: {'cat': cat_of[sid],
                                   'framework': T.ENV.f1_from_counts(*bc[sid]),
                                   'oracle': T.ENV.f1_from_counts(*oc[sid])} for sid in cat_of}},
              open(args.out, 'w'), indent=2)
    print(f'  saved -> {args.out}')


if __name__ == '__main__':
    main()
