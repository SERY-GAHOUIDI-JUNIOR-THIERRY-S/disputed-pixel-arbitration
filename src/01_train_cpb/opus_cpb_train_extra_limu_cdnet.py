#!/usr/bin/env python3
"""
================================================================================
CPB TRAINING for CDnet2014  (channels-aware) — same core as cpb_train_extra.py
================================================================================
This is the CDnet twin of your working cpb_train_extra.py. It reuses the EXACT
channels-aware CPB core (opus_final_hod_training) that produced your RGB LIMU
models (models/cpb_limu_rgb) — same math, same .pkl format — now pointed at the
CDnet2014 scenes with the paper's first-400-frame training budget.

Because ONE core trains both modes, you can build a GRAY and an RGB CDnet model
set that differ ONLY in channels, so the grayscale-vs-RGB comparison is
controlled by construction (no confound from two different trainers).

REQUIREMENT: CDnet2014 on disk as  <data_root>/<category>/<scene>/{input, groundtruth, temporalROI.txt}
(the same root your other scripts use). The models are saved as
cpb_paper_<category>_<scene>.pkl — identical naming to your existing cpb_numba,
so the framework / env / ablation pick them up with just --models_dir.

USAGE
  # ---- PILOT: colour-rich categories, RGB and a matching GRAY (recommended) ----
  python cpb_train_extra_cdnet.py --channels rgb  --category dynamicBackground --output_dir models/cpb_numba_rgb
  python cpb_train_extra_cdnet.py --channels gray --category dynamicBackground --output_dir models/cpb_numba_gray
  #   (repeat for --category baseline and --category cameraJitter for the pilot)

  # ---- a single scene ----
  python cpb_train_extra_cdnet.py --channels rgb --category baseline --scene highway --output_dir models/cpb_numba_rgb

  # ---- FULL CDnet in RGB (heavy; add a matching gray run with --channels gray) ----
  python cpb_train_extra_cdnet.py --channels rgb  --output_dir models/cpb_numba_rgb
  python cpb_train_extra_cdnet.py --channels gray --output_dir models/cpb_numba_gray
================================================================================
"""
import argparse, os, sys
from types import SimpleNamespace

# --- import the SAME channels-aware CPB core that cpb_train_extra.py uses ---
CPB = None
for _n in ('opus_final_hod_training', 'opus_cpb_training', 'cpb_training'):
    try:
        CPB = __import__(_n); break
    except Exception:
        continue
if CPB is None:
    sys.exit('Could not import your CPB training core (expected opus_final_hod_training.py — '
             'the SAME module cpb_train_extra.py imports). Run this from the project folder.')

# all 11 CDnet-2014 categories (category name == on-disk sub-folder)
CDNET2014 = {
    'baseline': ['highway', 'office', 'pedestrians', 'PETS2006'],
    'cameraJitter': ['badminton', 'boulevard', 'sidewalk', 'traffic'],
    'badWeather': ['blizzard', 'skating', 'snowFall', 'wetSnow'],
    'dynamicBackground': ['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
    'intermittentObjectMotion': ['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
    'lowFramerate': ['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
    'nightVideos': ['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
    'PTZ': ['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
    'shadow': ['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
    'thermal': ['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
    'turbulence': ['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', default=None, help='defaults to CPB.find_root() (the cdnet2014 folder)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--channels', choices=['gray', 'rgb'], default='rgb')
    ap.add_argument('--category', default=None, help='limit to one CDnet category (for the pilot)')
    ap.add_argument('--scene', default=None, help='limit to one scene within --category')
    ap.add_argument('--n_train', type=int, default=400, help='first-N frames for CPB training (Liang et al.: 400)')
    ap.add_argument('--poolnum', type=int, default=getattr(CPB, 'POOLNUM', 30))
    ap.add_argument('--min_gamma', type=float, default=getattr(CPB, 'MIN_GAMMA_DEFAULT', 0.0))
    ap.add_argument('--max_train_pixels', type=int, default=getattr(CPB, 'MAX_TRAINING_PIXELS', 250000))
    ap.add_argument('--no_calib_floor', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = args.dataset_root or CPB.find_root()
    if not root:
        sys.exit('Data root not found — pass --dataset_root pointing at the cdnet2014 folder.')
    if args.scene and not args.category:
        sys.exit('--scene requires --category.')
    if args.category and args.category not in CDNET2014:
        sys.exit(f'unknown category {args.category!r}. Choices: {", ".join(CDNET2014)}')
    os.makedirs(args.output_dir, exist_ok=True)

    # which scenes to train
    if args.category:
        scenes = [args.scene] if args.scene else CDNET2014[args.category]
        plan = {args.category: scenes}
    else:
        plan = CDNET2014

    # args namespace EXACTLY as CPB.train_scene expects (mirrors cpb_train_extra.py)
    sa = SimpleNamespace(channels=args.channels, n_train=args.n_train, poolnum=args.poolnum,
                         min_gamma=args.min_gamma, max_train_pixels=args.max_train_pixels,
                         no_calib_floor=args.no_calib_floor, force=args.force)

    print('=' * 74)
    print(f'  CPB TRAINING (CDnet2014)  channels={args.channels}  n_train={args.n_train}')
    print(f'  root={root}  out={args.output_dir}  downsample>{args.max_train_pixels:,}px')
    print('=' * 74)
    tally = {}
    for cat, scs in plan.items():
        for sc in scs:
            try:
                st = CPB.train_scene(root, cat, sc, args.output_dir, sa)
            except Exception as e:
                print(f'  [ERROR] {cat}/{sc}: {e}'); st = 'error'
            key = st if isinstance(st, str) else (st.get('status', 'done') if isinstance(st, dict) else str(st))
            tally[key] = tally.get(key, 0) + 1
    print('\n' + '=' * 74)
    print('  DONE  ' + '  '.join(f'{k}={v}' for k, v in sorted(tally.items())))
    print('  -> models named  cpb_paper_<category>_<scene>.pkl  (same format as your cpb_numba)')
    print('=' * 74)


if __name__ == '__main__':
    main()