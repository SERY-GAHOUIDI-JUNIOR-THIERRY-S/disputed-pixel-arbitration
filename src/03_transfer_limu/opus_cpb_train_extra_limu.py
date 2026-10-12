#!/usr/bin/env python3
"""
================================================================================
CPB TRAINING for LIMU / WallFlower  (thin wrapper over your existing trainer)
================================================================================
Reuses the EXACT Pearson CPB kernel from your opus_final_hod_training.py — same
math, same .pkl format — just pointed at the LIMU / WallFlower scenes with the
paper's frame budget:
    LIMU        : first 400 frames for training   (paper §4.1)
    WallFlower  : provided 200 frames for training (paper §4.1)

REQUIREMENT: the datasets are normalized to CDnet layout under the SAME data root
as cdnet2014, i.e.
    <data_root>/limu/<scene>/{input, groundtruth, temporalROI.txt}
    <data_root>/wallflower/<scene>/{input, groundtruth, temporalROI.txt}
with temporalROI.txt written so the TRAINING frames are the pre-ROI range
(frames < roi_start) — then the reused pre_roi_nums() picks them automatically.
(The converter — built once we see your raw layout — writes exactly this.)

USAGE
  python cpb_train_extra.py --dataset limu       --output_dir models/cpb_limu
  python cpb_train_extra.py --dataset wallflower --output_dir models/cpb_wallflower
  # native-resolution (no 250k downsample), if you decide to:
  python cpb_train_extra.py --dataset limu --output_dir models/cpb_limu --max_train_pixels 100000000 --force
================================================================================
"""
import argparse, os, sys
from types import SimpleNamespace

CPB = None
for _n in ('opus_final_hod_training', 'opus_cpb_training', 'cpb_training'):
    try:
        CPB = __import__(_n); break
    except Exception:
        continue
if CPB is None:
    sys.exit('Could not import your CPB training module (expected opus_final_hod_training.py).')

# cat/scene registries (cat is the sub-folder name under the data root)
SCENES = {
    'limu': {'limu': ['CameraParameter', 'Intersection', 'LightSwitch']},
    'wallflower': {'wallflower': ['Bootstrap', 'Camouflage', 'ForegroundAperture',
                                  'LightSwitch', 'MovedObject', 'TimeOfDay', 'WavingTrees']},
}
DEFAULT_NTRAIN = {'limu': 400, 'wallflower': 200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['limu', 'wallflower'])
    ap.add_argument('--dataset_root', default=None, help='defaults to the cdnet2014 folder')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--channels', choices=['gray', 'rgb'], default='gray')
    ap.add_argument('--n_train', type=int, default=None, help='override; else 400 (limu) / 200 (wallflower)')
    ap.add_argument('--poolnum', type=int, default=CPB.POOLNUM)
    ap.add_argument('--min_gamma', type=float, default=CPB.MIN_GAMMA_DEFAULT)
    ap.add_argument('--max_train_pixels', type=int, default=CPB.MAX_TRAINING_PIXELS)
    ap.add_argument('--no_calib_floor', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = args.dataset_root or CPB.find_root()
    if not root:
        sys.exit('Data root not found — pass --dataset_root pointing at the cdnet2014 folder.')
    n_train = args.n_train if args.n_train is not None else DEFAULT_NTRAIN[args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    # args namespace exactly as CPB.train_scene expects
    sa = SimpleNamespace(channels=args.channels, n_train=n_train, poolnum=args.poolnum,
                         min_gamma=args.min_gamma, max_train_pixels=args.max_train_pixels,
                         no_calib_floor=args.no_calib_floor, force=args.force)

    print('=' * 70)
    print(f'  CPB TRAINING  dataset={args.dataset}  n_train={n_train}  channels={args.channels}')
    print(f'  root={root}  out={args.output_dir}  downsample>{args.max_train_pixels:,}px')
    print('=' * 70)
    tally = {}
    for cat, scs in SCENES[args.dataset].items():
        for sc in scs:
            try:
                st = CPB.train_scene(root, cat, sc, args.output_dir, sa)
            except Exception as e:
                print(f'  [ERROR] {cat}/{sc}: {e}'); st = 'error'
            tally[st] = tally.get(st, 0) + 1
    print('\n' + '=' * 70)
    print('  DONE  ' + '  '.join(f'{k}={v}' for k, v in sorted(tally.items())))
    print('  -> models named  cpb_paper_<dataset>_<scene>.pkl  (same format as CDnet)')
    print('=' * 70)


if __name__ == '__main__':
    main()
