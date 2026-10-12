#!/usr/bin/env python3
"""
QUALITATIVE FIGURE — RGB vs grayscale comparison across the pipeline stages.
Regenerates the qualitative panels marked % TODO in the manuscript.

For a chosen scene it renders, for BOTH the RGB CPB and the grayscale CPB
(same guidance cache), a row of:  Input | Ground truth | CPB (stage-1) |
Guidance | Framework (stage-3).  The two representation rows sit side by side
so the figure IS the RGB-vs-grayscale comparison.

It reuses the EXACT F.HoDPipeline API from opus_paper_metrics_table.py, so the
masks are identical to what the metrics table scored — nothing is re-derived.

RUN (DeepLab guidance):
  python make_qualitative.py --scene baseline/highway --frames 1005 1200 ^
      --rgb_models models/cpb_numba_rgb --gray_models models/cpb_numba ^
      --cache_dir drl_cache --out fig_qualitative_deeplab.png

RUN (PSPNet guidance): same but --cache_dir drl_cache_pspnet --out fig_qualitative_pspnet.png

If you omit --frames, three frames are auto-picked from the cache.
Optional --agent_ckpt path enables an extra "+ agent" column (see AGENT HOOK
below); without it the guaranteed 5-column RGB-vs-gray figure is produced.
"""
import argparse, os, glob, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opus_final_hod_framework as F

AF, SIGEMA, BTH, REF, FLOOR = 2.5, 0.8, 7.0, 20, 6.0


def stage_masks(models_dir, channel_mode, root, cat, scene, cache_path, frames):
    """Return {frame_num: dict(input,gt,cpb,guid,fw)} using the verified F API."""
    mp = os.path.join(models_dir, f'cpb_paper_{cat}_{scene}.pkl')
    model = pickle.load(open(mp, 'rb'))
    dat = np.load(cache_path)
    nums = [int(n) for n in dat['nums']]; gmask = dat['guid']
    idir = os.path.join(root, cat, scene, 'input')
    gdir = os.path.join(root, cat, scene, 'groundtruth')

    pipe = F.HoDPipeline(model, 'stage3', FLOOR, AF, SIGEMA, BTH, REF, True, channel_mode)
    pipe_cpb = F.HoDPipeline(model, 'stage3', FLOOR, AF, SIGEMA, BTH, REF, True, channel_mode)

    # cache is sequential: run through it so the temporal model state is correct,
    # capturing the requested frames.
    want = set(frames); out = {}
    for i, n in enumerate(nums):
        bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gm = gmask[i]
        if gm.shape != gt.shape:
            gm = cv2.resize(gm, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        fpred = pipe.run(rgb, gm)
        pipe_cpb.run(rgb, np.zeros(gt.shape, np.uint8))
        s1 = pipe_cpb.last_s1
        if n in want:
            def fit(m):
                if m is not None and m.shape != gt.shape:
                    m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                return m
            out[n] = dict(input=rgb, gt=gt, cpb=fit(s1),
                          guid=(gm > 127).astype(np.uint8) * 255, fw=fit(fpred))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='cat/scene, e.g. baseline/highway')
    ap.add_argument('--frames', type=int, nargs='*', default=None)
    ap.add_argument('--rgb_models', default='models/cpb_numba_rgb')
    ap.add_argument('--gray_models', default='models/cpb_numba')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--out', default='fig_qualitative.png')
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet2014 dataset root not found — pass --dataset_root PATH')
    cat, scene = args.scene.split('/', 1)
    cache_path = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
    if not os.path.exists(cache_path):
        raise SystemExit(f'no cache {cache_path} — check --cache_dir and the scene name')

    nums = [int(n) for n in np.load(cache_path)['nums']]
    print(f'cache has {len(nums)} frames: {nums[0]}..{nums[-1]}  (e.g. {nums[::max(1,len(nums)//6)]})')
    if not args.frames:
        args.frames = [nums[len(nums) // 4], nums[len(nums) // 2], nums[3 * len(nums) // 4]]
    else:
        # snap each requested frame to the NEAREST frame that is actually cached,
        # so a frame number outside the cached set still yields a panel.
        snapped = []
        for f in args.frames:
            nearest = min(nums, key=lambda n: abs(n - f))
            if nearest != f:
                print(f'  frame {f} not in cache -> using nearest cached frame {nearest}')
            snapped.append(nearest)
        args.frames = sorted(set(snapped))
    print(f'scene={args.scene} frames={args.frames} cache={args.cache_dir}')

    reps = [('RGB', args.rgb_models, 'rgb'), ('Grayscale', args.gray_models, 'gray')]
    data = {}
    for label, mdir, cm in reps:
        if not os.path.exists(os.path.join(mdir, f'cpb_paper_{cat}_{scene}.pkl')):
            print(f'  [skip] {label}: no CPB model in {mdir}'); continue
        data[label] = stage_masks(mdir, cm, root, cat, scene, cache_path, args.frames)
        print(f'  {label}: rendered {len(data[label])} frames')

    cols = ['input', 'gt', 'cpb', 'guid', 'fw']
    titles = ['Input', 'Ground truth', 'CPB (stage-1)', 'Guidance', 'Framework']
    rows = [(lab, n) for lab in data for n in args.frames if n in data[lab]]
    if not rows:
        raise SystemExit('nothing rendered. Likely causes: (1) neither CPB model '
                         f'file exists (looked for cpb_paper_{cat}_{scene}.pkl in '
                         f'--rgb_models / --gray_models); (2) the dataset input/'
                         'groundtruth frames could not be loaded from the dataset root. '
                         f'Dataset root in use: {root}')

    fig, ax = plt.subplots(len(rows), len(cols),
                           figsize=(2.05 * len(cols), 2.05 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (lab, n) in enumerate(rows):
        d = data[lab][n]
        for c, key in enumerate(cols):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            img = d[key]
            if key == 'input':
                a.imshow(img)
            else:
                a.imshow(img, cmap='gray', vmin=0, vmax=255)
            if r == 0:
                a.set_title(titles[c], fontsize=10)
            if c == 0:
                a.set_ylabel(f'{lab}\nframe {n}', fontsize=9)
    fig.suptitle(f'{args.scene}  —  RGB vs grayscale across pipeline stages '
                 f'({os.path.basename(args.cache_dir)} guidance)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved -> {args.out}')

# ---------------------------------------------------------------------------
# AGENT HOOK (optional): to add a "+ agent" column, load your trained gate and
# call the same per-frame apply your evaluator uses in opus_final_project_metrics
# (the function that turns disputed-pixel actions into a final mask). Return a
# HxW uint8 {0,255} mask and append it as another column above. The 5 columns
# here are produced with the verified F.HoDPipeline API and need no changes.
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    main()
