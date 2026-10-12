#!/usr/bin/env python3
"""
REPRODUCTION ERROR-MAP FIGURE (fig_errormaps).

Shows the CPB-alone (unsupervised, guidance-free) error signature on hard scenes:
green = true positive, red = false positive, blue = false negative. The paper's point
is that the reproduced CPB deficit is UNDER-DETECTION (mostly blue) rather than false
alarms (little red) — high precision, collapsed recall.

Uses the verified F.HoDPipeline API with a guidance-free pass (zeros), exactly as the
CPB-alone row of opus_paper_metrics_table.py, so the masks are the pure unsupervised CPB.

RUN (auto-pick the lowest-recall frame per scene):
  python opus_make_errormaps.py --scenes lowFramerate/turnpike_0_5fps baseline/highway ^
      --models_dir models/cpb_numba_rgb --channel_mode rgb --cache_dir drl_cache ^
      --out fig_errormaps.png

Pass --frames N1 N2 to fix the frames. It PRINTS each scene's precision/recall so you
can copy the exact numbers into the caption.
"""
import argparse, os, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opus_final_hod_framework as F

AF, SIGEMA, BTH, REF, FLOOR = 2.5, 0.8, 7.0, 20, 6.0


def cdnet_masks(gt):
    fg = (gt == 255)
    bg = (gt == 0) | (gt == 50)       # hard shadow counts as background
    valid = fg | bg                   # 85 (outside ROI) and 170 (unknown) excluded
    return fg, bg, valid


def cpb_scene(models_dir, channel_mode, root, cat, scene, cache_path, target):
    model = pickle.load(open(os.path.join(models_dir, f'cpb_paper_{cat}_{scene}.pkl'), 'rb'))
    nums = [int(n) for n in np.load(cache_path)['nums']]
    idir = os.path.join(root, cat, scene, 'input')
    gdir = os.path.join(root, cat, scene, 'groundtruth')
    pipe = F.HoDPipeline(model, 'stage3', FLOOR, AF, SIGEMA, BTH, REF, True, channel_mode)
    chosen = None
    for n in nums:
        bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pipe.run(rgb, np.zeros(gt.shape, np.uint8))          # guidance-free -> pure CPB
        s1 = pipe.last_s1
        if s1 is None:
            continue
        if s1.shape != gt.shape:
            s1 = cv2.resize(s1, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred = s1 > 127
        fg, bg, valid = cdnet_masks(gt)
        tp = int((pred & fg & valid).sum()); fp = int((pred & bg & valid).sum())
        fn = int((~pred & fg & valid).sum())
        P = tp / (tp + fp + 1e-9); R = tp / (tp + fn + 1e-9)
        rec = dict(n=n, rgb=rgb, gt=gt, pred=pred, fg=fg, bg=bg, valid=valid,
                   P=P, R=R, nfg=int((fg & valid).sum()))
        if target is not None:
            if n == target:
                return rec
        else:
            # auto-pick the lowest-recall frame that actually has foreground
            if rec['nfg'] > 80 and (chosen is None or R < chosen['R']):
                chosen = rec
    return chosen


def error_map(rec):
    H, W = rec['gt'].shape
    em = np.zeros((H, W, 3), np.uint8)
    em[rec['valid']] = (45, 45, 45)                          # TN: dark gray
    tp = rec['pred'] & rec['fg'] & rec['valid']
    fp = rec['pred'] & rec['bg'] & rec['valid']
    fn = (~rec['pred']) & rec['fg'] & rec['valid']
    em[tp] = (70, 200, 90)                                   # green  TP
    em[fp] = (225, 70, 45)                                   # red    FP
    em[fn] = (70, 130, 235)                                  # blue   FN
    return em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', required=True, help='cat/scene list, e.g. lowFramerate/turnpike_0_5fps baseline/highway')
    ap.add_argument('--frames', type=int, nargs='*', default=None)
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--channel_mode', default='rgb', choices=['rgb', 'gray'])
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--out', default='fig_errormaps.png')
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet2014 root not found — pass --dataset_root PATH')

    recs = []
    for si, sp in enumerate(args.scenes):
        cat, scene = sp.split('/', 1)
        cache_path = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
        if not os.path.exists(cache_path):
            print(f'  [skip] no cache {cache_path}'); continue
        if not os.path.exists(os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')):
            print(f'  [skip] no CPB model for {sp} in {args.models_dir}'); continue
        target = args.frames[si] if (args.frames and si < len(args.frames)) else None
        rec = cpb_scene(args.models_dir, args.channel_mode, root, cat, scene, cache_path, target)
        if rec is None:
            print(f'  [skip] {sp}: no usable frame'); continue
        rec['sid'] = sp
        recs.append(rec)
        print(f'  {sp}  frame {rec["n"]}:  precision={rec["P"]:.3f}  recall={rec["R"]:.3f}  '
              f'(green=TP red=FP blue=FN)')
    if not recs:
        raise SystemExit('nothing rendered — check scenes / models_dir / cache_dir')

    cols = ['input', 'cpb', 'err']
    titles = ['Input', 'CPB (unsupervised)', 'Error map  (green TP / red FP / blue FN)']
    fig, ax = plt.subplots(len(recs), 3, figsize=(3 * 3.1, len(recs) * 2.15))
    ax = np.atleast_2d(ax)
    for r, rec in enumerate(recs):
        cpb_vis = (rec['pred'].astype(np.uint8)) * 255
        panels = [rec['rgb'], cpb_vis, error_map(rec)]
        for c in range(3):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            if c == 0 or c == 2:
                a.imshow(panels[c])
            else:
                a.imshow(panels[c], cmap='gray', vmin=0, vmax=255)
            if r == 0:
                a.set_title(titles[c], fontsize=10)
        ax[r, 0].set_ylabel(f'{rec["sid"].split("/")[0]}\nframe {rec["n"]}', fontsize=9)
        ax[r, 2].set_xlabel(f'P={rec["P"]:.3f}  R={rec["R"]:.3f}', fontsize=8)
    fig.suptitle('Reproduced CPB deficit is under-detection (mostly blue), not false alarms',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
