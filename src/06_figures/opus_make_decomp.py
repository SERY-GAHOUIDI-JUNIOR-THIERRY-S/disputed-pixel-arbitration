#!/usr/bin/env python3
"""
PER-STAGE DECOMPOSITION ON A REAL FRAME (fig_decomp / fig:alive).

One row, seven panels, for a single frame:
  Input | CPB (s_cpb) | Guidance (s_g) | Fuse | Framework (fixed) | + Agent (ours) | GT
where Fuse overlays, on a dimmed input: AGREED-foreground (white, both branches say fg)
and the DISPUTED set D (yellow, the branches disagree — the only pixels the agent acts on).

Built from the same DRL env path that produced the paper's numbers
(opus_final_project_metrics.Scene + the trained gate, held-out fold for the scene), so
the dispute set, the framework mask and the agent mask are mutually consistent.

RUN (RGB / DeepLab, canoe):
  python opus_make_decomp.py --scene dynamicBackground/canoe --frame 998 ^
      --models_dir models/cpb_numba_rgb --cache_dir drl_cache ^
      --ckpt models/drl_gate_rgb_cv --out fig_decomp.png
"""
import argparse, os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import opus_final_project_metrics as T
import opus_drl_env_junior_enhanced_with_patch_cnn as ENV

AF, SIGEMA, BTH, REF, FLOOR = 2.5, 0.8, 7.0, 20, 6.0


def load_agent(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    use_patch = ck.get('use_patch', False); patch_k = ck.get('patch_k', 4)
    gate = T.build_gate(ck['state_dim'], ck.get('n_actions', T.N_ACTIONS), 0.0, use_patch, patch_k)
    gate.load_state_dict(ck['gate']); gate.eval()
    norm = T.RunningNorm(ENV.FrameEnv.STATE_DIM)
    norm.mean = ck['norm_mean']; norm.M2 = ck['norm_M2']; norm.n = ck['norm_n']; norm.frozen = True
    return gate, norm, use_patch, patch_k


def resolve_ckpt(spec, sid, cv_folds):
    if os.path.isfile(spec):
        return spec, None
    if not os.path.isdir(spec):
        raise SystemExit(f'--ckpt "{spec}" is neither a file nor a directory')
    folds = T.make_kfold(cv_folds, T.SEED)
    held = next((k for k, f in enumerate(folds) if sid in f), None)
    if held is None:
        raise SystemExit(f'{sid} not in any of the {cv_folds} folds')
    cand = os.path.join(spec, f'fold{held}', 'best.pth')
    if not os.path.isfile(cand):
        raise SystemExit(f'no checkpoint {cand} (held-out fold {held} for {sid})')
    return cand, held


def rz(m, gt, interp=cv2.INTER_NEAREST):
    if m is None:
        return None
    if m.shape[:2] != gt.shape[:2]:
        m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=interp)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--ckpt', required=True, help='a best.pth OR a CV dir (auto held-out fold)')
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--out', default='fig_decomp.png')
    args = ap.parse_args()

    root = args.dataset_root or T.find_root()
    if not root:
        raise SystemExit('CDnet2014 root not found — pass --dataset_root')
    cat, scene = args.scene.split('/', 1)
    sid = f'{cat}/{scene}'
    cache_path = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
    if not os.path.exists(cache_path):
        raise SystemExit(f'no cache {cache_path}')

    envargs = dict(af=AF, floor=FLOOR, bth=BTH, ref=REF, sigema=SIGEMA)
    sc = T.Scene(sid, args.models_dir, root, envargs, args.cache_dir, args.win_size)
    nums = [int(n) for n in sc.nums]
    frame = args.frame if args.frame is not None else nums[len(nums) // 2]
    frame = min(nums, key=lambda n: abs(n - frame))
    i = nums.index(frame)

    ckpt, held = resolve_ckpt(args.ckpt, sid, args.cv_folds)
    gate, norm, use_patch, patch_k = load_agent(ckpt)
    print(f'{sid} frame {frame}  ckpt={ckpt}' + (f' (held-out fold {held})' if held is not None else ''))

    e = sc.env(i)
    gt = T.load_gt(sc.gdir, frame)
    rgb = cv2.cvtColor(T.load_bgr(sc.idir, frame), cv2.COLOR_BGR2RGB)
    r, c = e.dispute_ij()
    base = e.baseline_mask()
    if len(r) == 0:
        agent = base; mix = (0, 0, 0)
    else:
        Sn = norm(e.state(r, c))
        Pt = torch.from_numpy(e.patches(r, c, patch_k)) if use_patch else None
        with torch.no_grad():
            actions = gate(torch.from_numpy(Sn), Pt).argmax(-1).numpy()
        agent = T._segment_gate3(e, actions, r, c)
        mix = (int((actions == 0).sum()), int((actions == 1).sum()), int((actions == 2).sum()))
    print(f'  disputed={len(r)}  action C/G/B={mix}')

    cpb_fit = e.s1_default
    guid_fit = e.guid_fg
    disp_fit = np.zeros(cpb_fit.shape, bool); disp_fit[r, c] = True
    agreed_fit = (cpb_fit > 0) & (guid_fit > 0)
    disp_n = rz(disp_fit.astype(np.uint8), gt).astype(bool)
    agreed_n = rz(agreed_fit.astype(np.uint8), gt).astype(bool)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fuse = (np.stack([gray] * 3, -1) * 0.55).astype(np.uint8)
    fuse[agreed_n] = (255, 255, 255)          # agreed foreground
    fuse[disp_n] = (255, 215, 0)              # disputed set D (yellow)

    try:
        cpb_vis = e._to_native(e.s1_default.copy())
    except Exception:
        cpb_vis = e.s1_default.copy()
    cpb_vis = rz(cpb_vis, gt)
    guid_vis = rz((guid_fit > 0).astype(np.uint8) * 255, gt)
    base_vis = rz(base, gt)
    agent_vis = rz(agent, gt)

    panels = [('Input', rgb, 'rgb'),
              ('CPB $s_{cpb}$', cpb_vis, 'g'),
              ('Guidance $s_g$', guid_vis, 'g'),
              ('Fuse: agreed (white) + disputed $\\mathcal{D}$ (yellow)', fuse, 'rgb'),
              ('Framework (fixed)', base_vis, 'g'),
              ('+ Agent (ours)', agent_vis, 'g'),
              ('Ground truth', gt, 'g')]

    fig, ax = plt.subplots(1, len(panels), figsize=(len(panels) * 2.0, 2.4))
    for a, (title, img, kind) in zip(ax, panels):
        a.set_xticks([]); a.set_yticks([])
        if kind == 'rgb':
            a.imshow(img)
        else:
            a.imshow(img, cmap='gray', vmin=0, vmax=255)
        a.set_title(title, fontsize=9)
    ax[5].set_xlabel(f'disputed={len(r)}  C/G/B={mix}', fontsize=7)
    fig.suptitle(f'Per-stage decomposition — {sid} (frame {frame})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
