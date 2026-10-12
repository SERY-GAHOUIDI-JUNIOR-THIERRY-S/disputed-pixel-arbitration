#!/usr/bin/env python3
"""
QUALITATIVE FIGURE WITH THE +AGENT COLUMN — RGB vs grayscale comparison.

Six columns:  Input | Ground truth | CPB (stage-1) | Guidance | Framework (fixed) | + Agent (ours)
Two rows per frame: RGB CPB and grayscale CPB, so the figure IS the RGB-vs-grayscale
comparison and shows the learned arbiter's effect directly.

Unlike the 5-column make_qualitative.py (which used F.HoDPipeline), this script builds
CPB / Framework / Agent from the SAME DRL env path that produced the paper's numbers
(opus_final_project_metrics.Scene + the trained gate), so the agent column is exactly
what was scored, and all three mask columns share one disputed-pixel set.

HONESTY: for each scene it uses the agent from the CV fold where that scene was
HELD OUT (never trained on it). Point --rgb_ckpt / --gray_ckpt at the CV directory
(e.g. models/drl_gate_rgb_cv) and it auto-picks fold{k}/best.pth; or pass a direct
best.pth file to force a specific checkpoint.

RUN (DeepLab guidance):
  python opus_make_qualitative_agent.py --scene dynamicBackground/canoe --frames 998 ^
      --rgb_models models/cpb_numba_rgb --gray_models models/cpb_numba ^
      --cache_dir drl_cache ^
      --rgb_ckpt models/drl_gate_rgb_cv --gray_ckpt models/drl_gate_gray_cv ^
      --out fig_qualitative_agent_deeplab.png

RUN (PSPNet guidance): same but --cache_dir drl_cache_pspnet, the PSPNet CV dirs,
and --out fig_qualitative_agent_pspnet.png
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
    use_patch = ck.get('use_patch', False)
    patch_k = ck.get('patch_k', 4)
    gate = T.build_gate(ck['state_dim'], ck.get('n_actions', T.N_ACTIONS), 0.0, use_patch, patch_k)
    gate.load_state_dict(ck['gate'])
    gate.eval()
    norm = T.RunningNorm(ENV.FrameEnv.STATE_DIM)
    norm.mean = ck['norm_mean']; norm.M2 = ck['norm_M2']; norm.n = ck['norm_n']
    norm.frozen = True
    return gate, norm, use_patch, patch_k


def resolve_ckpt(spec, sid, cv_folds):
    """spec is a direct best.pth OR a CV dir -> auto-pick the held-out fold for sid."""
    if os.path.isfile(spec):
        return spec, None
    if not os.path.isdir(spec):
        raise SystemExit(f'--ckpt "{spec}" is neither a file nor a directory')
    folds = T.make_kfold(cv_folds, T.SEED)
    held = next((k for k, f in enumerate(folds) if sid in f), None)
    if held is None:
        raise SystemExit(f'{sid} not found in any of the {cv_folds} CV folds '
                         f'(check the scene name / cv_folds)')
    cand = os.path.join(spec, f'fold{held}', 'best.pth')
    if not os.path.isfile(cand):
        raise SystemExit(f'no checkpoint {cand} — expected the held-out fold ({held}) for {sid}. '
                         f'Pass a direct best.pth with --{"rgb" if "rgb" in spec else "gray"}_ckpt instead.')
    return cand, held


def fit_to(m, gt):
    if m is None:
        return None
    if m.shape != gt.shape:
        m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def rep_masks(models_dir, root, cat, scene, cache_dir, win_size, ckpt_spec, cv_folds, req_frames):
    sid = f'{cat}/{scene}'
    envargs = dict(af=AF, floor=FLOOR, bth=BTH, ref=REF, sigema=SIGEMA)
    sc = T.Scene(sid, models_dir, root, envargs, cache_dir, win_size)
    nums = [int(n) for n in sc.nums]
    # snap requested frames to the nearest cached frame
    snapped = sorted(set(min(nums, key=lambda n: abs(n - f)) for f in req_frames))

    ckpt, held = resolve_ckpt(ckpt_spec, sid, cv_folds)
    gate, norm, use_patch, patch_k = load_agent(ckpt)
    print(f'  [{models_dir}] {sid}  ckpt={ckpt}'
          + (f'  (held-out fold {held})' if held is not None else '  (explicit file)'))

    out = {}
    for i, n in enumerate(nums):
        if n not in snapped:
            continue
        e = sc.env(i)
        gt = T.load_gt(sc.gdir, n)
        bgr = T.load_bgr(sc.idir, n)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r, c = e.dispute_ij()
        base = e.baseline_mask()                      # the fixed framework (Liang static)
        if len(r) == 0:
            agent = base; mix = (0, 0, 0)
        else:
            Sn = norm(e.state(r, c))
            Pt = torch.from_numpy(e.patches(r, c, patch_k)) if use_patch else None
            with torch.no_grad():
                logits = gate(torch.from_numpy(Sn), Pt)
                actions = logits.argmax(-1).numpy()
            agent = T._segment_gate3(e, actions, r, c)   # exact mask the paper scored
            mix = (int((actions == 0).sum()), int((actions == 1).sum()), int((actions == 2).sum()))
        try:
            cpb = e._to_native(e.s1_default.copy())
        except Exception:
            cpb = e.s1_default.copy()
        guid = (sc.guid[i] > 127).astype(np.uint8) * 255
        out[n] = dict(input=rgb, gt=gt, cpb=fit_to(cpb, gt), guid=fit_to(guid, gt),
                      fw=fit_to(base, gt), agent=fit_to(agent, gt), ndisp=len(r), mix=mix)
        print(f'      frame {n}: disputed={len(r)}  action C/G/B={mix}')
    return out, snapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='cat/scene, e.g. dynamicBackground/canoe')
    ap.add_argument('--frames', type=int, nargs='*', default=None)
    ap.add_argument('--rgb_models', default='models/cpb_numba_rgb')
    ap.add_argument('--gray_models', default='models/cpb_numba')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--rgb_ckpt', required=True, help='RGB agent: a best.pth OR a CV dir (auto held-out fold)')
    ap.add_argument('--gray_ckpt', required=True, help='grayscale agent: a best.pth OR a CV dir')
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--out', default='fig_qualitative_agent.png')
    args = ap.parse_args()

    root = args.dataset_root or T.find_root()
    if not root:
        raise SystemExit('CDnet2014 root not found — pass --dataset_root PATH')
    cat, scene = args.scene.split('/', 1)
    cache_path = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
    if not os.path.exists(cache_path):
        raise SystemExit(f'no cache {cache_path} — check --cache_dir / scene name')
    nums = [int(n) for n in np.load(cache_path)['nums']]
    print(f'cache has {len(nums)} frames: {nums[0]}..{nums[-1]}')
    if not args.frames:
        args.frames = [nums[len(nums) // 3], nums[2 * len(nums) // 3]]

    reps = [('RGB', args.rgb_models, args.rgb_ckpt), ('Grayscale', args.gray_models, args.gray_ckpt)]
    data = {}
    for label, mdir, ckspec in reps:
        if not os.path.exists(os.path.join(mdir, f'cpb_paper_{cat}_{scene}.pkl')):
            print(f'  [skip] {label}: no CPB model cpb_paper_{cat}_{scene}.pkl in {mdir}'); continue
        data[label], used = rep_masks(mdir, root, cat, scene, args.cache_dir,
                                      args.win_size, ckspec, args.cv_folds, args.frames)
        print(f'  {label}: rendered {len(data[label])} frames {used}')

    cols = ['input', 'gt', 'cpb', 'guid', 'fw', 'agent']
    titles = ['Input', 'Ground truth', 'CPB (stage-1)', 'Guidance', 'Framework (fixed)', '+ Agent (ours)']
    rows = [(lab, n) for lab in data for n in sorted({n for d in data.values() for n in d}) if n in data[lab]]
    if not rows:
        raise SystemExit('nothing rendered — check models / cache / checkpoints / scene')

    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.05 * len(cols), 2.05 * len(rows)))
    ax = np.atleast_2d(ax)
    for rr, (lab, n) in enumerate(rows):
        d = data[lab][n]
        for cc, key in enumerate(cols):
            a = ax[rr, cc]; a.set_xticks([]); a.set_yticks([])
            img = d[key]
            if key == 'input':
                a.imshow(img)
            else:
                a.imshow(img, cmap='gray', vmin=0, vmax=255)
            if rr == 0:
                a.set_title(titles[cc], fontsize=10)
            if cc == 0:
                a.set_ylabel(f'{lab}\nframe {n}', fontsize=9)
            if key == 'agent' and d['ndisp'] > 0:
                a.set_xlabel(f'disp={d["ndisp"]}  C/G/B={d["mix"]}', fontsize=7)
    fig.suptitle(f'{args.scene}  —  RGB vs grayscale, fixed framework vs learned agent '
                 f'({os.path.basename(args.cache_dir)} guidance)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
