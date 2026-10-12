#!/usr/bin/env python3
"""
================================================================================
EXPORT FIGURE PANELS  — real-image assets for ALL of the paper's figures
================================================================================
Two engines in one script:

  (A) FRAMEWORK engine  (needs only the CPB pkl + frames + guidance weights)
      Runs the FROZEN framework (opus_final_hod_framework) on one scene+frame and
      saves the per-stage panels, INCLUDING the new "alive" decomposition used for
      the method figure (agree map / dispute map).

  (B) ENV engine  (adds framework+MLP, framework+agent, agent-ACTION map, oracle)
      Reuses the PROVEN helpers of opus_headtohead_ablation.py (same math as the
      reported numbers) to build the scene from the caches, pick the CORRECT
      leave-one-scene-out fold for that scene, load that fold's agent, optionally
      train that fold's 3-way MLP, and render every method's mask + the agent's
      per-disputed-pixel decision map. Enabled with --agent_dir.

It emits, per scene/frame:
    01_input.png            RGB frame
    02_groundtruth.png      CDnet GT (white FG / gray unknown / black BG)
    03_guidance.png         supervised guidance mask s_g (binary)
    04_cpb_stage1.png       CPB Stage-1 label s_cpb (binary)
    05_framework.png        full Stage-1-2-3 output s_fix (Liang's fixed rule)
    06_agree.png            AGREEING pixels (s_cpb==s_g) painted on the frame
    07_dispute.png          DISPUTED pixels (s_cpb!=s_g) painted on the frame  (+ %)
    08_mlp.png              framework + supervised MLP           [ENV engine]
    09_agent.png            framework + DRL agent (ours)         [ENV engine]
    10_agent_actions.png    disputed pixels coloured by the agent's action
                            (blue=trust-CPB, orange=trust-guidance, gray=keep)  [ENV]
    11_oracle.png           per-pixel GT ceiling on disputes     [ENV engine]
  method_strip.png          labelled horizontal strip for the METHOD figure (Fig. B)
  results_row.png           labelled row for the BIG side-by-side results figure
  contact_sheet.png         everything tiled, for a quick look

USAGE
  # 1) list ROI frames, pick a good one (clear moving object):
  python opus_export_figure_panels.py --category cameraJitter --scene badminton \
      --models_dir models/cpb_numba --list

  # 2) FRAMEWORK-only panels (fast, no agent):
  python opus_export_figure_panels.py --category cameraJitter --scene badminton \
      --models_dir models/cpb_numba --guidance deeplab --frame -1 --out fig_panels/cjt

  # 3) FULL panels incl. agent/MLP/oracle + action map (the good stuff):
  python opus_export_figure_panels.py --category cameraJitter --scene badminton \
      --models_dir models/cpb_numba --cache_dir drl_cache_junior \
      --agent_dir models/drl_gate_cv_3way --with_mlp --frame -1 --out fig_panels/cjt

  # good scenes:  cameraJitter/badminton , baseline/PETS2006 , intermittentObjectMotion/sofa
  #               dynamicBackground/fall (shows CPB-strong / agent-careful) ,
  #               turbulence/turbulence2 & nightVideos/tramStation (failure story)
================================================================================
"""
import argparse, os, pickle
import numpy as np
import cv2

# ---- framework engine (always available) ----------------------------------
try:
    import opus_final_hod_framework as F
except Exception:
    import opus_liang_hod_framework as F

# ---- env engine (optional; only needed for agent/MLP/oracle panels) --------
HH = None
def _load_env_engine():
    """Import opus_headtohead_ablation lazily; it carries proven helpers."""
    global HH
    if HH is not None:
        return HH
    import importlib
    HH = importlib.import_module('opus_headtohead_ablation')
    return HH


def _load_mlp_ckpt(path):
    """Load a PRE-TRAINED 3-way supervised MLP from a checkpoint, so we do not
       retrain it for the figure. Tolerant of the common save layouts:
       state dict under 'net'/'model'/'state_dict'/'sup'/'mlp' (or the ck itself),
       normalisation as mu/sd (mean/std) OR RunningNorm (norm_mean/norm_M2/norm_n).
       The net is Linear(dim,64)-ReLU-Linear(64,64)-ReLU-Linear(64,n_act)."""
    import torch, torch.nn as nn
    ck = torch.load(path, map_location='cpu', weights_only=False)
    # locate the state dict
    sd = None
    if isinstance(ck, dict):
        for k in ('net', 'model', 'state_dict', 'sup', 'mlp', 'sup_state', 'net_state'):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]; break
        if sd is None and all(isinstance(v, torch.Tensor) for v in ck.values()):
            sd = ck                     # the ck itself is a bare state dict
    if sd is None:
        raise ValueError(f'could not find a state dict in {path}; keys={list(ck.keys()) if isinstance(ck,dict) else type(ck)}')
    # infer dims from the first and last Linear weights
    w = [v for kk, v in sd.items() if kk.endswith('weight') and v.ndim == 2]
    dim = w[0].shape[1]; n_act = w[-1].shape[0]
    net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, n_act))
    # strip a possible prefix (e.g. 'net.0.weight' already matches Sequential keys)
    try:
        net.load_state_dict(sd)
    except Exception:
        clean = { '.'.join(kk.split('.')[-2:]) if kk.count('.') > 1 else kk: v
                  for kk, v in sd.items() }
        net.load_state_dict(clean)
    net.eval()
    # locate normalisation
    def _get(*names):
        for nm in names:
            if isinstance(ck, dict) and nm in ck and ck[nm] is not None:
                return np.asarray(ck[nm], np.float64)
        return None
    mu = _get('mu', 'mean', 'norm_mean', 'x_mean')
    sdv = _get('sd', 'std', 'x_std')
    if sdv is None:
        M2 = _get('norm_M2', 'M2'); nrm = _get('norm_n', 'n')
        if M2 is not None and nrm is not None:
            sdv = np.sqrt(M2 / max(float(nrm) - 1.0, 1.0)) + 1e-6
    if mu is None or sdv is None:
        print(f'  [mlp] WARNING: no normalisation stats in {path}; using identity '
              '(panel may look off — prefer a ckpt that stored mu/sd).')
        mu = np.zeros(dim, np.float64); sdv = np.ones(dim, np.float64)
    print(f'  [mlp] loaded pre-trained MLP from {path}  (dim={dim}, actions={n_act})')
    return net, mu.astype(np.float32), sdv.astype(np.float32)


# ============================ drawing helpers ==============================
def gt_to_rgb(gt):
    """white=FG, mid-gray=unknown(85/170), dark=shadow(50), black=BG."""
    out = np.zeros((*gt.shape, 3), np.uint8)
    out[gt == F.GT_FG] = (255, 255, 255)
    out[gt == 50] = (90, 90, 90)
    for v in F.GT_EXCLUDE:
        out[gt == v] = (150, 150, 150)
    return out


def binmask_to_rgb(m):
    b = (m > 127).astype(np.uint8) * 255
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def paint(frame_bgr, mask_bool, color, dim=0.45, alpha=0.65):
    """Darken the frame and paint the True pixels of mask_bool in BGR `color`."""
    base = (frame_bgr.astype(np.float32) * dim).astype(np.uint8)
    over = base.copy()
    over[mask_bool] = color
    return cv2.addWeighted(base, 1 - alpha, over, alpha, 0)


def agree_overlay(frame_bgr, s_cpb, s_g):
    """Green = both say FG (agree-FG); the rest of the agree set is left dark."""
    a_fg = (s_cpb > 127) & (s_g > 127)
    return paint(frame_bgr, a_fg, (0, 200, 0)), float(((s_cpb > 127) == (s_g > 127)).mean())


def dispute_overlay(frame_bgr, s_cpb, s_g, color=(0, 220, 255)):
    disp = (s_cpb > 127) != (s_g > 127)
    return paint(frame_bgr, disp, color), float(disp.mean())


def action_overlay(frame_bgr, r, c, actions):
    """Colour disputed pixels by the agent's chosen action.
       0 trust-CPB = blue, 1 trust-guidance = orange, 2 keep-baseline = gray."""
    base = (frame_bgr.astype(np.float32) * 0.40).astype(np.uint8)
    palette = {0: (255, 120, 0), 1: (0, 170, 255), 2: (170, 170, 170)}  # BGR
    over = base.copy()
    for a, col in palette.items():
        sel = actions == a
        if sel.any():
            over[r[sel], c[sel]] = col
    out = cv2.addWeighted(base, 0.35, over, 0.65, 0)
    frac = {int(a): float((actions == a).mean()) for a in (0, 1, 2)}
    return out, frac


def _fit(m, HW, interp=cv2.INTER_NEAREST):
    H, W = HW
    return m if m.shape[:2] == (H, W) else cv2.resize(m, (W, H), interpolation=interp)


def label_strip(panels, cell_h=220, pad=6, barh=24, fs=0.45):
    """panels = list of (label, bgr_img) -> one horizontal labelled strip (BGR)."""
    tiles = []
    for label, im in panels:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (int(im.shape[1] * cell_h / im.shape[0]), cell_h))
        bar = np.full((barh, im.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, label, (5, barh - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, im]))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 30, np.uint8)])
             for t in tiles]
    sep = np.full((hmax, pad, 3), 255, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.hstack([out, sep, t])
    return out


# ============================ ENV engine ===================================
def env_masks(cat, scene, n, args):
    """Build the scene from caches, pick the correct LOSO fold, and return a dict
       of native masks + the agent action map. Returns None on any failure."""
    from types import SimpleNamespace
    hh = _load_env_engine()
    T, ENV, SEED = hh.T, hh.ENV, hh.SEED
    root = T.find_root()
    sid = f'{cat}/{scene}'

    # namespace build_scenes / _gather_disputes / _fit_supervised expect
    ns = SimpleNamespace(
        cache_dir=args.cache_dir, models_dir=args.models_dir, win_size=args.win_size,
        af=args.af, floor_min=args.floor_min, bth_pixels=args.bth_pixels,
        ref=args.ref, sigema=args.sigema,
        sup_epochs=args.sup_epochs, sup_lr=args.sup_lr, sup_balance=args.sup_balance,
        sup_cap=args.sup_cap)
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels,
                   ref=args.ref, sigema=args.sigema)

    folds = T.make_kfold(args.cv_folds, SEED)
    kf = args.fold
    if kf < 0:
        kf = next((k for k in range(args.cv_folds) if sid in folds[k]), -1)
    if kf < 0:
        print(f'  [env] scene {sid} not found in any fold; pass --fold to force one.')
        return None
    print(f'  [env] scene is the held-out test scene of fold {kf}; using fold{kf} agent.')

    test_sc = T.build_scenes([sid], ns, root, envargs)
    if not test_sc:
        print('  [env] could not build scene from caches (check --cache_dir/--models_dir).')
        return None
    sc = test_sc[0]
    if n in sc.nums:
        i = sc.nums.index(n)
    else:
        i = min(range(len(sc.nums)), key=lambda j: abs(sc.nums[j] - n))
        print(f'  [env] snapping to nearest cached labelled frame {sc.nums[i]}')
    e = sc.env(i)
    r, c = e.dispute_ij()

    out = {'hw': e.gtfg_fit.shape}
    out['guid'] = e._to_native((e.guid_fg > 0).astype(np.uint8) * 255)
    out['cpb'] = e._to_native((e.s1_default > 0).astype(np.uint8) * 255)
    out['framework'] = e.baseline_mask()
    out['r'], out['c'] = r, c

    # ---- agent ----
    agent = hh._load_agent(args.agent_dir, kf)
    if agent is None:
        print(f'  [env] no agent checkpoint at {args.agent_dir}/fold{kf}/best.pth')
        return None
    import torch
    gate, norm, use_patch, patch_k = agent
    if len(r):
        Sn = norm(e.state(r, c))
        Pt = torch.from_numpy(e.patches(r, c, patch_k)) if use_patch else None
        with torch.no_grad():
            aact = gate(torch.from_numpy(Sn), Pt).argmax(-1).numpy()
        out['agent'] = T._segment_gate3(e, aact, r, c)
        out['aact'] = aact
    else:
        out['agent'] = out['framework']; out['aact'] = np.array([], np.int64)

    # ---- oracle ----
    of = e.s1_default.copy()
    if len(r):
        of[r, c] = (e.gtfg_fit[r, c].astype(np.uint8) * 255)
    out['oracle'] = e._to_native(of)

    # ---- MLP panel: fold-correct dir > explicit ckpt > retrain ----
    mlp_path = None
    if args.mlp_dir:
        cand = os.path.join(args.mlp_dir, f'fold{kf}', 'sup.pth')
        if os.path.exists(cand):
            mlp_path = cand
            print(f'  [mlp] using fold-correct MLP {cand}')
        else:
            print(f'  [mlp] {cand} not found; falling back to --mlp_ckpt/--with_mlp')
    if mlp_path is None and args.mlp_ckpt:
        mlp_path = args.mlp_ckpt
    if mlp_path:
        net, mu, sd = _load_mlp_ckpt(mlp_path)
        if len(r):
            Xn = ((e.state(r, c).astype(np.float32) - mu) / sd).astype(np.float32)
            with torch.no_grad():
                sact = net(torch.from_numpy(Xn)).argmax(-1).numpy()
            out['mlp'] = T._segment_gate3(e, sact, r, c)
        else:
            out['mlp'] = out['framework']
    elif args.with_mlp:
        print('  [env] no --mlp_ckpt given; training this fold\'s 3-way MLP for the panel ...')
        rest = [s for j in range(args.cv_folds) if j != kf for s in folds[j]]
        val = T.carve_val(rest, seed=SEED + kf)
        train_ids = [s for s in rest if s not in val]
        train_sc = T.build_scenes(train_ids, ns, root, envargs)
        X, y = hh._gather_disputes(train_sc, args.sup_cap)
        if X is not None:
            net, mu, sd = hh._fit_supervised(X, y, ns, seed=SEED + kf)
            if len(r):
                Xn = ((e.state(r, c).astype(np.float32) - mu) / sd).astype(np.float32)
                with torch.no_grad():
                    sact = net(torch.from_numpy(Xn)).argmax(-1).numpy()
                out['mlp'] = T._segment_gate3(e, sact, r, c)
            else:
                out['mlp'] = out['framework']
        del train_sc
    return out


# ================================ main =====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--guidance', default='deeplab', choices=['zero', 'deeplab', 'pspnet'])
    ap.add_argument('--guidance_models_dir', default='./models')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--channel_mode', default='auto', choices=['auto', 'gray', 'rgb'])
    ap.add_argument('--frame', type=int, default=-1, help='frame number; -1 = middle of ROI')
    ap.add_argument('--list', action='store_true', help='list available ROI frames and exit')
    ap.add_argument('--out', default='fig_panels')
    ap.add_argument('--upscale', type=int, default=2)
    # framework params (Liang Table 1)
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--no_hod_pool', action='store_true')
    # ENV engine (agent / MLP / oracle / action map)
    ap.add_argument('--agent_dir', default=None,
                    help='dir with fold0/best.pth ... enables the agent/oracle panels')
    ap.add_argument('--cache_dir', default='drl_cache_junior')
    ap.add_argument('--with_mlp', action='store_true', help='also render the +MLP panel')
    ap.add_argument('--mlp_ckpt', default=None,
                    help='path to a PRE-TRAINED 3-way MLP checkpoint; loads it instead '
                         'of retraining (implies the +MLP panel)')
    ap.add_argument('--mlp_dir', default=None,
                    help='dir of fold-correct MLPs (e.g. models/sup_gate_folds); loads '
                         '<mlp_dir>/fold{kf}/sup.pth for the scene\'s held-out fold')
    ap.add_argument('--fold', type=int, default=-1, help='force a fold (default: auto by scene)')
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--sup_epochs', type=int, default=15)
    ap.add_argument('--sup_lr', type=float, default=1e-3)
    ap.add_argument('--sup_balance', action='store_true')
    ap.add_argument('--sup_cap', type=int, default=3000)
    args = ap.parse_args()

    root = args.dataset_root or F.find_root()
    if not root:
        raise SystemExit('CDnet root not found — pass --dataset_root')
    mp = os.path.join(args.models_dir, f'cpb_paper_{args.category}_{args.scene}.pkl')
    if not os.path.exists(mp):
        raise SystemExit(f'CPB model not found: {mp}')
    model = pickle.load(open(mp, 'rb'))
    roi_s, roi_e = int(model['params']['roi_start']), int(model['params']['roi_end'])
    sp = os.path.join(root, args.category, args.scene)
    idir, gdir = os.path.join(sp, 'input'), os.path.join(sp, 'groundtruth')
    nums = [n for n in F.all_nums(sp) if roi_s <= n <= roi_e and F.load_gt(gdir, n) is not None]
    if not nums:
        raise SystemExit('no ROI frames with GT found')
    if args.list:
        print(f'{args.category}/{args.scene}: ROI [{roi_s},{roi_e}], {len(nums)} labelled frames')
        print('  first/mid/last:', nums[0], nums[len(nums) // 2], nums[-1])
        print('  sample:', nums[::max(1, len(nums) // 20)][:20])
        return

    n = args.frame if args.frame >= 0 else nums[len(nums) // 2]
    if n not in nums:
        n = min(nums, key=lambda x: abs(x - n))
        print(f'  [snap] using nearest labelled frame {n}')

    # ---------- framework engine ----------
    guidance = F.make_guidance(args.guidance, False, models_dir=args.guidance_models_dir,
                               allow_fallback=False, device=args.device)
    pipe = F.HoDPipeline(model, 'stage3', args.floor_min, args.af, args.sigema,
                         args.bth_pixels, args.ref, not args.no_hod_pool, args.channel_mode)
    bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
    if bgr is None or gt is None:
        raise SystemExit(f'could not load frame {n}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gmask = guidance.predict(bgr)
    final = pipe.run(rgb, gmask)
    s1 = pipe.last_s1

    # ---- resolution diagnostic (tests the downsample hypothesis) ----
    print('\n  [resolution] native shapes (H,W):')
    print(f'    input frame   : {bgr.shape[:2]}')
    print(f'    ground truth  : {gt.shape[:2]}')
    print(f'    guidance s_g  : {np.asarray(gmask).shape[:2]}')
    print(f'    CPB s_cpb     : {np.asarray(s1).shape[:2]}')
    print(f'    framework out : {np.asarray(final).shape[:2]}')
    _shapes = {np.asarray(gmask).shape[:2], np.asarray(s1).shape[:2]}
    if len(_shapes) > 1:
        print('    >> guidance and CPB are at DIFFERENT native resolutions '
              '(one is resampled before fusion) — possible root cause.')
    else:
        print('    >> guidance and CPB share a native resolution.')
    _pp = model.get('params', {})
    for k in ('max_train_pixels', 'downsample', 'scale', 'ds', 'train_h', 'train_w',
              'proc_h', 'proc_w', 'height', 'width', 'block', 'poolnum'):
        if k in _pp:
            print(f'    CPB param {k} = {_pp[k]}')

    H, W = gt.shape
    bgr_f = _fit(bgr, (H, W), cv2.INTER_AREA)
    gmask_f = _fit(gmask, (H, W)); s1_f = _fit(s1, (H, W)); final_f = _fit(final, (H, W))

    os.makedirs(args.out, exist_ok=True)

    def save(name, img):
        if args.upscale > 1:
            img = cv2.resize(img, (img.shape[1] * args.upscale, img.shape[0] * args.upscale),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, name), img)

    gtv = gt_to_rgb(gt)
    gv, cv_ = binmask_to_rgb(gmask_f), binmask_to_rgb(s1_f)
    fv = binmask_to_rgb(final_f)
    agree_img, agree_frac = agree_overlay(bgr_f, s1_f, gmask_f)
    disp_img, dfrac = dispute_overlay(bgr_f, s1_f, gmask_f)

    save('01_input.png', bgr_f)
    save('02_groundtruth.png', gtv)
    save('03_guidance.png', gv)
    save('04_cpb_stage1.png', cv_)
    save('05_framework.png', fv)
    save('06_agree.png', agree_img)
    save('07_dispute.png', disp_img)

    # ---------- env engine (agent / MLP / oracle / action map) ----------
    ev = None
    if args.agent_dir:
        try:
            ev = env_masks(args.category, args.scene, n, args)
        except Exception as ex:
            print(f'  [env] disabled ({type(ex).__name__}: {ex}); framework panels still written.')
            ev = None

    mlp_v = agent_v = oracle_v = act_v = None
    if ev is not None:
        hw = ev['hw']
        fr_env = _fit(bgr_f, hw, cv2.INTER_AREA)
        agent_v = binmask_to_rgb(_fit(ev['agent'], (H, W)))
        save('09_agent.png', agent_v)
        oracle_v = binmask_to_rgb(_fit(ev['oracle'], (H, W)))
        save('11_oracle.png', oracle_v)
        act_img, afrac = action_overlay(fr_env, ev['r'], ev['c'], ev['aact'])
        act_v = _fit(act_img, (H, W))
        save('10_agent_actions.png', act_v)
        print(f'  [env] agent action mix on disputes: trust-CPB {afrac[0]:.2f} | '
              f'trust-guid {afrac[1]:.2f} | keep {afrac[2]:.2f}')
        if 'mlp' in ev:
            mlp_v = binmask_to_rgb(_fit(ev['mlp'], (H, W)))
            save('08_mlp.png', mlp_v)

    # ---------- METHOD strip (Fig. B — the "alive" figure) ----------
    method_panels = [('input', bgr_f), ('guidance $s_g$', gv), ('CPB stage-1 $s_{cpb}$', cv_),
                     (f'agree ({agree_frac*100:.0f}%)', agree_img),
                     (f'dispute ({dfrac*100:.1f}%)', disp_img)]
    if act_v is not None:
        method_panels.append(('agent actions', act_v))
    method_panels.append(('framework (fixed)', fv))
    if agent_v is not None:
        method_panels.append(('+agent (ours)', agent_v))
    cv2.imwrite(os.path.join(args.out, 'method_strip.png'), label_strip(method_panels))

    # ---------- RESULTS row (big side-by-side figure) ----------
    row = [('input', bgr_f), ('GT', gtv), ('guidance', gv), ('CPB', cv_),
           ('framework', fv)]
    if mlp_v is not None:
        row.append(('+MLP', mlp_v))
    if agent_v is not None:
        row.append(('+agent', agent_v))
    if oracle_v is not None:
        row.append(('oracle', oracle_v))
    cv2.imwrite(os.path.join(args.out, 'results_row.png'), label_strip(row))

    # ---------- contact sheet (everything) ----------
    cv2.imwrite(os.path.join(args.out, 'contact_sheet.png'), label_strip(method_panels + row))

    print(f'\n  {args.category}/{args.scene}  frame {n}')
    print(f'  dispute set = {dfrac*100:.2f}% of pixels  agree = {agree_frac*100:.2f}%')
    print(f'  panels + method_strip.png + results_row.png saved -> {os.path.abspath(args.out)}/')
    print('  send me method_strip.png and results_row.png and I will compose the paper figures.')


if __name__ == '__main__':
    main()