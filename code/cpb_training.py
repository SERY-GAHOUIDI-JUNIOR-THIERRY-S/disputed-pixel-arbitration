#!/usr/bin/env python3
"""
================================================================================
cpb_training.py
================================================================================
Corrects the Python CPB trainer so it reproduces the ORIGINAL MATLAB trainncc2.m
(the one validated to reach paper level on LIMU). Three training-side differences
were weakening the dictionary on CDnet's hard categories:

  FIX A — SUPPORT SELECTION.  MATLAB selects the supports with the MINIMUM
          residual-std  std(block - pixel)  (trainncc2.m line 32,37). Your Python
          selected the MAXIMUM Pearson r. The detector thresholds
          |(p - neighbour) - b| >= eta*sigma, so the right supports are the most
          *stable* ones (min residual-std), NOT the most correlated. Pearson is
          the paper's Eq.1 text, but the paper's RESULTS come from the code,
          which uses min-residual-std.  -> default selection = 'minstd'.

  FIX B — GAMMA (support WEIGHT).  MATLAB stores gamma = r/2 + 0.5 in [0,1]
          (trainncc2.m line 48). Your Python stored raw r in [-1,1], and the
          detector drops gamma<=0. On hard, low-correlation scenes (turbulence,
          thermal, night) that under-weights or discards most supports, so the
          vote collapses to 1-2 supports -> low recall. With r/2+0.5 EVERY
          support votes (r=0 -> 0.5, r=-0.3 -> 0.35), a robust vote, exactly as
          MATLAB.  -> default gamma = 'matlab'.

  FIX C — BLOCK REPRESENTATIVE.  MATLAB uses the MEDIAN of the 8x8 block
          (trainncc2.m line 22-23); your Python used the mean. The median is
          robust to a moving object inside the block, which is why the paper
          trains on RAW frames with no cleaning.  -> default block = 'median'.

The b sign stays as your Python detector expects (b = mean(pixel - block)): this
is algebraically identical to MATLAB's b = mean(block - pixel) because the
detector also flips the difference (subncc1.m line 45 vs your k_stage1).

SIGMA FLOOR: the original MATLAB uses NO sigma floor (trainncc2.m stores raw
std, subncc1.m uses std*af directly). So this trainer stores RAW sigma by
default (--train_sigma_floor 0.0), and the acceptance-band floor is a single
GLOBAL TEST-time knob (framework --floor_min), NOT per-category. af stays at the
paper value 2.5; lower the band by lowering --floor_min at detection. --validate
sweeps that floor for you so you can pick one global value.

RECOMMENDED (paper-matching) DEFAULTS:
  selection=minstd, block=median, gamma=matlab, robust_sigma=std, clean_frac=1.0
  (no cleaning, no MAD — the median block already gives the robustness those were
   trying to add; use them only to experiment).

Produces .pkl models byte-compatible with cpb_kernels (same keys), so
the framework / head-to-head / diagnostics load them with no change.

--------------------------------------------------------------------------------
VALIDATE FIRST (one scene, current-vs-fixed, pure-CPB recall/precision/F1 vs GT):
  python cpb_training.py --validate \
      --category turbulence --scene turbulence3 --channels gray \
      --output_dir models/cpb_fix_check --n_train 400 --n_test 30

TRAIN ONE SCENE with the fix:
  python cpb_training.py \
      --category thermal --scene park --channels gray \
      --output_dir models/cpb_matlab --force

RETRAIN ALL 53 (copy guidance stays; models_dir -> this out):
  python cpb_training.py \
      --channels rgb --output_dir models/cpb_matlab --force
  # (gray categories: thermal/turbulence are single-channel; use --channels gray
  #  for those, rgb for the colour ones — or just run rgb everywhere, the env
  #  auto-detects gray models.)
================================================================================
"""
import argparse, os, sys, time, pickle
import numpy as np
import cv2

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    prange = range

    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]

        def deco(f):
            return f
        return deco

import cpb_training_base as T0
import cpb_kernels as F

BLOCK_SIZE = T0.BLOCK_SIZE
K_SUPPORT = T0.K_SUPPORT
POOLNUM = T0.POOLNUM
ETA = T0.ETA
LAMBDA_DEC = T0.LAMBDA_DEC
SIGMA_FLOOR_MIN = T0.SIGMA_FLOOR_MIN
N_TRAIN = T0.N_TRAIN
MAX_TRAINING_PIXELS = T0.MAX_TRAINING_PIXELS
MIN_FRAMES = T0.MIN_FRAMES
ALL = T0.ALL_CDNET_SCENES
GT_FG = F.GT_FG
NEG = -1e30


# ============================================================================
# block MEDIAN stack (FIX C)  — per-frame spatial median of each 8x8 block
# ============================================================================
def block_median_stack(gray_stack, ws, hs, bs=BLOCK_SIZE):
    T, H, W = gray_stack.shape
    pad = np.zeros((T, ws * bs, hs * bs), np.float32); pad[:, :H, :W] = gray_stack
    r = np.median(pad.reshape(T, ws, bs, hs, bs), axis=(2, 4))      # [T, ws, hs]
    return np.ascontiguousarray(r.transpose(1, 2, 0), np.float32)   # -> [ws, hs, T]


# ============================================================================
# corrected training kernel (parametric so --validate can flip one knob)
#   sel_minstd=1  -> select MIN residual-std   (FIX A);  0 -> MAX Pearson (old)
#   gamma_matlab=1-> store r/2+0.5             (FIX B);  0 -> raw r       (old)
#   DBL passed in is the block MEDIAN (FIX C) or block MEAN (old)
# ============================================================================
@njit(cache=False, parallel=True, fastmath=False)
def train_kernel_fixed(gray_stack, DBL, poolnum, sfloor, min_gamma, out, sel_minstd, gamma_matlab):
    T, H, W = gray_stack.shape
    ws, hs = DBL.shape[0], DBL.shape[1]
    for x in prange(H):
        sc_ = np.full(poolnum, NEG)
        bi_ = np.full(poolnum, -1); bj_ = np.full(poolnum, -1)
        bb_ = np.zeros(poolnum); bs_ = np.zeros(poolnum); gm_ = np.zeros(poolnum)
        for y in range(W):
            for q in range(poolnum):
                sc_[q] = NEG; bi_[q] = -1; bj_[q] = -1; bb_[q] = 0.0; bs_[q] = 0.0; gm_[q] = 0.0
            mp = 0.0
            for t in range(T):
                mp += gray_stack[t, x, y]
            mp /= T
            sp2 = 0.0
            for t in range(T):
                d = gray_stack[t, x, y] - mp; sp2 += d * d
            sp_sum = sp2 ** 0.5
            self_bi = x // BLOCK_SIZE; self_bj = y // BLOCK_SIZE
            if sp_sum < 1e-6:
                k = 0
                for bi in range(ws):
                    for bj in range(hs):
                        if k >= poolnum:
                            break
                        if bi == self_bi and bj == self_bj:
                            continue
                        out[x, y, k, 0] = bi; out[x, y, k, 1] = bj
                        out[x, y, k, 2] = 0.0; out[x, y, k, 3] = sfloor; out[x, y, k, 4] = 0.0
                        k += 1
                continue
            filled = 0; worst = 1e30; worst_idx = 0
            for bi in range(ws):
                for bj in range(hs):
                    if bi == self_bi and bj == self_bj:
                        continue
                    mb = 0.0
                    for t in range(T):
                        mb += DBL[bi, bj, t]
                    mb /= T
                    cov = 0.0; vb = 0.0
                    for t in range(T):
                        pp = gray_stack[t, x, y] - mp; qq = DBL[bi, bj, t] - mb
                        cov += pp * qq; vb += qq * qq
                    sb_sum = vb ** 0.5
                    r = 0.0 if sb_sum < 1e-9 else cov / (sp_sum * sb_sum)
                    if r > 1.0:
                        r = 1.0
                    elif r < -1.0:
                        r = -1.0
                    b = mp - mb
                    var = (sp2 - 2.0 * cov + vb) / T
                    if var < 0.0:
                        var = 0.0
                    sig = var ** 0.5
                    gamma = (r * 0.5 + 0.5) if gamma_matlab == 1 else r
                    sc = (-sig) if sel_minstd == 1 else r        # score to MAXIMISE
                    if filled < poolnum:
                        sc_[filled] = sc; bi_[filled] = bi; bj_[filled] = bj
                        bb_[filled] = b; bs_[filled] = sig; gm_[filled] = gamma
                        filled += 1
                        if filled == poolnum:
                            worst = sc_[0]; worst_idx = 0
                            for q in range(1, poolnum):
                                if sc_[q] < worst:
                                    worst = sc_[q]; worst_idx = q
                    elif sc > worst:
                        sc_[worst_idx] = sc; bi_[worst_idx] = bi; bj_[worst_idx] = bj
                        bb_[worst_idx] = b; bs_[worst_idx] = sig; gm_[worst_idx] = gamma
                        worst = sc_[0]; worst_idx = 0
                        for q in range(1, poolnum):
                            if sc_[q] < worst:
                                worst = sc_[q]; worst_idx = q
            # sort DESC by sc_ (best support first) carrying all fields
            for a in range(1, poolnum):
                sv = sc_[a]; iv = bi_[a]; jv = bj_[a]; bv = bb_[a]; gv = bs_[a]; wv = gm_[a]; c = a - 1
                while c >= 0 and sc_[c] < sv:
                    sc_[c + 1] = sc_[c]; bi_[c + 1] = bi_[c]; bj_[c + 1] = bj_[c]
                    bb_[c + 1] = bb_[c]; bs_[c + 1] = bs_[c]; gm_[c + 1] = gm_[c]; c -= 1
                sc_[c + 1] = sv; bi_[c + 1] = iv; bj_[c + 1] = jv
                bb_[c + 1] = bv; bs_[c + 1] = gv; gm_[c + 1] = wv
            for k in range(poolnum):
                bi = bi_[k]; bj = bj_[k]
                if bi < 0 or gm_[k] < min_gamma:
                    out[x, y, k, 0] = bi if bi >= 0 else 0
                    out[x, y, k, 1] = bj if bj >= 0 else 0
                    out[x, y, k, 2] = 0.0; out[x, y, k, 3] = sfloor; out[x, y, k, 4] = 0.0
                    continue
                sig = bs_[k]
                if sig < sfloor:
                    sig = sfloor
                out[x, y, k, 0] = bi; out[x, y, k, 1] = bj
                out[x, y, k, 2] = bb_[k]; out[x, y, k, 3] = sig; out[x, y, k, 4] = gm_[k]


def train_channel_fixed(gray_stack, poolnum, min_gamma, sfloor, cname,
                        block, selection, gamma_conv, robust_sigma):
    T, H, W = gray_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    DBL = (block_median_stack(gray_stack, ws, hs) if block == 'median'
           else T0.block_mean_stack(gray_stack, ws, hs))
    out = np.zeros((H, W, poolnum, 5), np.float32)
    train_kernel_fixed(np.ascontiguousarray(gray_stack), DBL, poolnum, sfloor, min_gamma, out,
                       1 if selection == 'minstd' else 0, 1 if gamma_conv == 'matlab' else 0)
    s_before = float(np.median(out[:, :, :K_SUPPORT, 3]))
    if robust_sigma == 'mad':
        out = T0.robust_sigma_pass(gray_stack, out, sfloor)
    g = out[:, :, :K_SUPPORT, 4]
    usable = float((g > 0.0).mean() * 100)
    print(f"      [{cname}] sel={selection} block={block} gamma={gamma_conv} "
          f"| gamma(topK) mean={np.mean(g):+.3f} usable(>0)={usable:.0f}%  "
          f"sigma(topK) med={s_before:.2f}", flush=True)
    return out


# ============================================================================
# ADAPTIVE SELECTOR  (--res_auto)
#   Picks, per scene, the spatial RESOLUTION and a per-scene band FLOOR from
#   background-dynamics statistics measured on the pre-ROI TRAINING frames only
#   (no test labels). Two rules, from the turbulence/cubicle discovery:
#
#   RULE 1  resolution <-> RECALL (the all-64 detection rule).
#     Textured / dynamic backgrounds make each 8x8 block too varied for the
#     "all 64 pixels break the band" rule to fire -> recall collapses (turbulence
#     0.19 @608x408). Downsample until the median intra-block spatial std drops
#     to <= tex_target. Bounded in [px_min, px_max]; px_max=77k = the validated
#     Zhou scale, so the worst case is simply the known-good global 77k.
#
#   RULE 2  band floor <-> PRECISION.
#     The min-residual-std supports carry an over-optimistic sigma; on near-static
#     scenes that thin band floods (cubicle 30% fg). Lift it so the effective
#     sigma reaches target_sigma:  floor = clip(target_sigma - sigma_med, 0,
#     floor_max). It is BAKED into the stored sigma, so you run the framework /
#     eval with --floor_min 0 and every scene carries its own floor.
# ============================================================================
def _luma_frame(scene_path, num, target_size):
    im = T0._read_frame(scene_path, num)
    if im is None:
        return None
    if target_size is not None:
        im = cv2.resize(im, target_size)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def intra_block_std(gray, bs=BLOCK_SIZE):
    """Median over 8x8 blocks of the per-block spatial std — the quantity the
    all-64 rule chokes on. High = textured/dynamic; low = smooth/static."""
    H, W = gray.shape
    ws = int(np.ceil(H / bs)); hs = int(np.ceil(W / bs))
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = gray
    s = pad.reshape(ws, bs, hs, bs).std(axis=(1, 3))
    return float(np.median(s))


def choose_resolution(scene_path, pre_nums, native, args, max_frames=40):
    """RULE 1: largest pixel budget (<= px_max) whose resampled median intra-block
    std <= tex_target. Returns (target_size, Hf, Wf, S_chosen, budget)."""
    H0, W0 = native; native_px = H0 * W0
    samp = pre_nums
    if len(samp) > max_frames:
        idx = np.linspace(0, len(samp) - 1, max_frames).round().astype(int)
        samp = [samp[i] for i in sorted(set(idx.tolist()))]
    nat = [g for g in (_luma_frame(scene_path, n, None) for n in samp) if g is not None]
    top = int(max(args.px_min, min(native_px, args.px_max)))
    if not nat:
        ts, Hf, Wf = T0.plan_size(H0, W0, top); return ts, Hf, Wf, 0.0, top
    ladder = sorted({b for b in (top, 77000, 60000, 48000, args.px_min)
                     if args.px_min <= b <= top}, reverse=True) or [top]
    pick = None
    for b in ladder:
        ts, Hf, Wf = T0.plan_size(H0, W0, b)
        vals = [intra_block_std(g if (g.shape[0] == Hf and g.shape[1] == Wf) else cv2.resize(g, (Wf, Hf)))
                for g in nat]
        S = float(np.mean(vals))
        if S <= args.tex_target:
            pick = (ts, Hf, Wf, S, b); break
    if pick is None:                      # even the smallest budget is still too textured
        b = ladder[-1]; ts, Hf, Wf = T0.plan_size(H0, W0, b)
        vals = [intra_block_std(g if (g.shape[0] == Hf and g.shape[1] == Wf) else cv2.resize(g, (Wf, Hf)))
                for g in nat]
        pick = (ts, Hf, Wf, float(np.mean(vals)), b)
    return pick


def save_model_fixed(D, category, scene, out_dir, img_size, original_size, target_size,
                     rs, re_, floors, channels, clean_stats, args, extra_params=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    obj = {'DBLK_r': D['R'], 'DBLK_g': D['G'], 'DBLK_b': D['B'],
           'category': category, 'scene': scene, 'img_size': img_size,
           'original_size': original_size, 'target_size': target_size,
           'params': {'block_size': BLOCK_SIZE, 'K': K_SUPPORT, 'poolnum': args.poolnum,
                      'eta': ETA, 'lambda_thresh': LAMBDA_DEC, 'sigma_floor': SIGMA_FLOOR_MIN,
                      'sigma_floor_calibrated': floors, 'min_gamma': args.min_gamma,
                      'stat': args.robust_sigma, 'selection': args.selection, 'block': args.block,
                      'gamma': args.gamma, 'channels': channels,
                      'clean_stats': clean_stats, 'roi_start': rs, 'roi_end': re_},
           'version': 'cpb_matlab_corrected_v1', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    if extra_params:
        obj['params'].update(extra_params)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def calibrate_band_floor(sp, cal_nums, target_size, channels, chan, pool,
                         af, bth, ref, cands, fg_target):
    """TRAIN-TIME, LEAKAGE-FREE per-scene band calibration.

    The training frames are (near-)pure background, so ANY pixel CPB flags on
    them is a false alarm. We raise the sigma-floor to the SMALLEST value at
    which the detected foreground fraction on these background frames drops to
    <= fg_target. That floor is baked into the stored sigma. No groundtruth and
    no evaluation frames are ever touched -> zero test leakage.

    Returns (chosen_floor, [(floor, bg_fg_fraction), ...]).
    """
    stacks, img_size = T0.load_stacks(sp, cal_nums, target_size, channels)
    if stacks is None:
        return float(cands[-1]), []
    Hf, Wf = img_size
    ws = int(np.ceil(Hf / BLOCK_SIZE)); hs = int(np.ceil(Wf / BLOCK_SIZE))
    names = ['L'] if channels == 'gray' else ['R', 'G', 'B']
    T = stacks[names[0]].shape[0]
    guid_fg = np.zeros((Hf, Wf), np.uint8)                 # guidance OFF -> pure CPB
    gcnt = F.block_fg_count(guid_fg, ws, hs)
    # precompute DBS per frame/channel once
    DBS = {nm: [F.block_mean(stacks[nm][t], ws, hs) for t in range(T)] for nm in names}
    curve = []
    chosen = float(cands[-1])
    for fl in sorted(float(c) for c in cands):             # ascending; fg-frac is monotone down
        fsum = 0.0
        for t in range(T):
            acc = None
            for nm in names:
                Dc = chan[nm]
                s1 = np.zeros((Hf, Wf), np.uint8)
                sc = np.zeros((Hf, Wf), np.float32); su = np.zeros((Hf, Wf), np.float32)
                F.k_stage1(stacks[nm][t], DBS[nm][t], gcnt, guid_fg, Dc, s1, sc, su,
                           float(af), float(fl), F.LAMBDA_DEC, float(bth), int(ref), pool, True)
                acc = s1 if acc is None else np.maximum(acc, s1)
            acc = cv2.medianBlur(acc, 3)
            fsum += float((acc > 0).mean())
        frac = fsum / max(T, 1)
        curve.append((fl, frac))
        if frac <= fg_target:
            chosen = fl; break
    return chosen, curve


def train_scene_fixed(root, category, scene, out_dir, args, return_model=False):
    sp = os.path.join(root, category, scene)
    if not os.path.isdir(sp):
        print(f"    [SKIP] not found: {sp}"); return None if return_model else 'missing'
    out_path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    if (not return_model) and os.path.exists(out_path) and not args.force:
        print(f"    [resume] already trained: {category}/{scene}"); return 'exists'
    pre, rs, re_ = T0.pre_roi_nums(sp, args.n_train)
    if len(pre) < MIN_FRAMES:
        print(f"    [SKIP] too few frames: {scene}"); return None if return_model else 'too_few'
    # ---- auto_floor: reserve the LATEST pre-ROI frames (closest to roi_start, but
    #      still strictly BEFORE it) as a held-out calibration set, and train the
    #      dictionary on the earlier pre-ROI frames so there is no overlap. No frame
    #      inside the temporalROI is ever touched -> paper-faithful, leakage-free.
    calib_nums = []
    if args.auto_floor:
        all_pre = [n for n in T0.list_nums(os.path.join(sp, 'input')) if n < rs]
        K = max(10, int(round(args.auto_floor_nframes * 1.6)))     # reserve a bit extra; cleaning trims it
        if len(all_pre) >= args.n_train + K + MIN_FRAMES:
            calib_nums = all_pre[-K:]
            pre = all_pre[:-K][-args.n_train:]
        elif len(all_pre) >= 2 * MIN_FRAMES + 10:
            K = max(10, min(K, len(all_pre) // 4))
            calib_nums = all_pre[-K:]
            pre = (all_pre[:-K])[-args.n_train:] or all_pre[:-K]
        else:
            calib_nums = []                                        # too few -> fall back inside the block
    native = T0.probe_native(sp, pre[0])
    H0, W0 = native

    # ---- RULE 1: resolution (adaptive or fixed cap) ----
    S_tex = None; budget = None
    if args.res_auto:
        target_size, Hf, Wf, S_tex, budget = choose_resolution(sp, pre, native, args)
        tsf = 0.0                          # train on RAW sigma; bake per-scene floor after
    else:
        target_size, Hf, Wf = T0.plan_size(H0, W0, args.max_train_pixels)
        tsf = args.train_sigma_floor       # 0.0 = raw sigma (MATLAB); band floor is a TEST-time knob

    pre_clean, cstats = T0.clean_frames(sp, pre, target_size, args.clean_frac, args.fg_thr)
    fit = pre_clean
    extra = (f"  [res_auto] intra_block_std={S_tex:.2f} budget_px={budget}" if args.res_auto else "")
    print(f"  > {category}/{scene}  native {W0}x{H0} -> fit {Wf}x{Hf}  pool={args.poolnum} "
          f"ch={args.channels} sel={args.selection} block={args.block} gamma={args.gamma} "
          f"train_sigma_floor={tsf} clean={args.clean_frac} frames={len(fit)}{extra}")
    stacks, img_size = T0.load_stacks(sp, fit, target_size, args.channels)
    if stacks is None:
        print(f"    [SKIP] load failed: {scene}"); return None if return_model else 'load_failed'
    H, W = img_size; t0 = time.time(); chan = {}
    names = ['L'] if args.channels == 'gray' else ['R', 'G', 'B']
    for nm in names:
        chan[nm] = train_channel_fixed(stacks[nm], args.poolnum, args.min_gamma, tsf,
                                       nm, args.block, args.selection, args.gamma, args.robust_sigma)
    del stacks

    # ---- RULE 2: per-scene band floor, baked into stored sigma ----
    if args.auto_floor:
        # TRAIN-TIME unsupervised calibration: widen the band to the smallest floor
        # that keeps the (background) training frames quiet. One floor for all channels
        # (OR-across-channel detection), baked into every channel's stored sigma.
        cands = [float(x) for x in args.auto_floor_cands.split(',')]
        # Held-out = latest pre-ROI frames (reserved above, disjoint from training).
        # Motion-clean them to drop any with activity, then calibrate.
        if calib_nums:
            cc, _ = T0.clean_frames(sp, calib_nums, target_size, 0.7, args.fg_thr)
            src, src_tag = (cc or calib_nums), 'held-out latest-preROI'
        else:
            src, src_tag = fit, 'train(fallback)'
        if len(src) > args.auto_floor_nframes:
            idx = np.linspace(0, len(src) - 1, args.auto_floor_nframes).round().astype(int)
            cn = [src[i] for i in sorted(set(idx.tolist()))]
        else:
            cn = src
        fl, curve = calibrate_band_floor(sp, cn, target_size, args.channels, chan, args.poolnum,
                                         ETA, 7.0, K_SUPPORT, cands, args.auto_floor_target)
        print(f"      [auto_floor] calib frames: {len(cn)} {src_tag} (strictly pre-ROI)")
        # sigma-gated SAFETY min floor: thin-band (very static) scenes are the ones that
        # collapse on in-window drift; give only those a minimum floor. Fat-sigma (dynamic)
        # scenes already have a cushion and keep their calibrated floor untouched. sigma_med
        # is a pre-ROI quantity, so this stays leakage-free.
        sig_med = float(np.median(chan[names[0]][:, :, :K_SUPPORT, 3]))
        fl_cal = fl
        # online-update policy, baked into the model: thin-sigma (static) scenes adapt
        # online (b+sigma); fat-sigma (dynamic/jitter) scenes stay frozen at inference.
        auto_sig_med = sig_med
        auto_online = bool(sig_med < args.min_floor_sigma_gate)
        if args.min_floor > 0.0 and sig_med < args.min_floor_sigma_gate:
            fl = max(fl, args.min_floor)
            print(f"      [auto_floor] sigma_med={sig_med:.2f} < gate {args.min_floor_sigma_gate} "
                  f"-> safety min floor: {fl_cal:.1f} -> {fl:.1f}")
        else:
            print(f"      [auto_floor] sigma_med={sig_med:.2f} >= gate {args.min_floor_sigma_gate} "
                  f"-> keep calibrated floor {fl:.1f}")
        for nm in names:
            chan[nm][:, :, :, 3] = np.maximum(chan[nm][:, :, :, 3], fl)
        floors = {'R': fl, 'G': fl, 'B': fl} if args.channels == 'gray' else {nm: fl for nm in names}
        cstr = ' '.join(f'{int(c)}:{f:.3f}' for c, f in curve)
        print(f"      [auto_floor] bg-fg-fraction by floor -> {cstr}")
        print(f"      [auto_floor] chosen floor={fl:.2f}  (target bg-fg<={args.auto_floor_target}) "
              f"eff sigma_med={float(np.median(chan[names[0]][:, :, :K_SUPPORT, 3])):.2f}")
    elif args.res_auto:
        bf = {}
        for nm in names:
            sig_med = float(np.median(chan[nm][:, :, :K_SUPPORT, 3]))
            fl = float(np.clip(args.target_sigma - sig_med, 0.0, args.floor_max))
            chan[nm][:, :, :, 3] = np.maximum(chan[nm][:, :, :, 3], fl)
            bf[nm] = fl
            print(f"      [{nm}] sigma_med(raw)={sig_med:.2f} -> baked floor={fl:.2f} "
                  f"(eff sigma_med={float(np.median(chan[nm][:, :, :K_SUPPORT, 3])):.2f})")
        floors = {'R': bf['L'], 'G': bf['L'], 'B': bf['L']} if args.channels == 'gray' else bf
    else:
        floors = {'R': tsf, 'G': tsf, 'B': tsf}

    D = {'R': chan['L'], 'G': chan['L'], 'B': chan['L']} if args.channels == 'gray' else chan
    print(f"    total {time.time()-t0:.0f}s")
    if return_model:
        return {'DBLK_r': D['R'], 'DBLK_g': D['G'], 'DBLK_b': D['B'], 'img_size': img_size,
                'params': {'roi_start': rs, 'roi_end': re_, 'poolnum': args.poolnum}}
    extra_params = (dict(res_auto=True, intra_block_std=S_tex, budget_px=budget,
                         target_sigma=args.target_sigma, baked_floor=floors)
                    if args.res_auto else None)
    if args.auto_floor:
        extra_params = extra_params or {}
        extra_params.update(sigma_med_raw=auto_sig_med, online_update=auto_online,
                            sigma_gate=args.min_floor_sigma_gate)
    save_model_fixed(D, category, scene, out_dir, img_size, (H0, W0), target_size, rs, re_,
                     floors, args.channels, cstats, args, extra_params)
    return 'trained'


# ============================================================================
# EVAL — pure CPB stage-1 (guidance = zero) vs GT, to isolate dictionary quality
# ============================================================================
def _detect_channels(model):
    Dr = np.ascontiguousarray(model['DBLK_r'], np.float32)
    Dg = np.ascontiguousarray(model.get('DBLK_g', Dr), np.float32)
    Db = np.ascontiguousarray(model.get('DBLK_b', Dr), np.float32)
    if np.allclose(Dr, Dg) and np.allclose(Dr, Db):
        return [('L', Dr)]
    return [('R', Dr), ('G', Dg), ('B', Db)]


def eval_cpb(model, root, cat, scene, af, floor, n_test):
    Hf, Wf = model['img_size']
    ws = int(np.ceil(Hf / BLOCK_SIZE)); hs = int(np.ceil(Wf / BLOCK_SIZE))
    chans = _detect_channels(model); pool = chans[0][1].shape[2]
    sp = os.path.join(root, cat, scene)
    idir = os.path.join(sp, 'input'); gdir = os.path.join(sp, 'groundtruth')
    rs = int(model['params'].get('roi_start', 1)); re_ = int(model['params'].get('roi_end', 10 ** 9))
    nums = [n for n in F.all_nums(sp) if rs <= n <= re_
            and os.path.exists(os.path.join(gdir, f'gt{n:06d}.png'))]
    if n_test and len(nums) > n_test:
        idx = np.linspace(0, len(nums) - 1, n_test).round().astype(int)
        nums = [nums[i] for i in sorted(set(idx.tolist()))]
    agg = [0, 0, 0, 0]
    for n in nums:
        bgr = F.load_bgr(idir, n); gt = F.load_gt(gdir, n)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        fr = cv2.resize(rgb, (Wf, Hf)) if rgb.shape[:2] != (Hf, Wf) else rgb
        guid_fg = np.zeros((Hf, Wf), np.uint8)
        gcnt = F.block_fg_count(guid_fg, ws, hs)
        s1_acc = None
        for nm, Dc in chans:
            chan = F.to_luma(fr) if nm == 'L' else fr[:, :, {'R': 0, 'G': 1, 'B': 2}[nm]].astype(np.float32)
            DBS = F.block_mean(chan, ws, hs)
            s1 = np.zeros((Hf, Wf), np.uint8); sc = np.zeros((Hf, Wf), np.float32); su = np.zeros((Hf, Wf), np.float32)
            F.k_stage1(chan, DBS, gcnt, guid_fg, Dc, s1, sc, su,
                       float(af), float(floor), F.LAMBDA_DEC, 7.0, 20, pool, True)
            s1_acc = s1 if s1_acc is None else np.maximum(s1_acc, s1)
        s1_acc = cv2.medianBlur(s1_acc, 3)   # Zhou ForegroundDetection.m: IIB2 = medfilt2(IIB,[3,3])
        pred = cv2.resize(s1_acc, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        for j, v in enumerate(F.frame_counts(pred, gt)):
            agg[j] += v
    e = 1e-10; tp, fp, fn, tn = agg
    rec = tp / (tp + fn + e); prec = tp / (tp + fp + e)
    f1 = 2 * rec * prec / (rec + prec + e)
    g = chans[0][1][:, :, :K_SUPPORT, 4]
    return dict(recall=rec, precision=prec, f1=f1,
                usable_pct=float((g > 0.0).mean() * 100),
                gamma_mean=float(g.mean()), sigma_med=float(np.median(chans[0][1][:, :, :K_SUPPORT, 3])),
                cpb_fg_pct=float(100 * (pred > 0).mean()), n=len(nums))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--output_dir', default='models/cpb_matlab')
    ap.add_argument('--category', default=None); ap.add_argument('--scene', default=None)
    ap.add_argument('--channels', choices=['gray', 'rgb'], default='gray')
    ap.add_argument('--n_train', type=int, default=N_TRAIN)
    ap.add_argument('--poolnum', type=int, default=POOLNUM)
    ap.add_argument('--min_gamma', type=float, default=0.0)
    ap.add_argument('--max_train_pixels', type=int, default=MAX_TRAINING_PIXELS)
    # ---- the corrected recipe (paper-matching defaults) ----
    ap.add_argument('--selection', choices=['minstd', 'pearson'], default='minstd', help='FIX A (default minstd = MATLAB)')
    ap.add_argument('--block', choices=['median', 'mean'], default='median', help='FIX C (default median = MATLAB)')
    ap.add_argument('--gamma', choices=['matlab', 'raw'], default='matlab', help='FIX B (default matlab = r/2+0.5)')
    ap.add_argument('--robust_sigma', choices=['std', 'mad'], default='std', help='std = MATLAB (recommended)')
    ap.add_argument('--clean_frac', type=float, default=1.0, help='1.0 = no cleaning (MATLAB); try 0.7 to experiment')
    ap.add_argument('--fg_thr', type=float, default=30.0)
    ap.add_argument('--train_sigma_floor', type=float, default=0.0,
                    help='floor on the STORED sigma. 0.0 = raw sigma (MATLAB has NO floor). The band floor '
                         'is a TEST-time knob (framework --floor_min), so leave this 0 and lower the band at detection.')
    # ---- ADAPTIVE selector (scene-adaptive resolution + per-scene baked floor) ----
    ap.add_argument('--res_auto', action='store_true',
                    help='ADAPTIVE: per scene, pick resolution (RULE 1) + bake a per-scene band floor (RULE 2) '
                         'from training-frame background dynamics. Run the framework / eval with --floor_min 0.')
    ap.add_argument('--tex_target', type=float, default=12.0,
                    help='res_auto RULE 1: downsample below px_max only if median intra-block std exceeds this '
                         '(default 12 keeps the validated 77k for most scenes; LOWER it for more aggressive per-scene downsampling).')
    ap.add_argument('--px_min', type=int, default=40000, help='res_auto: smallest pixel budget.')
    ap.add_argument('--px_max', type=int, default=77000, help='res_auto: largest pixel budget (77k = validated Zhou scale).')
    ap.add_argument('--target_sigma', type=float, default=4.0,
                    help='res_auto RULE 2: lift the band so effective sigma_med reaches this (floor=clip(target-sigma_med,0,floor_max)).')
    ap.add_argument('--floor_max', type=float, default=6.0, help='res_auto: cap on the per-scene baked floor.')
    # ---- ADAPTIVE band (train-time, unsupervised, leakage-free) ----
    ap.add_argument('--auto_floor', action='store_true',
                    help='Replace RULE 2 with a per-scene band calibrated on BACKGROUND training frames: '
                         'raise the sigma-floor to the smallest value keeping bg foreground-fraction <= '
                         '--auto_floor_target. No labels, no test frames. Run eval with --floor_min 0.')
    ap.add_argument('--auto_floor_target', type=float, default=0.005,
                    help='auto_floor: target background foreground-fraction (the "quiet" threshold).')
    ap.add_argument('--auto_floor_cands', default='0,1,2,3,4,6,8,10,12,15,18,22,26,30',
                    help='auto_floor: candidate floors swept ascending; first one meeting the target is chosen.')
    ap.add_argument('--auto_floor_nframes', type=int, default=20,
                    help='auto_floor: number of background training frames used for calibration.')
    ap.add_argument('--min_floor', type=float, default=6.0,
                    help='auto_floor: sigma-gated SAFETY min floor applied ONLY to thin-band scenes '
                         '(sigma_med < --min_floor_sigma_gate), to stop in-window drift collapse. 0 = off.')
    ap.add_argument('--min_floor_sigma_gate', type=float, default=2.0,
                    help='auto_floor: only scenes whose calibrated sigma_med is below this get the min floor.')
    # ---- eval / validate ----
    ap.add_argument('--validate', action='store_true',
                    help='train CURRENT (pearson/mean/raw) AND FIXED (minstd/median/matlab) on this scene, '
                         'then sweep the TEST-time band floor (af fixed) and print pure-CPB R/P/F1. Needs --category/--scene.')
    ap.add_argument('--n_test', type=int, default=30)
    ap.add_argument('--af', type=float, default=ETA, help='eta — keep at the paper value 2.5')
    ap.add_argument('--floor_min', type=float, default=6.0, help='single-eval band floor (used when not sweeping)')
    ap.add_argument('--floor_sweep', default='6,4,3,2,1,0',
                    help='TEST-time band floors to sweep in --validate (af stays fixed). 0 = raw sigma (MATLAB).')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    root = args.dataset_root or T0.find_root()
    if not root:
        sys.exit('ERROR: CDNet2014 not found — use --dataset_root')

    if args.validate:
        if not (args.category and args.scene):
            sys.exit('--validate needs --category and --scene')
        import copy
        print('\n' + '=' * 84)
        print(f'  VALIDATE  {args.category}/{args.scene}   CURRENT (old) vs FIXED (MATLAB recipe)')
        print('=' * 84)
        configs = {'CURRENT(pearson/mean/raw)': dict(selection='pearson', block='mean', gamma='raw'),
                   'FIXED(minstd/median/matlab)': dict(selection='minstd', block='median', gamma='matlab')}
        models = {}
        for label, cfg in configs.items():
            a = copy.deepcopy(args)
            a.res_auto = False      # --validate stays a controlled fixed-resolution / floor-sweep comparison
            a.selection = cfg['selection']; a.block = cfg['block']; a.gamma = cfg['gamma']
            print(f'\n  --- training {label} ---')
            models[label] = train_scene_fixed(root, args.category, args.scene, args.output_dir, a, return_model=True)
        floors = [float(x) for x in args.floor_sweep.split(',')]
        print('\n' + '-' * 84)
        print(f"  TEST-time band-floor sweep  (af fixed = {args.af}; floor=0 -> raw sigma = MATLAB)")
        print('-' * 84)
        print(f"  {'floor':>6} | {'CUR F1':>7} | {'FIX F1':>7}{'FIX R':>8}{'FIX P':>8}{'FIXfg%':>8}   {'dF1(fix-cur)':>12}")
        best = (-1, None)
        for fl in floors:
            rc = eval_cpb(models['CURRENT(pearson/mean/raw)'], root, args.category, args.scene, args.af, fl, args.n_test) \
                if models['CURRENT(pearson/mean/raw)'] else None
            rf = eval_cpb(models['FIXED(minstd/median/matlab)'], root, args.category, args.scene, args.af, fl, args.n_test) \
                if models['FIXED(minstd/median/matlab)'] else None
            if rf is None:
                continue
            cf = rc['f1'] if rc else float('nan')
            print(f"  {fl:>6.1f} | {cf:>7.3f} | {rf['f1']:>7.3f}{rf['recall']:>8.3f}{rf['precision']:>8.3f}"
                  f"{rf['cpb_fg_pct']:>8.2f}   {rf['f1']-cf:>+12.3f}")
            if rf['f1'] > best[0]:
                best = (rf['f1'], fl)
        print('-' * 84)
        print(f"  best FIXED pure-CPB F1 = {best[0]:.3f} at band floor = {best[1]}  (af={args.af})")
        print("  Pick ONE global floor from this (lower = more recall, watch precision). Then run the framework / "
              "head-to-head with --floor_min <that value>. af stays 2.5.")
        print("  NOTE: also run this on an EASY scene (e.g. baseline/highway) to be sure a low global floor "
              "doesn't cost precision there — the floor is global, not per-category.")
        print('=' * 84)
        return

    # ---- normal training ----
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}\n  CPB TRAINING (MATLAB-corrected: sel={args.selection} block={args.block} "
          f"gamma={args.gamma} clean={args.clean_frac} sigma={args.robust_sigma})")
    print(f"  engine={'numba' if HAVE_NUMBA else 'PYTHON-FALLBACK(slow)'}  root={root}  out={args.output_dir}\n{'='*70}")
    cats = [args.category] if args.category else list(ALL.keys())
    tally = {}
    for cat in cats:
        if cat not in ALL:
            print(f"  [WARN] unknown category {cat}"); continue
        for sc in ALL[cat]:
            if args.scene and sc != args.scene:
                continue
            try:
                st = train_scene_fixed(root, cat, sc, args.output_dir, args)
            except Exception as e:
                print(f"    [ERROR] {cat}/{sc}: {e}"); st = 'error'
            tally[st] = tally.get(st, 0) + 1
    print("\n" + "=" * 70 + "\n  DONE  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())) + "\n" + "=" * 70)


if __name__ == '__main__':
    main()
