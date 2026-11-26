#!/usr/bin/env python3
"""
================================================================================
CPB TRAINING — NUMBA / VECTORIZED  (Pearson selection)  + CLEAN-FRAME FIX   v2
================================================================================
CPB trainer with two reproduction-gap fixes added (a weak CPB caps the
pipeline; the cause is sigma inflation from foreground-contaminated training
frames).

WHAT'S NEW vs the original file  (both changes are OFF unless you ask for them,
so default behaviour is byte-identical to before):

  FIX 1 — FRAME CLEANING  (--clean_frac F, default 1.0 = off)
      Before training a scene, build a per-pixel temporal-median background from
      the pre-ROI frames, score each frame by its foreground fraction
      (share of pixels deviating > --fg_thr from that background), and KEEP only
      the cleanest fraction F of frames.  This removes the frames full of moving
      cars/people that corrupt the background statistics.  Example: --clean_frac 0.7
      keeps the cleanest 70% of the pre-ROI frames.  New function: clean_frames().

  FIX 2 — ROBUST SIGMA  (--robust_sigma {std,mad}, default std)
      'mad' replaces the stored per-support sigma with 1.4826*MAD of the residual
      (block - pixel) over the (cleaned) training stack, i.e. a median-absolute-
      deviation estimator that foreground spikes cannot inflate, instead of the
      plain standard deviation.  New function: robust_sigma_pass().

Why both: FIX 1 removes most contamination at the frame level; FIX 2 makes the
per-support scale robust to whatever foreground survives.  Used together they are
the "robust median/MAD + cleaner foreground-free frames" the authors leave
unstated (paper Sec. 4.2.1/4.2.3).  On already-clean scenes they are near-neutral;
on busy scenes (dynamicBackground, turbulence, IOM) they recover CPB recall.

Model .pkl format is unchanged, so cpb_kernels.py loads it as before.

USAGE
  # faithful (unchanged):
  python cpb_training_base.py --output_dir models/cpb_clean
  # with both fixes (recommended for CDnet):
  python cpb_training_base.py --output_dir models/cpb_clean \
         --clean_frac 0.7 --fg_thr 30 --robust_sigma mad --force
================================================================================
"""

import argparse, json, os, pickle, re, sys, time, warnings
import cv2, numpy as np
warnings.filterwarnings('ignore')

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return (deco(a[0]) if (a and callable(a[0])) else deco)
    prange = range

# ---- paper params ----
BLOCK_SIZE = 8
K_SUPPORT = 20
NC_CANDIDATE = 10
POOLNUM = K_SUPPORT + NC_CANDIDATE
ETA = 2.5; LAMBDA_DEC = 0.5
MIN_GAMMA_DEFAULT = 0.0
SIGMA_FLOOR_MIN = 3.0
CLIP_KAPPA = 3.0; CLIP_ITERS = 4
CALIB_FRACTION = 0.20; CALIB_MIN = 30; CALIB_MAX = 100; FIT_MIN = 150
N_TRAIN = 400; MIN_FRAMES = 10; MAX_TRAINING_PIXELS = 250000; BIG = 1e18

ALL_CDNET_SCENES = {
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

# ============================================================================ data
def find_root():
    here = os.path.abspath('.')
    if os.path.basename(here) == 'cdnet2014' and os.path.exists(os.path.join(here, 'baseline')):
        return here
    for p in ['data/cdnet2014', '../data/cdnet2014', '.', ]:
        if os.path.exists(os.path.join(p, 'baseline')):
            return os.path.abspath(p)
    return None

def temporal_roi(scene_path):
    f = os.path.join(scene_path, 'temporalROI.txt')
    if not os.path.exists(f): return 1, 99999
    txt = open(f).read().strip().replace('-', ' ').split()
    if len(txt) >= 2: return int(txt[0]), int(txt[1])
    if len(txt) == 1: return 1, int(txt[0])
    return 1, 99999

def list_nums(input_dir):
    ns = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(('.jpg', '.png')):
            m = re.findall(r'\d+', fn)
            if m: ns.append(int(m[-1]))
    return sorted(ns)

def pre_roi_nums(scene_path, n_train):
    nums = list_nums(os.path.join(scene_path, 'input'))
    rs, re_ = temporal_roi(scene_path)
    pre = [n for n in nums if n < rs] or nums[:n_train]
    if len(pre) > n_train: pre = pre[-n_train:]
    return pre, rs, re_

def split_fit_calib(pre):
    n = len(pre)
    nc = max(CALIB_MIN, min(CALIB_MAX, int(round(n * CALIB_FRACTION))))
    nc = min(nc, max(0, n - FIT_MIN))
    if nc <= 0: return pre, []
    return pre[nc:], pre[:nc]

def _read_frame(scene_path, num):
    d = os.path.join(scene_path, 'input')
    for pat in (f'in{num:06d}.jpg', f'in{num:05d}.jpg', f'{num:06d}.jpg', f'in{num:06d}.png'):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None: return im
    return None

def probe_native(scene_path, num):
    im = _read_frame(scene_path, num)
    return None if im is None else im.shape[:2]

def plan_size(H0, W0, cap):
    if H0 * W0 <= cap: return None, H0, W0
    s = np.sqrt(cap / (H0 * W0))
    nH = max(BLOCK_SIZE, (int(H0 * s) // BLOCK_SIZE) * BLOCK_SIZE)
    nW = max(BLOCK_SIZE, (int(W0 * s) // BLOCK_SIZE) * BLOCK_SIZE)
    return (nW, nH), nH, nW


# ============================================================================ FIX 1: frame cleaning
def clean_frames(scene_path, nums, target_size, clean_frac, fg_thr):
    """Keep only the cleanest `clean_frac` of `nums`, judged by foreground
    fraction against a temporal-median background. Returns (kept_nums, stats)."""
    if clean_frac >= 1.0 or len(nums) <= MIN_FRAMES:
        return nums, {'kept': len(nums), 'total': len(nums), 'fg_med': None}
    imgs, keep_idx = [], []
    for k, num in enumerate(nums):
        im = _read_frame(scene_path, num)
        if im is None: continue
        if target_size is not None: im = cv2.resize(im, target_size)
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)); keep_idx.append(num)
    if len(imgs) <= MIN_FRAMES:
        return keep_idx, {'kept': len(keep_idx), 'total': len(keep_idx), 'fg_med': None}
    stack = np.stack(imgs, 0)                          # [T,H,W,3]
    bg = np.median(stack, 0)                           # temporal-median background
    dev = np.abs(stack - bg).max(3)                    # [T,H,W] max-channel deviation
    fgfrac = (dev > fg_thr).mean((1, 2))               # per-frame foreground fraction
    nkeep = max(MIN_FRAMES, int(round(clean_frac * len(keep_idx))))
    order = np.argsort(fgfrac)                         # cleanest first
    kept = sorted(keep_idx[i] for i in order[:nkeep])
    return kept, {'kept': nkeep, 'total': len(keep_idx),
                  'fg_min': float(fgfrac.min()), 'fg_med': float(np.median(fgfrac)),
                  'fg_max': float(fgfrac.max()), 'fg_cut': float(np.sort(fgfrac)[nkeep-1])}


def load_stacks(scene_path, nums, target_size, channels):
    d = os.path.join(scene_path, 'input'); imgs = []
    for num in nums:
        img = _read_frame(scene_path, num)
        if img is None: continue
        if target_size is not None: img = cv2.resize(img, target_size)
        imgs.append(img)
    if not imgs: return None, None
    T = len(imgs); H, W = imgs[0].shape[:2]
    if channels == 'gray':
        L = np.empty((T, H, W), np.float32)
        for t, fr in enumerate(imgs):
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)
            L[t] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return {'L': L}, (H, W)
    R = np.empty((T, H, W), np.float32); G = np.empty((T, H, W), np.float32); B = np.empty((T, H, W), np.float32)
    for t, fr in enumerate(imgs):
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        R[t], G[t], B[t] = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return {'R': R, 'G': G, 'B': B}, (H, W)

def block_mean_stack(gray_stack, ws, hs, bs=BLOCK_SIZE):
    T, H, W = gray_stack.shape
    pad = np.zeros((T, ws * bs, hs * bs), np.float32); pad[:, :H, :W] = gray_stack
    r = pad.reshape(T, ws, bs, hs, bs).mean(axis=(2, 4))
    return np.ascontiguousarray(r.transpose(1, 2, 0), np.float32)

def block_mean_frame(gray, ws, hs, bs=BLOCK_SIZE):
    H, W = gray.shape
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = gray
    return pad.reshape(ws, bs, hs, bs).mean(axis=(1, 3))


# ============================================================================ kernel (unchanged selection + plain-std sigma)
@njit(cache=False, parallel=True, fastmath=False)
def train_kernel(gray_stack, DBL, poolnum, sfloor, min_gamma, out):
    T, H, W = gray_stack.shape
    ws, hs = DBL.shape[0], DBL.shape[1]
    for x in prange(H):
        bg = np.full(poolnum, -2.0); bi_ = np.full(poolnum, -1); bj_ = np.full(poolnum, -1)
        bb_ = np.zeros(poolnum); bs_ = np.zeros(poolnum)
        for y in range(W):
            for q in range(poolnum):
                bg[q] = -2.0; bi_[q] = -1; bj_[q] = -1; bb_[q] = 0.0; bs_[q] = 0.0
            mp = 0.0
            for t in range(T): mp += gray_stack[t, x, y]
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
                        if k >= poolnum: break
                        if bi == self_bi and bj == self_bj: continue
                        out[x, y, k, 0] = bi; out[x, y, k, 1] = bj
                        out[x, y, k, 2] = 0.0; out[x, y, k, 3] = sfloor; out[x, y, k, 4] = 0.0
                        k += 1
                continue
            filled = 0; worst = BIG; worst_idx = 0
            for bi in range(ws):
                for bj in range(hs):
                    if bi == self_bi and bj == self_bj: continue
                    mb = 0.0
                    for t in range(T): mb += DBL[bi, bj, t]
                    mb /= T
                    cov = 0.0; vb = 0.0
                    for t in range(T):
                        pp = gray_stack[t, x, y] - mp; qq = DBL[bi, bj, t] - mb
                        cov += pp * qq; vb += qq * qq
                    sb_sum = vb ** 0.5
                    g = 0.0 if sb_sum < 1e-9 else cov / (sp_sum * sb_sum)
                    if g > 1.0: g = 1.0
                    elif g < -1.0: g = -1.0
                    b = mp - mb
                    var = (sp2 - 2.0 * cov + vb) / T
                    if var < 0.0: var = 0.0
                    sig = var ** 0.5
                    if filled < poolnum:
                        bg[filled] = g; bi_[filled] = bi; bj_[filled] = bj; bb_[filled] = b; bs_[filled] = sig
                        filled += 1
                        if filled == poolnum:
                            worst = bg[0]; worst_idx = 0
                            for q in range(1, poolnum):
                                if bg[q] < worst: worst = bg[q]; worst_idx = q
                    elif g > worst:
                        bg[worst_idx] = g; bi_[worst_idx] = bi; bj_[worst_idx] = bj; bb_[worst_idx] = b; bs_[worst_idx] = sig
                        worst = bg[0]; worst_idx = 0
                        for q in range(1, poolnum):
                            if bg[q] < worst: worst = bg[q]; worst_idx = q
            for a in range(1, poolnum):
                gv = bg[a]; iv = bi_[a]; jv = bj_[a]; bv = bb_[a]; sv = bs_[a]; c = a - 1
                while c >= 0 and bg[c] < gv:
                    bg[c + 1] = bg[c]; bi_[c + 1] = bi_[c]; bj_[c + 1] = bj_[c]; bb_[c + 1] = bb_[c]; bs_[c + 1] = bs_[c]; c -= 1
                bg[c + 1] = gv; bi_[c + 1] = iv; bj_[c + 1] = jv; bb_[c + 1] = bv; bs_[c + 1] = sv
            for k in range(poolnum):
                bi = bi_[k]; bj = bj_[k]
                if bi < 0 or bg[k] < min_gamma:
                    out[x, y, k, 0] = bi if bi >= 0 else 0; out[x, y, k, 1] = bj if bj >= 0 else 0
                    out[x, y, k, 2] = 0.0; out[x, y, k, 3] = sfloor; out[x, y, k, 4] = 0.0
                    continue
                sig = bs_[k]
                if sig < sfloor: sig = sfloor
                out[x, y, k, 0] = bi; out[x, y, k, 1] = bj
                out[x, y, k, 2] = bb_[k]; out[x, y, k, 3] = sig; out[x, y, k, 4] = bg[k]


# ============================================================================ FIX 2: robust (MAD) sigma post-pass
def robust_sigma_pass(gray_stack, DBLK, sfloor, K=K_SUPPORT):
    """Replace stored sigma with 1.4826*MAD of (block_mean - pixel) over the
    (cleaned) training stack, for the top-K supports of every pixel. Foreground
    spikes that survive frame-cleaning cannot inflate a MAD the way they inflate
    a std. Runs in numpy (vectorised per frame); minutes for a CDnet scene."""
    T, H, W = gray_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    ii = np.clip(DBLK[:, :, :K, 0].astype(np.int64), 0, ws - 1)   # [H,W,K]
    jj = np.clip(DBLK[:, :, :K, 1].astype(np.int64), 0, hs - 1)
    b  = DBLK[:, :, :K, 2]
    # collect residuals over time: r_t = block_mean_t[ii,jj] - pixel_t - b
    res = np.empty((T, H, W, K), np.float32)
    for t in range(T):
        bm = block_mean_frame(gray_stack[t], ws, hs)             # [ws,hs]
        res[t] = (bm[ii, jj] - gray_stack[t][:, :, None]) - b
    med = np.median(res, 0)                                       # [H,W,K]
    mad = 1.4826 * np.median(np.abs(res - med[None]), 0)
    DBLK[:, :, :K, 3] = np.maximum(mad, sfloor)
    return DBLK

def train_channel(gray_stack, poolnum, min_gamma, sfloor, cname, robust_sigma):
    T, H, W = gray_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    DBL = block_mean_stack(gray_stack, ws, hs)
    out = np.zeros((H, W, poolnum, 5), np.float32)
    t0 = time.time()
    train_kernel(np.ascontiguousarray(gray_stack), DBL, poolnum, sfloor, min_gamma, out)
    s_before = float(np.median(out[:, :, :K_SUPPORT, 3]))
    if robust_sigma == 'mad':
        out = robust_sigma_pass(gray_stack, out, sfloor)
    s_after = float(np.median(out[:, :, :K_SUPPORT, 3]))
    g = out[:, :, :K_SUPPORT, 4]
    print(f"        [{cname}] {ws}x{hs} grid  T={T}  {H*W:,}px  {time.time()-t0:.0f}s  "
          f"gamma(topK) mean={np.mean(g):+.3f}  sigma(topK) med {s_before:.2f}->{s_after:.2f}", flush=True)
    return out


# ============================================================================ sigma-floor calibration (unchanged)
def _clipped_std(d, kappa=CLIP_KAPPA, iters=CLIP_ITERS):
    c = float(np.median(d)); s = float(np.median(np.abs(d - c)) * 1.4826)
    if s < 1e-9: s = float(d.std(ddof=0))
    for _ in range(iters):
        keep = np.abs(d - c) <= kappa * max(s, 1e-9)
        if keep.sum() < 8: break
        c = float(d[keep].mean()); s = float(d[keep].std(ddof=0))
    return c, s

def calibrate_floor(calib_stack, DBLK, K=K_SUPPORT):
    Tc, H, W = calib_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    D = DBLK[:, :, :K, :]
    ii = np.clip(D[..., 0].astype(np.int64), 0, ws - 1); jj = np.clip(D[..., 1].astype(np.int64), 0, hs - 1)
    b = D[..., 2]; valid = D[..., 4] > 0.0; samples = []
    for t in range(0, Tc, max(1, Tc // 40)):
        bm = block_mean_frame(calib_stack[t], ws, hs)[ii, jj]
        r = ((calib_stack[t][:, :, None] - bm) - b)[valid]
        if r.size:
            if r.size > 200000: r = r[np.random.default_rng(t).integers(0, r.size, 200000)]
            samples.append(r.astype(np.float32))
    if not samples: return SIGMA_FLOOR_MIN
    _, s = _clipped_std(np.concatenate(samples))
    return float(max(s, SIGMA_FLOOR_MIN))


# ============================================================================ save
def save_model(D, category, scene, out_dir, img_size, original_size, target_size,
               rs, re_, min_gamma, floors, channels, clean_stats, robust_sigma):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    obj = {'DBLK_r': D['R'], 'DBLK_g': D['G'], 'DBLK_b': D['B'],
           'category': category, 'scene': scene, 'img_size': img_size,
           'original_size': original_size, 'target_size': target_size,
           'params': {'block_size': BLOCK_SIZE, 'K': K_SUPPORT, 'NC': NC_CANDIDATE, 'poolnum': POOLNUM,
                      'eta': ETA, 'lambda_thresh': LAMBDA_DEC, 'sigma_floor': SIGMA_FLOOR_MIN,
                      'sigma_floor_calibrated': floors, 'min_gamma': min_gamma,
                      'stat': robust_sigma, 'selection': 'pearson', 'channels': channels,
                      'clean_stats': clean_stats, 'roi_start': rs, 'roi_end': re_},
           'version': 'cpb_numba_pearson_clean_v2', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


# ============================================================================ per scene
def train_scene(root, category, scene, out_dir, args):
    sp = os.path.join(root, category, scene)
    if not os.path.isdir(sp): print(f"    [SKIP] not found: {sp}"); return 'missing'
    out_path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    if os.path.exists(out_path) and not args.force:
        print(f"    [resume] already trained: {category}/{scene}"); return 'exists'
    pre, rs, re_ = pre_roi_nums(sp, args.n_train)
    if len(pre) < MIN_FRAMES: print(f"    [SKIP] too few frames: {scene}"); return 'too_few'
    native = probe_native(sp, pre[0])
    if native is None: print(f"    [SKIP] unreadable: {scene}"); return 'unreadable'
    H0, W0 = native
    target_size, Hf, Wf = plan_size(H0, W0, args.max_train_pixels)

    # ----- FIX 1: frame cleaning on the pre-ROI set, BEFORE the fit/calib split -----
    pre_clean, cstats = clean_frames(sp, pre, target_size, args.clean_frac, args.fg_thr)
    if cstats.get('fg_med') is not None:
        print(f"    [clean] fg-frac min/med/max={cstats['fg_min']:.3f}/{cstats['fg_med']:.3f}/"
              f"{cstats['fg_max']:.3f}  keep {cstats['kept']}/{cstats['total']} (cut<= {cstats['fg_cut']:.3f})")
    fit, calib = split_fit_calib(pre_clean) if not args.no_calib_floor else (pre_clean, [])

    print(f"\n  > {category}/{scene}  native {W0}x{H0} -> fit {Wf}x{Hf}  pool={args.poolnum}  "
          f"channels={args.channels}  robust_sigma={args.robust_sigma}  frames(fit={len(fit)},calib={len(calib)})")
    stacks, img_size = load_stacks(sp, fit, target_size, args.channels)
    if stacks is None: print(f"    [SKIP] load failed: {scene}"); return 'load_failed'
    H, W = img_size; t0 = time.time(); chan = {}
    if args.channels == 'gray':
        chan['L'] = train_channel(stacks['L'], args.poolnum, args.min_gamma, SIGMA_FLOOR_MIN, 'L', args.robust_sigma)
    else:
        for nm in ('R', 'G', 'B'):
            print(f"    -- {nm} --")
            chan[nm] = train_channel(stacks[nm], args.poolnum, args.min_gamma, SIGMA_FLOOR_MIN, nm, args.robust_sigma)
    del stacks
    floors = {}
    if calib:
        cstacks, _ = load_stacks(sp, calib, target_size, args.channels)
        if args.channels == 'gray':
            f = calibrate_floor(cstacks['L'], chan['L']); chan['L'][..., 3] = np.maximum(chan['L'][..., 3], f)
            floors = {'R': f, 'G': f, 'B': f}; print(f"    held-out sigma floor = {f:.2f}")
        else:
            for nm in ('R', 'G', 'B'):
                f = calibrate_floor(cstacks[nm], chan[nm]); chan[nm][..., 3] = np.maximum(chan[nm][..., 3], f); floors[nm] = f
            print(f"    held-out sigma floors = {floors}")
        del cstacks
    else:
        floors = {'R': SIGMA_FLOOR_MIN, 'G': SIGMA_FLOOR_MIN, 'B': SIGMA_FLOOR_MIN}
    D = {'R': chan['L'], 'G': chan['L'], 'B': chan['L']} if args.channels == 'gray' else chan
    print(f"    total {time.time()-t0:.0f}s")
    save_model(D, category, scene, out_dir, img_size, (H0, W0), target_size, rs, re_,
               args.min_gamma, floors, args.channels, cstats, args.robust_sigma)
    return 'trained'


def main():
    ap = argparse.ArgumentParser(description='CPB training — Numba/Pearson + clean-frame fix',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--output_dir', default='models/cpb_paper')
    ap.add_argument('--category', default=None); ap.add_argument('--scene', default=None)
    ap.add_argument('--channels', choices=['gray', 'rgb'], default='gray')
    ap.add_argument('--n_train', type=int, default=N_TRAIN)
    ap.add_argument('--poolnum', type=int, default=POOLNUM)
    ap.add_argument('--min_gamma', type=float, default=MIN_GAMMA_DEFAULT)
    ap.add_argument('--max_train_pixels', type=int, default=MAX_TRAINING_PIXELS)
    ap.add_argument('--no_calib_floor', action='store_true')
    # ---- the two new fixes ----
    ap.add_argument('--clean_frac', type=float, default=1.0,
                    help='FIX 1: keep only the cleanest this fraction of pre-ROI frames (1.0=off; try 0.7)')
    ap.add_argument('--fg_thr', type=float, default=30.0,
                    help='FIX 1: per-pixel deviation (0-255) from the temporal-median bg that counts as foreground')
    ap.add_argument('--robust_sigma', choices=['std', 'mad'], default='std',
                    help='FIX 2: sigma estimator; mad = 1.4826*MAD (robust to foreground), std = plain (original)')
    ap.add_argument('--force', action='store_true'); ap.add_argument('--list_scenes', action='store_true')
    args = ap.parse_args()
    if args.list_scenes:
        for c, s in ALL_CDNET_SCENES.items(): print(f"  {c}: {s}")
        return
    root = args.dataset_root or find_root()
    if not root: sys.exit('ERROR: CDNet2014 not found — use --dataset_root')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}\n  CPB TRAINING (Numba, Pearson, {args.channels})  "
          f"clean_frac={args.clean_frac} fg_thr={args.fg_thr} robust_sigma={args.robust_sigma}")
    print(f"  engine={'numba' if HAVE_NUMBA else 'PYTHON-FALLBACK(slow)'}  root={root}  out={args.output_dir}\n{'='*70}")
    cats = [args.category] if args.category else list(ALL_CDNET_SCENES.keys())
    tally = {}
    for cat in cats:
        if cat not in ALL_CDNET_SCENES: print(f"  [WARN] unknown category {cat}"); continue
        for sc in ALL_CDNET_SCENES[cat]:
            if args.scene and sc != args.scene: continue
            try: st = train_scene(root, cat, sc, args.output_dir, args)
            except Exception as e: print(f"    [ERROR] {cat}/{sc}: {e}"); st = 'error'
            tally[st] = tally.get(st, 0) + 1
    print("\n" + "=" * 70 + "\n  DONE  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())) + "\n" + "=" * 70)


if __name__ == '__main__':
    main()
