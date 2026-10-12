#!/usr/bin/env python3
"""
================================================================================
CPB TRAINING — NUMBA / VECTORIZED  (Pearson selection, Liang et al. Eq.1-3)
================================================================================
Same math as the multiprocessing trainer, but the per-pixel loop is a single
@njit(parallel=True) kernel instead of a ProcessPool — ~1-2 orders of magnitude
faster and far lighter on RAM (no shared-memory copies, no per-pixel task args).

FAITHFUL CORE:
  * supports selected by the POOLNUM highest Pearson correlations   (Eq.1)
  * b     = mean_t( I_p - I_Qblock )                                (Eq.3 mean)
  * sigma = std_t ( I_p - I_Qblock )                               (Eq.3 std, plain Gaussian)
  * gamma = Pearson(I_p, I_Q)   (raw, the decision weight)
  * held-out sigma-floor calibration (optional), sigma_eff = max(sigma, floor)

DIFFERENCES vs the old file (both intentional):
  * NO SCENE IS SKIPPED. Large scenes are downsampled to --max_train_pixels
    (default 250k) and trained; nothing is dropped. Set --max_train_pixels huge
    to train at native resolution.
  * GRAY (luma) by default — this is what the framework loads (ch=['L']) and
    what every validated result used. Stored identically in R/G/B so the model
    format is unchanged. Use --channels rgb for the 3-channel variant.
  * sigma estimator is plain std (paper Eq.3). --stat mad also available.
    (clipped_std dropped: we found it over-robust on dynamic scenes.)

Output model format is identical in structure to the existing .pkl files, so
opus_final_hod_framework.py loads them with no change.

USAGE
  python opus_final_hod_training.py --output_dir models/cpb_numba
  python opus_final_hod_training.py --output_dir models/cpb_numba --category shadow
  python opus_final_hod_training.py --output_dir models/cpb_numba --max_train_pixels 100000000 --force
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
        def deco(f):
            return f
        return (deco(a[0]) if (a and callable(a[0])) else deco)
    prange = range

# ---- paper params ----
BLOCK_SIZE = 8
K_SUPPORT = 20
NC_CANDIDATE = 10
POOLNUM = K_SUPPORT + NC_CANDIDATE      # 30
ETA = 2.5
LAMBDA_DEC = 0.5
MIN_GAMMA_DEFAULT = 0.0
SIGMA_FLOOR_MIN = 3.0
CLIP_KAPPA = 3.0
CLIP_ITERS = 4
CALIB_FRACTION = 0.20
CALIB_MIN = 30
CALIB_MAX = 100
FIT_MIN = 150
N_TRAIN = 400
MIN_FRAMES = 10
MAX_TRAINING_PIXELS = 250000
BIG = 1e18

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
    for p in ['data/cdnet2014', '../data/cdnet2014', '.', '/media/data/cdnet2014', '/mnt/data/cdnet2014']:
        if os.path.exists(os.path.join(p, 'baseline')):
            return os.path.abspath(p)
    return None


def temporal_roi(scene_path):
    f = os.path.join(scene_path, 'temporalROI.txt')
    if not os.path.exists(f):
        return 1, 99999
    txt = open(f).read().strip().replace('-', ' ').split()
    if len(txt) >= 2:
        return int(txt[0]), int(txt[1])
    if len(txt) == 1:
        return 1, int(txt[0])
    return 1, 99999


def list_nums(input_dir):
    ns = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(('.jpg', '.png')):
            m = re.findall(r'\d+', fn)
            if m:
                ns.append(int(m[-1]))
    return sorted(ns)


def pre_roi_nums(scene_path, n_train):
    nums = list_nums(os.path.join(scene_path, 'input'))
    rs, re_ = temporal_roi(scene_path)
    pre = [n for n in nums if n < rs] or nums[:n_train]
    if len(pre) > n_train:
        pre = pre[-n_train:]
    return pre, rs, re_


def split_fit_calib(pre):
    n = len(pre)
    nc = max(CALIB_MIN, min(CALIB_MAX, int(round(n * CALIB_FRACTION))))
    nc = min(nc, max(0, n - FIT_MIN))
    if nc <= 0:
        return pre, []
    return pre[nc:], pre[:nc]


def probe_native(scene_path, num):
    d = os.path.join(scene_path, 'input')
    for pat in (f'in{num:06d}.jpg', f'in{num:05d}.jpg', f'{num:06d}.jpg', f'in{num:06d}.png'):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return im.shape[:2]
    return None


def plan_size(H0, W0, cap):
    if H0 * W0 <= cap:
        return None, H0, W0
    s = np.sqrt(cap / (H0 * W0))
    nH = max(BLOCK_SIZE, (int(H0 * s) // BLOCK_SIZE) * BLOCK_SIZE)
    nW = max(BLOCK_SIZE, (int(W0 * s) // BLOCK_SIZE) * BLOCK_SIZE)
    return (nW, nH), nH, nW


def load_stacks(scene_path, nums, target_size, channels):
    """Return dict of [T,H,W] float32 stacks. gray -> {'L':...}; rgb -> {'R','G','B'}."""
    d = os.path.join(scene_path, 'input')
    imgs = []
    for num in nums:
        img = None
        for pat in (f'in{num:06d}.jpg', f'in{num:05d}.jpg', f'{num:06d}.jpg', f'in{num:06d}.png'):
            p = os.path.join(d, pat)
            if os.path.exists(p):
                img = cv2.imread(p)
                break
        if img is None:
            continue
        if target_size is not None:
            img = cv2.resize(img, target_size)
        imgs.append(img)
    if not imgs:
        return None, None
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
    return np.ascontiguousarray(r.transpose(1, 2, 0), np.float32)   # [ws,hs,T]


def block_mean_frame(gray, ws, hs, bs=BLOCK_SIZE):
    H, W = gray.shape
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = gray
    return pad.reshape(ws, bs, hs, bs).mean(axis=(1, 3))


# ============================================================================ kernel
@njit(cache=False, parallel=True, fastmath=False)
def train_kernel(gray_stack, DBL, poolnum, sfloor, min_gamma, out):
    """Per pixel: select the POOLNUM blocks with the HIGHEST Pearson correlation.
    Pearson, b=mean(I_p-I_Q) and sigma=std(I_p-I_Q) all come from one pass of
    accumulators (sp2 - 2cov + vb == sum of squared centred differences)."""
    T, H, W = gray_stack.shape
    ws, hs = DBL.shape[0], DBL.shape[1]
    for x in prange(H):
        bg = np.full(poolnum, -2.0)     # best gammas (keep the highest)
        bi_ = np.full(poolnum, -1)
        bj_ = np.full(poolnum, -1)
        bb_ = np.zeros(poolnum)
        bs_ = np.zeros(poolnum)
        for y in range(W):
            for q in range(poolnum):
                bg[q] = -2.0; bi_[q] = -1; bj_[q] = -1; bb_[q] = 0.0; bs_[q] = 0.0
            mp = 0.0
            for t in range(T):
                mp += gray_stack[t, x, y]
            mp /= T
            sp2 = 0.0
            for t in range(T):
                d = gray_stack[t, x, y] - mp
                sp2 += d * d
            sp_sum = sp2 ** 0.5
            self_bi = x // BLOCK_SIZE; self_bj = y // BLOCK_SIZE
            if sp_sum < 1e-6:           # constant pixel -> zero-weight placeholders
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
            filled = 0; worst = BIG; worst_idx = 0
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
                        pp = gray_stack[t, x, y] - mp
                        qq = DBL[bi, bj, t] - mb
                        cov += pp * qq; vb += qq * qq
                    sb_sum = vb ** 0.5
                    if sb_sum < 1e-9:
                        g = 0.0
                    else:
                        g = cov / (sp_sum * sb_sum)
                    if g > 1.0:
                        g = 1.0
                    elif g < -1.0:
                        g = -1.0
                    b = mp - mb
                    var = (sp2 - 2.0 * cov + vb) / T
                    if var < 0.0:
                        var = 0.0
                    sig = var ** 0.5
                    if filled < poolnum:
                        bg[filled] = g; bi_[filled] = bi; bj_[filled] = bj; bb_[filled] = b; bs_[filled] = sig
                        filled += 1
                        if filled == poolnum:
                            worst = bg[0]; worst_idx = 0
                            for q in range(1, poolnum):
                                if bg[q] < worst:
                                    worst = bg[q]; worst_idx = q
                    elif g > worst:
                        bg[worst_idx] = g; bi_[worst_idx] = bi; bj_[worst_idx] = bj; bb_[worst_idx] = b; bs_[worst_idx] = sig
                        worst = bg[0]; worst_idx = 0
                        for q in range(1, poolnum):
                            if bg[q] < worst:
                                worst = bg[q]; worst_idx = q
            # sort selected by gamma DESCENDING (insertion)
            for a in range(1, poolnum):
                gv = bg[a]; iv = bi_[a]; jv = bj_[a]; bv = bb_[a]; sv = bs_[a]; c = a - 1
                while c >= 0 and bg[c] < gv:
                    bg[c + 1] = bg[c]; bi_[c + 1] = bi_[c]; bj_[c + 1] = bj_[c]; bb_[c + 1] = bb_[c]; bs_[c + 1] = bs_[c]; c -= 1
                bg[c + 1] = gv; bi_[c + 1] = iv; bj_[c + 1] = jv; bb_[c + 1] = bv; bs_[c + 1] = sv
            for k in range(poolnum):
                bi = bi_[k]; bj = bj_[k]
                if bi < 0 or bg[k] < min_gamma:
                    out[x, y, k, 0] = bi if bi >= 0 else 0
                    out[x, y, k, 1] = bj if bj >= 0 else 0
                    out[x, y, k, 2] = 0.0; out[x, y, k, 3] = sfloor; out[x, y, k, 4] = 0.0
                    continue
                sig = bs_[k]
                if sig < sfloor:
                    sig = sfloor
                out[x, y, k, 0] = bi; out[x, y, k, 1] = bj
                out[x, y, k, 2] = bb_[k]; out[x, y, k, 3] = sig; out[x, y, k, 4] = bg[k]


def train_channel(gray_stack, poolnum, min_gamma, sfloor, cname):
    T, H, W = gray_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    DBL = block_mean_stack(gray_stack, ws, hs)
    out = np.zeros((H, W, poolnum, 5), np.float32)
    t0 = time.time()
    train_kernel(np.ascontiguousarray(gray_stack), DBL, poolnum, sfloor, min_gamma, out)
    g = out[:, :, :K_SUPPORT, 4]; s = out[:, :, :K_SUPPORT, 3]
    print(f"        [{cname}] {ws}x{hs} grid  T={T}  {H*W:,}px  {time.time()-t0:.0f}s  "
          f"gamma(topK) mean={np.mean(g):+.3f}  sigma(topK) med={np.median(s):.2f}", flush=True)
    return out


# ============================================================================ sigma-floor calibration
def _clipped_std(d, kappa=CLIP_KAPPA, iters=CLIP_ITERS):
    c = float(np.median(d)); s = float(np.median(np.abs(d - c)) * 1.4826)
    if s < 1e-9:
        s = float(d.std(ddof=0))
    for _ in range(iters):
        keep = np.abs(d - c) <= kappa * max(s, 1e-9)
        if keep.sum() < 8:
            break
        c = float(d[keep].mean()); s = float(d[keep].std(ddof=0))
    return c, s


def calibrate_floor(calib_stack, DBLK, K=K_SUPPORT):
    Tc, H, W = calib_stack.shape
    ws = int(np.ceil(H / BLOCK_SIZE)); hs = int(np.ceil(W / BLOCK_SIZE))
    D = DBLK[:, :, :K, :]
    ii = np.clip(D[..., 0].astype(np.int64), 0, ws - 1)
    jj = np.clip(D[..., 1].astype(np.int64), 0, hs - 1)
    b = D[..., 2]; valid = D[..., 4] > 0.0
    samples = []
    for t in range(0, Tc, max(1, Tc // 40)):
        bm = block_mean_frame(calib_stack[t], ws, hs)[ii, jj]
        r = ((calib_stack[t][:, :, None] - bm) - b)[valid]
        if r.size:
            if r.size > 200000:
                r = r[np.random.default_rng(t).integers(0, r.size, 200000)]
            samples.append(r.astype(np.float32))
    if not samples:
        return SIGMA_FLOOR_MIN
    _, s = _clipped_std(np.concatenate(samples))
    return float(max(s, SIGMA_FLOOR_MIN))


# ============================================================================ save
def save_model(D, category, scene, out_dir, img_size, original_size, target_size,
               rs, re_, min_gamma, floors, channels):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    obj = {'DBLK_r': D['R'], 'DBLK_g': D['G'], 'DBLK_b': D['B'],
           'category': category, 'scene': scene, 'img_size': img_size,
           'original_size': original_size, 'target_size': target_size,
           'params': {'block_size': BLOCK_SIZE, 'K': K_SUPPORT, 'NC': NC_CANDIDATE, 'poolnum': POOLNUM,
                      'eta': ETA, 'lambda_thresh': LAMBDA_DEC, 'sigma_floor': SIGMA_FLOOR_MIN,
                      'sigma_floor_calibrated': floors, 'min_gamma': min_gamma, 'stat': 'std',
                      'selection': 'pearson', 'channels': channels,
                      'roi_start': rs, 'roi_end': re_},
           'version': 'cpb_numba_pearson_v1', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


# ============================================================================ per scene
def train_scene(root, category, scene, out_dir, args):
    sp = os.path.join(root, category, scene)
    if not os.path.isdir(sp):
        print(f"    [SKIP] not found: {sp}"); return 'missing'
    out_path = os.path.join(out_dir, f'cpb_paper_{category}_{scene}.pkl')
    if os.path.exists(out_path) and not args.force:
        print(f"    [resume] already trained: {category}/{scene}"); return 'exists'

    pre, rs, re_ = pre_roi_nums(sp, args.n_train)
    if len(pre) < MIN_FRAMES:
        print(f"    [SKIP] too few frames: {scene}"); return 'too_few'
    native = probe_native(sp, pre[0])
    if native is None:
        print(f"    [SKIP] unreadable: {scene}"); return 'unreadable'
    H0, W0 = native
    target_size, Hf, Wf = plan_size(H0, W0, args.max_train_pixels)
    fit, calib = split_fit_calib(pre) if not args.no_calib_floor else (pre, [])

    print(f"\n  > {category}/{scene}  native {W0}x{H0} -> fit {Wf}x{Hf}  "
          f"pool={args.poolnum}  channels={args.channels}  frames(fit={len(fit)},calib={len(calib)})")
    stacks, img_size = load_stacks(sp, fit, target_size, args.channels)
    if stacks is None:
        print(f"    [SKIP] load failed: {scene}"); return 'load_failed'
    H, W = img_size
    t0 = time.time()
    chan = {}
    if args.channels == 'gray':
        chan['L'] = train_channel(stacks['L'], args.poolnum, args.min_gamma, SIGMA_FLOOR_MIN, 'L')
    else:
        for nm in ('R', 'G', 'B'):
            print(f"    -- {nm} --")
            chan[nm] = train_channel(stacks[nm], args.poolnum, args.min_gamma, SIGMA_FLOOR_MIN, nm)
    del stacks

    floors = {}
    if calib:
        cstacks, _ = load_stacks(sp, calib, target_size, args.channels)
        if args.channels == 'gray':
            f = calibrate_floor(cstacks['L'], chan['L']); chan['L'][..., 3] = np.maximum(chan['L'][..., 3], f)
            floors = {'R': f, 'G': f, 'B': f}
            print(f"    held-out sigma floor = {f:.2f}")
        else:
            for nm in ('R', 'G', 'B'):
                f = calibrate_floor(cstacks[nm], chan[nm]); chan[nm][..., 3] = np.maximum(chan[nm][..., 3], f)
                floors[nm] = f
            print(f"    held-out sigma floors = {floors}")
        del cstacks
    else:
        floors = {'R': SIGMA_FLOOR_MIN, 'G': SIGMA_FLOOR_MIN, 'B': SIGMA_FLOOR_MIN}

    if args.channels == 'gray':
        D = {'R': chan['L'], 'G': chan['L'], 'B': chan['L']}     # identical channels -> framework auto ch=['L']
    else:
        D = chan
    print(f"    total {time.time()-t0:.0f}s")
    save_model(D, category, scene, out_dir, img_size, (H0, W0), target_size, rs, re_, args.min_gamma, floors, args.channels)
    return 'trained'


def main():
    ap = argparse.ArgumentParser(description='CPB training — Numba/vectorized (Pearson)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--output_dir', default='models/cpb_paper')
    ap.add_argument('--category', default=None)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--channels', choices=['gray', 'rgb'], default='gray',
                    help="gray=luma (what the framework uses; validated); rgb=3 channels")
    ap.add_argument('--n_train', type=int, default=N_TRAIN)
    ap.add_argument('--poolnum', type=int, default=POOLNUM)
    ap.add_argument('--min_gamma', type=float, default=MIN_GAMMA_DEFAULT)
    ap.add_argument('--max_train_pixels', type=int, default=MAX_TRAINING_PIXELS,
                    help='downsample scenes above this (NO scene is skipped). Huge = native res.')
    ap.add_argument('--no_calib_floor', action='store_true')
    ap.add_argument('--force', action='store_true', help='retrain scenes that already have a model')
    ap.add_argument('--list_scenes', action='store_true')
    args = ap.parse_args()

    if args.list_scenes:
        for c, s in ALL_CDNET_SCENES.items():
            print(f"  {c}: {s}")
        return

    root = args.dataset_root or find_root()
    if not root:
        sys.exit('ERROR: CDNet2014 not found — use --dataset_root')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}\n  CPB TRAINING (Numba, Pearson, {args.channels})  engine="
          f"{'numba' if HAVE_NUMBA else 'PYTHON-FALLBACK(slow)'}")
    print(f"  root={root}  out={args.output_dir}  no-skip (downsample>{args.max_train_pixels:,}px)\n{'='*70}")

    cats = [args.category] if args.category else list(ALL_CDNET_SCENES.keys())
    tally = {}
    for cat in cats:
        if cat not in ALL_CDNET_SCENES:
            print(f"  [WARN] unknown category {cat}"); continue
        for sc in ALL_CDNET_SCENES[cat]:
            if args.scene and sc != args.scene:
                continue
            try:
                st = train_scene(root, cat, sc, args.output_dir, args)
            except Exception as e:
                print(f"    [ERROR] {cat}/{sc}: {e}"); st = 'error'
            tally[st] = tally.get(st, 0) + 1
    print("\n" + "=" * 70)
    print("  DONE  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("=" * 70)


if __name__ == '__main__':
    main()