#!/usr/bin/env python3
"""
================================================================================
CPB + HoD + MODEL-COMMUNICATION FRAMEWORK  (faithful Python port of the authors'
MATLAB reference)                                                          v2
================================================================================
This file is a LITERAL port of the authors' MATLAB (subncc1.m / BlockReplacement.m
/ Getrsigementation.m / trainncc2.m), i.e. the Liang et al. PR-117 (2021) framework
whose base model is Zhou et al.'s CPB+HoD (ICPR 2018 / Signal Processing 2019).
It replaces the earlier "equations-only" reproduction, which was missing the HoD
degradation-modification and used the wrong support-exclusion threshold.

WHAT WAS MISSING BEFORE (and is fixed here) — verified against the .m files
---------------------------------------------------------------------------
(A) HoD "spare-pair / structure modification" = the CANDIDATE POOL WALK.
    MATLAB loops k=1..POOLNUM (30-35) and stops once it has collected REF=20
    supports whose block is currently BACKGROUND. When a support block is
    degraded (occluded / guidance-foreground) it is SKIPPED and the next spare
    candidate in the pool takes its place. The old Python used a fixed top-20 and
    simply dropped excluded supports (dropping to <20), which destabilised the
    decision. This pool-walk IS Zhou's "exchange the broken pair by a new one
    kept as a spare pair" (ICPR 2018, Sec. III-C).

(B) SUPPORT-EXCLUSION THRESHOLD.  MATLAB excludes a support block when it has
    >= 7 guidance-foreground pixels out of 8x8=64  (ratio ~0.109), i.e.
    `sum(guidance_block) < 7` means "background support".  The old Python used
    bth_ratio=0.5 (block foreground only if >=50%). 0.109 is far more aggressive,
    which is what makes the framework hand hard regions over to the guidance
    instead of letting CPB flood them.  --> here BTH is an ABSOLUTE PIXEL COUNT
    (default 7), exactly as MATLAB.

(C) ABSTAIN -> GUIDANCE.  In MATLAB, when a pixel has NO valid background support
    (blocknum==0) the output is set to the GUIDANCE value, not to background.
    The old Python abstained to background (used0=bg). Deferring to the guidance
    is what recovers objects the CPB cannot see.

(D) HoD decision rule (Zhou Eq.6-11) ported literally in Stage 2 (NaB / NaE
    broken-pair re-decision) and Stage 3 (foreground-similarity recovery).

(E) *** ALL-64 BLOCK RULE (v2 fix) ***  The CPB steady/unsteady test in the
    reference MATLAB (subncc1.m L45; BlockReplacement.m L30/57/97/124) compares
    the target pixel against the WHOLE 8x8 supporting block, and MATLAB's `if`
    on a matrix is true only when EVERY one of the 64 elements satisfies it. So a
    support votes "unsteady" only when the entire block deviates, and "steady"
    only when the entire block is within threshold. The earlier port collapsed
    the block to its MEAN (a single scalar), which over-fires (e.g. LIMU
    LightSwitch 0.42 -> 0.79 once corrected). This version uses the full-block
    (all-64) test in k_stage1 and k_stage2. Stage-3 already matched the MATLAB
    (block median for Spb, per Getrsigementation.m L24) and is unchanged.
    NOTE vs Liang Eq.4: the paper writes the state via the block AVERAGE Ī; the
    reference implementation uses this full-block consistency test. We adopt the
    implementation to reproduce the reported numbers.

Base CPB decision (subncc1): FG iff  (sum gamma over UNSTEADY supports) >=
0.5 * (sum gamma over all used supports).  Unsteady <=> the whole 8x8 support
block satisfies |(Ip - I_q) - b| >= af*sigma.  This equals the paper's Eq.5 with
lambda=0.5.

PARAMS (Table 1 / MATLAB): af(eta)=2.5, sigma(eps)=0.8, REF=20, POOL=model,
BTH=7 px, lambda=0.5, floor_min=6.0 (self-calibrating sigma floor).
Metric: CDnet (exclude 85/170; shadow 50 = BG).  Oracle + guidance-F1 kept.

USAGE
  # 30-frame dynamicBackground with HoD + oracle:
  python opus_liang_hod_framework.py --mode stage3 --guidance deeplab \
      --category dynamicBackground --n_frames 30 --debug --output r_dbg_hod.json

  # full dataset:
  python opus_liang_hod_framework.py --mode stage3 --guidance deeplab \
      --n_frames 0 --debug --output r_all_hod.json

  # pure CPB+HoD, no guidance (Zhou base model):  --mode stage1 --guidance zero
  # ablation: turn HoD pool-walk OFF (old behaviour):  --no_hod_pool
================================================================================
"""

import argparse
import json
import os
import pickle
import re
import time
import warnings

import cv2
import numpy as np

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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ---- paper / MATLAB params ----
BLOCK_SIZE = 8
REF_SUPPORT = 20          # refnum : active supports per pixel
ETA = 2.5                 # af : Gaussian threshold
LAMBDA_DEC = 0.5
EPSILON_SIM = 0.8         # sigema : stage-3 similarity threshold
DEFAULT_FLOOR_MIN = 6.0
BTH_PIXELS_DEFAULT = 7    # support-block is background if <7 guidance-fg px (of 64)
S3_MAX_RINGS_FG = 20

GT_FG = 255
GT_EXCLUDE = (85, 170)    # 50 (shadow) counts as BACKGROUND

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


# ============================================================================
# METRICS
# ============================================================================
def frame_counts(pred, gt):
    valid = np.ones_like(gt, dtype=bool)
    for v in GT_EXCLUDE:
        valid &= (gt != v)
    fg_gt = (gt == GT_FG); fg_pr = (pred == 255)
    tp = int(np.sum(fg_pr & fg_gt & valid)); fp = int(np.sum(fg_pr & ~fg_gt & valid))
    fn = int(np.sum(~fg_pr & fg_gt & valid)); tn = int(np.sum(~fg_pr & ~fg_gt & valid))
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    e = 1e-10
    pr = tp / (tp + fp + e); rc = tp / (tp + fn + e)
    f1 = 2 * pr * rc / (pr + rc + e); sp = tn / (tn + fp + e)
    fpr = fp / (fp + tn + e); fnr = fn / (tp + fn + e)
    pwc = 100.0 * (fp + fn) / (tp + fp + fn + tn + e)
    iou = tp / (tp + fp + fn + e)
    return {'f1': f1, 'precision': pr, 'recall': rc, 'specificity': sp, 'fpr': fpr,
            'fnr': fnr, 'pwc': pwc, 'iou': iou,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def agg_counts(counts):
    return metrics_from_counts(sum(c[0] for c in counts), sum(c[1] for c in counts),
                               sum(c[2] for c in counts), sum(c[3] for c in counts))


# ============================================================================
# BLOCK HELPERS
# ============================================================================
def block_mean(gray, ws, hs, bs=BLOCK_SIZE):
    H, W = gray.shape
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = gray
    return pad.reshape(ws, bs, hs, bs).mean(axis=(1, 3)).astype(np.float32)


def block_median(gray, ws, hs, bs=BLOCK_SIZE):
    H, W = gray.shape
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = gray
    return np.median(pad.reshape(ws, bs, hs, bs), axis=(1, 3)).astype(np.float32)


def block_fg_count(guid_fg, ws, hs, bs=BLOCK_SIZE):
    H, W = guid_fg.shape
    pad = np.zeros((ws * bs, hs * bs), np.float32); pad[:H, :W] = guid_fg.astype(np.float32)
    return pad.reshape(ws, bs, hs, bs).sum(axis=(1, 3)).astype(np.float32)


# ============================================================================
# NUMBA KERNELS  (literal port; one channel)  --  ALL-64 block rule (v2)
# ============================================================================
@njit(cache=False, inline='always')
def _block_flags(gray, r0, c0, p, b, thr, H, W):
    """Full-block (all-64) test for one 8x8 support block at (r0,c0).
    Returns (steady, unsteady):
        steady   = 1 iff EVERY pixel q of the block has |(p - q) - b| <= thr
        unsteady = 1 iff EVERY pixel q of the block has |(p - q) - b| >= thr
    (MATLAB `if abs(II(block) - II(p) - b) <=/>= thr` is true only for all()).
    Out-of-range block pixels (padding at the image border) are ignored."""
    nval = 0; nle = 0; nge = 0
    for u in range(8):
        rr = r0 + u
        if rr >= H:
            continue
        for v in range(8):
            cc = c0 + v
            if cc >= W:
                continue
            nval += 1
            d = abs((p - gray[rr, cc]) - b)
            if d <= thr:
                nle += 1
            if d >= thr:
                nge += 1
    steady = 1 if (nval > 0 and nle == nval) else 0
    unsteady = 1 if (nval > 0 and nge == nval) else 0
    return steady, unsteady


@njit(cache=False, parallel=True, fastmath=False)
def k_stage1(gray, DBS, gcnt, guid_fg, D, out, score_out, supp_out,
             af, floor, lam, BTH, REF, pool, hod_pool):
    """subncc1 (all-64): base CPB decision with HoD pool-walk + abstain->guidance.
    A support votes UNSTEADY only when EVERY pixel of its 8x8 block deviates.
    Also writes DRL features: score_out = changed/total (the continuous CPB
    foregroundness in [0,1]) and supp_out = # live supports."""
    H, W = gray.shape
    ws, hs = DBS.shape
    for x in prange(H):
        for y in range(W):
            total = 0.0; changed = 0.0; blocknum = 0
            kmax = pool if hod_pool else min(REF, pool)
            for k in range(kmax):
                if blocknum >= REF:
                    break
                gam = D[x, y, k, 4]
                if gam <= 0.0:
                    continue
                bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                if bi < 0:
                    bi = 0
                elif bi >= ws:
                    bi = ws - 1
                if bj < 0:
                    bj = 0
                elif bj >= hs:
                    bj = hs - 1
                if gcnt[bi, bj] < BTH:                       # background support
                    total += gam
                    sig = D[x, y, k, 3]
                    if sig < floor:
                        sig = floor
                    st, un = _block_flags(gray, bi * 8, bj * 8, gray[x, y],
                                          D[x, y, k, 2], af * sig, H, W)
                    if un:                                   # UNSTEADY (all 64 deviate)
                        changed += gam
                    blocknum += 1
            supp_out[x, y] = np.float32(blocknum)
            if blocknum == 0:
                out[x, y] = 255 if guid_fg[x, y] > 0 else 0  # abstain -> guidance
                score_out[x, y] = np.float32(1.0 if guid_fg[x, y] > 0 else 0.0)
            else:
                frac = changed / total if total > 1e-9 else 0.0
                score_out[x, y] = np.float32(frac)
                out[x, y] = 255 if changed >= total * lam else 0


# ============================================================================
# PATCH for opus_final_hod_framework.py
# ----------------------------------------------------------------------------
# Aligns the DRL env's Stage-2 with liang_v7 --decision all64 --stage2_mode paper
# (the locked RGB 0.542 baseline).  APPLY:
#   1) In opus_final_hod_framework.py rename the existing  def k_stage2(...)
#      to  def k_stage2_port(...)   (keeps the old two-pass for ablation).
#   2) Paste the function below as the new  k_stage2  (SAME 11-arg signature,
#      so the env/trainer/metrics need NO changes).
#   3) k_stage1 and k_stage3 stay untouched.
# ============================================================================
from numba import njit, prange   # (already imported in the target file)
# LAMBDA_DEC and _block_flags already exist in opus_final_hod_framework.py

@njit(cache=False, parallel=True, fastmath=False)
def k_stage2(gray, DBS, gcnt, guid_fg, D, cpb_in, out, af, floor, BTH, REF, pool):
    """Stage 2 - FAITHFUL Liang Eq.6-8 (paper), all-64 test.
    Reproduces liang_v7 --decision all64 --stage2_mode paper (locked 0.542 RGB)."""
    H, W = gray.shape
    ws, hs = DBS.shape
    lam = LAMBDA_DEC
    for x in prange(H):
        for y in range(W):
            out[x, y] = cpb_in[x, y]
            cpb_fg = cpb_in[x, y] > 0
            g_fg = guid_fg[x, y] > 0
            if cpb_fg == g_fg:
                continue
            Ks = 0; sg_s = 0.0; Ku = 0; sg_u = 0.0; nb = 0
            for k in range(pool):
                if nb >= REF:
                    break
                gam = D[x, y, k, 4]
                if gam <= 0.0:
                    continue
                bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                if bi < 0: bi = 0
                elif bi >= ws: bi = ws - 1
                if bj < 0: bj = 0
                elif bj >= hs: bj = hs - 1
                if gcnt[bi, bj] < BTH:
                    nb += 1
                    sig = D[x, y, k, 3]
                    if sig < floor: sig = floor
                    st, un = _block_flags(gray, bi * 8, bj * 8, gray[x, y],
                                          D[x, y, k, 2], af * sig, H, W)
                    if st: Ks += 1; sg_s += gam
                    if un: Ku += 1; sg_u += gam
            if nb == 0:
                out[x, y] = 255 if g_fg else 0
                continue
            if cpb_fg and (not g_fg):
                gbar = (sg_s / Ks) if Ks > 0 else 1e18
            else:
                gbar = (sg_u / Ku) if Ku > 0 else 1e18
            total = 0.0; unstw = 0.0; nb2 = 0
            for k in range(pool):
                if nb2 >= REF:
                    break
                gam = D[x, y, k, 4]
                if gam <= 0.0:
                    continue
                bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                if bi < 0: bi = 0
                elif bi >= ws: bi = ws - 1
                if bj < 0: bj = 0
                elif bj >= hs: bj = hs - 1
                if gcnt[bi, bj] < BTH and gam < gbar:
                    nb2 += 1; total += gam
                    sig = D[x, y, k, 3]
                    if sig < floor: sig = floor
                    st, un = _block_flags(gray, bi * 8, bj * 8, gray[x, y],
                                          D[x, y, k, 2], af * sig, H, W)
                    if un: unstw += gam
            if nb2 == 0:
                out[x, y] = 255 if g_fg else 0
            else:
                out[x, y] = 255 if unstw >= lam * total else 0
@njit(cache=False, fastmath=False)
def _min_fg_sim(gray, hod, x, y, H, W):
    """GetMinFGSim: mean |I(fg neighbour)-I(p)| over up-to-20 nearest FG px (square rings)."""
    S = 0.0; fgn = 0
    lim = min(min(x, y), min(H - 1 - x, W - 1 - y))
    for k in range(1, lim):
        if fgn > S3_MAX_RINGS_FG:
            break
        i0 = x - k; i1 = x + k
        for i in range(i0, i1 + 1):
            if i == i0 or i == i1:
                for j in range(y - k, y + k + 1):
                    if hod[i, j] > 0:
                        S += abs(gray[i, j] - gray[x, y]); fgn += 1
            else:
                if hod[i, y - k] > 0:
                    S += abs(gray[i, y - k] - gray[x, y]); fgn += 1
                if hod[i, y + k] > 0:
                    S += abs(gray[i, y + k] - gray[x, y]); fgn += 1
    if fgn == 0:
        return 1e6
    return S / fgn


@njit(cache=False, parallel=True, fastmath=False)
def k_stage3(gray, DBM, gcnt, guid_fg, D, hod_in, out, sigema, BTH, REF, pool):
    """Getrsigementation: stage-3 foreground-similarity recovery for NaB pixels.
    Unchanged from v1: Spb uses the block MEDIAN (DBM), matching
    Getrsigementation.m L24 (median(median(IItest(block))))."""
    H, W = gray.shape
    ws, hs = DBM.shape
    for x in prange(H):
        for y in range(W):
            out[x, y] = hod_in[x, y]
            if hod_in[x, y] == 0 and guid_fg[x, y] > 0:
                Spq = _min_fg_sim(gray, hod_in, x, y, H, W)
                total = 0.0; blocknum = 0
                for k in range(pool):
                    if blocknum >= REF:
                        break
                    gam = D[x, y, k, 4]
                    if gam <= 0.0:
                        continue
                    bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                    if bi < 0: bi = 0
                    elif bi >= ws: bi = ws - 1
                    if bj < 0: bj = 0
                    elif bj >= hs: bj = hs - 1
                    if gcnt[bi, bj] < BTH:
                        total += abs(DBM[bi, bj] - gray[x, y]); blocknum += 1
                if blocknum != 0:
                    Spb = total / blocknum
                    out[x, y] = 255 if Spq < sigema * Spb else 0
                else:
                    out[x, y] = 255 if guid_fg[x, y] > 0 else 0


# ============================================================================
# GUIDANCE
# ============================================================================
class ZeroGuidance:
    name = 'zero'
    def predict(self, frame_bgr):
        return np.zeros(frame_bgr.shape[:2], np.uint8)


def make_guidance(kind, smooth, models_dir='./models', allow_fallback=False, device='cuda:0'):
    if kind == 'zero':
        return ZeroGuidance()
    if str(device).lower().startswith('cuda'):
        try:
            import torch
            if not torch.cuda.is_available():
                print("  [Guidance] CUDA requested but torch is CPU-only -> device='cpu'")
                device = 'cpu'
        except Exception:
            device = 'cpu'
    from opus_guidance_config import ADE20KGuidance
    backbone = 'deeplabv3plus' if kind in ('deeplab', 'deeplabv3plus') else 'pspnet'
    return ADE20KGuidance(backbone=backbone, device=device, models_dir=models_dir,
                          smooth=smooth, allow_fallback=allow_fallback)


# ============================================================================
# VISUALIZATION
# ============================================================================
def save_panel(out_path, rgb, gt, pred):
    if not _HAVE_MPL:
        return
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    gt_disp = np.zeros_like(rgb_u8); gt_disp[gt == GT_FG] = (255, 255, 255)
    gt_disp[gt == 50] = (120, 120, 60)
    for v in GT_EXCLUDE:
        gt_disp[gt == v] = (80, 80, 80)
    pred_disp = np.zeros_like(rgb_u8); pred_disp[pred == 255] = (255, 255, 255)
    valid = np.ones(gt.shape, bool)
    for v in GT_EXCLUDE:
        valid &= (gt != v)
    fg_gt = (gt == GT_FG); fg_pr = (pred == 255)
    err = np.zeros((*gt.shape, 3), np.uint8)
    err[fg_pr & fg_gt & valid] = (0, 200, 0)
    err[fg_pr & ~fg_gt & valid] = (220, 0, 0)
    err[~fg_pr & fg_gt & valid] = (0, 80, 255)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, t in zip(ax, [rgb_u8, gt_disp, pred_disp, err],
                        ['input', 'ground truth', 'prediction', 'error (G=TP R=FP B=FN)']):
        a.imshow(im); a.set_title(t, fontsize=11); a.axis('off')
    fig.tight_layout(); fig.savefig(out_path, dpi=90, bbox_inches='tight'); plt.close(fig)


def save_summary_charts(out_dir, all_results):
    if not _HAVE_MPL or not all_results:
        return
    keys = list(all_results.keys())
    f1s = [all_results[k]['f1'] for k in keys]
    gs = [all_results[k].get('guidance_f1', 0.0) for k in keys]
    os_ = [all_results[k].get('oracle_f1', 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.6), 6))
    order = np.argsort(f1s); yy = np.arange(len(keys))
    ax.barh(yy - 0.25, [f1s[i] for i in order], height=0.25, color='#3a7bd5', label='HoD framework')
    ax.barh(yy, [gs[i] for i in order], height=0.25, color='#f5a623', label='guidance')
    ax.barh(yy + 0.25, [os_[i] for i in order], height=0.25, color='#4caf50', label='oracle')
    ax.set_yticks(yy); ax.set_yticklabels([keys[i] for i in order], fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel('F-Measure'); ax.set_title('HoD framework vs guidance vs oracle')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '_summary_f1_bar.png'), dpi=110); plt.close(fig)


# ============================================================================
# PIPELINE
# ============================================================================
def to_luma(rgb):
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)


class HoDPipeline:
    def __init__(self, model, mode, floor_min, af, sigema, bth, ref, hod_pool, channel_mode):
        self.mode = mode
        self.af = float(af); self.sigema = float(sigema)
        self.floor = float(floor_min); self.bth = float(bth)
        self.ref = int(ref); self.hod_pool = bool(hod_pool)
        self.Hf, self.Wf = model['img_size']
        self.ws = int(np.ceil(self.Hf / BLOCK_SIZE)); self.hs = int(np.ceil(self.Wf / BLOCK_SIZE))
        Dr = np.ascontiguousarray(model['DBLK_r'], np.float32)
        Dg = np.ascontiguousarray(model['DBLK_g'], np.float32)
        Db = np.ascontiguousarray(model['DBLK_b'], np.float32)
        self.pool = Dr.shape[2]
        gray_model = np.allclose(Dr, Dg) and np.allclose(Dr, Db)
        if channel_mode == 'gray' or (channel_mode == 'auto' and gray_model):
            self.channels = [('L', Dr)]                 # single luma channel
        else:
            self.channels = [('R', Dr), ('G', Dg), ('B', Db)]
        self.last_s1 = None            # native CPB stage-1 mask (0/255)
        self.last_cpb_score = None     # native float [0,1] : CPB foregroundness (changed/total)
        self.last_cpb_supp = None      # native float : # live supports (confidence)

    def _fields(self, gray, guid_fg):
        DBS = block_mean(gray, self.ws, self.hs)
        DBM = block_median(gray, self.ws, self.hs)
        gcnt = block_fg_count(guid_fg, self.ws, self.hs)
        return DBS, DBM, gcnt

    def _run_channel(self, gray, guid_fg, D, DBS, DBM, gcnt):
        H, W = gray.shape
        s1 = np.zeros((H, W), np.uint8)
        score = np.zeros((H, W), np.float32)
        supp = np.zeros((H, W), np.float32)
        k_stage1(gray, DBS, gcnt, guid_fg, D, s1, score, supp, self.af, self.floor,
                 LAMBDA_DEC, self.bth, self.ref, self.pool, self.hod_pool)
        if self.mode == 'stage1':
            return s1, s1, score, supp
        s2 = np.zeros((H, W), np.uint8)
        k_stage2(gray, DBS, gcnt, guid_fg, D, s1, s2, self.af, self.floor,
                 self.bth, self.ref, self.pool)
        if self.mode == 'stage2':
            return s1, s2, score, supp
        s3 = np.zeros((H, W), np.uint8)
        k_stage3(gray, DBM, gcnt, guid_fg, D, s2, s3, self.sigema, self.bth, self.ref, self.pool)
        return s1, s3, score, supp

    def run(self, frame_native_rgb, guidance_native):
        native = frame_native_rgb.shape[:2]
        if native != (self.Hf, self.Wf):
            fr = cv2.resize(frame_native_rgb, (self.Wf, self.Hf)).astype(np.float32)
            g = cv2.resize(guidance_native, (self.Wf, self.Hf), interpolation=cv2.INTER_NEAREST)
        else:
            fr = frame_native_rgb.astype(np.float32); g = guidance_native
        guid_fg = (g > 127).astype(np.uint8)

        s1_acc = None; out_acc = None; score_acc = None; supp_acc = None
        for nm, D in self.channels:
            gray = to_luma(fr) if nm == 'L' else fr[:, :, {'R': 0, 'G': 1, 'B': 2}[nm]].astype(np.float32)
            DBS, DBM, gcnt = self._fields(gray, guid_fg)
            s1, out, score, supp = self._run_channel(gray, guid_fg, D, DBS, DBM, gcnt)
            s1_acc = s1 if s1_acc is None else np.maximum(s1_acc, s1)      # OR across channels
            out_acc = out if out_acc is None else np.maximum(out_acc, out)
            score_acc = score if score_acc is None else np.maximum(score_acc, score)
            supp_acc = supp if supp_acc is None else np.maximum(supp_acc, supp)
        self.last_s1 = self._native(s1_acc, native)
        self.last_cpb_score = self._native_f(score_acc, native)
        self.last_cpb_supp = self._native_f(supp_acc, native)
        return self._native(out_acc, native)

    @staticmethod
    def _native_f(arr, native):
        if arr.shape != native:
            arr = cv2.resize(arr, (native[1], native[0]), interpolation=cv2.INTER_NEAREST)
        return arr

    @staticmethod
    def _native(mask, native):
        m = mask.astype(np.uint8)
        if m.shape != native:
            m = cv2.resize(m, (native[1], native[0]), interpolation=cv2.INTER_NEAREST)
        return m


# ============================================================================
# DATA
# ============================================================================
def find_root():
    for p in ['data/cdnet2014', '../data/cdnet2014', '.', '/media/data/cdnet2014']:
        if os.path.exists(os.path.join(p, 'baseline')) or os.path.basename(os.path.abspath(p)) == 'cdnet2014':
            return os.path.abspath(p)
    return None


def all_nums(scene_path):
    d = os.path.join(scene_path, 'input'); ns = []
    for fn in os.listdir(d):
        if fn.lower().endswith(('.jpg', '.png')):
            m = re.findall(r'\d+', fn)
            if m:
                ns.append(int(m[-1]))
    return sorted(ns)


def load_bgr(input_dir, n):
    for pat in (f'in{n:06d}.jpg', f'in{n:05d}.jpg', f'{n:06d}.jpg', f'in{n:06d}.png'):
        p = os.path.join(input_dir, pat)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return im
    return None


def load_gt(gt_dir, n):
    for pat in (f'gt{n:06d}.png', f'gt{n:05d}.png'):
        p = os.path.join(gt_dir, pat)
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None


# ============================================================================
# PER-SCENE RUN
# ============================================================================
def run_scene(models_dir, dataset_root, category, scene, guidance, args, viz_root=None):
    model_path = os.path.join(models_dir, f'cpb_paper_{category}_{scene}.pkl')
    if not os.path.exists(model_path):
        print(f'  [skip] model not found: {model_path}'); return None
    model = pickle.load(open(model_path, 'rb'))
    p = model['params']
    roi_s, roi_e = int(p['roi_start']), int(p['roi_end'])
    sp = os.path.join(dataset_root, category, scene)
    idir, gdir = os.path.join(sp, 'input'), os.path.join(sp, 'groundtruth')
    nums = [n for n in all_nums(sp) if roi_s <= n <= roi_e
            and os.path.exists(os.path.join(gdir, f'gt{n:06d}.png'))]
    if args.n_frames and args.n_frames > 0 and len(nums) > args.n_frames:
        nums = nums[::max(1, len(nums) // args.n_frames)][:args.n_frames]
    if not nums:
        print('  [skip] no ROI frames'); return None

    pipe = HoDPipeline(model, args.mode, args.floor_min, args.af, args.sigema,
                       args.bth_pixels, args.ref, not args.no_hod_pool, args.channel_mode)
    print(f"\n  > {category}/{scene}  mode={args.mode}  guid={guidance.name}  "
          f"ch={[c[0] for c in pipe.channels]}  pool={pipe.pool}  hod_pool={not args.no_hod_pool}  "
          f"BTH={args.bth_pixels}px  af={args.af}  floor={args.floor_min}  fit={pipe.Hf}x{pipe.Wf}  frames={len(nums)}")

    viz_dir = None; n_viz = 0
    if args.viz and viz_root:
        viz_dir = os.path.join(viz_root, f"{category}_{scene}"); os.makedirs(viz_dir, exist_ok=True)
    counts = []; gcounts = []; ocounts = []; t0 = time.time()
    for idx, n in enumerate(nums):
        bgr = load_bgr(idir, n); gt = load_gt(gdir, n)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gmask = guidance.predict(bgr)
        pred = pipe.run(rgb, gmask)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        counts.append(frame_counts(pred, gt))
        gp = cv2.resize(gmask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        guid_fg = (gp > 127)
        gcounts.append(frame_counts((guid_fg.astype(np.uint8) * 255), gt))
        s1m = pipe.last_s1
        if s1m is not None and s1m.shape != gt.shape:
            s1m = cv2.resize(s1m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        s1_fg = (s1m == 255) if s1m is not None else np.zeros_like(guid_fg)
        agree = (s1_fg == guid_fg)
        oracle_pred = (np.where(agree, s1_fg, (gt == GT_FG)).astype(np.uint8) * 255)
        ocounts.append(frame_counts(oracle_pred, gt))
        if args.viz and viz_dir and n_viz < args.viz_max and (idx % max(1, args.viz_every) == 0):
            save_panel(os.path.join(viz_dir, f'panel_{n:06d}.png'), rgb, gt, pred); n_viz += 1
        if (idx + 1) % 200 == 0:
            a = agg_counts(counts)
            print(f"    {idx+1:5d}/{len(nums)}  F1={a['f1']:.4f}  ({(idx+1)/max(time.time()-t0,1e-3):.2f} fr/s)")
    if not counts:
        return None
    r = agg_counts(counts); gq = agg_counts(gcounts); oq = agg_counts(ocounts)
    r.update({'category': category, 'scene': scene, 'n_frames': len(counts),
              'guidance_f1': gq['f1'], 'oracle_f1': oq['f1'],
              'sec': round(time.time() - t0, 1), 'fps': round(len(counts) / max(time.time() - t0, 1e-3), 2)})
    print(f"    FINAL  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"FPR={r['fpr']:.4f}  IoU={r['iou']:.4f}  ({r['fps']} fr/s)")
    print(f"    GUIDANCE-only F1={gq['f1']:.4f}   ORACLE F1={oq['f1']:.4f}  <- DRL headroom")
    return r


# ============================================================================
# REPORTING
# ============================================================================
def print_tables(all_results):
    print(f"\n{'='*104}")
    print(f"  {'Scene':<28} {'F1':<7} {'guidF1':<7} {'oracF1':<7} {'P':<7} {'R':<7} {'FPR':<7} {'IoU':<7}")
    print('  ' + '-' * 100)
    for k, r in all_results.items():
        print(f"  {k:<28} {r['f1']:<7.4f} {r.get('guidance_f1',0):<7.4f} {r.get('oracle_f1',0):<7.4f} "
              f"{r['precision']:<7.4f} {r['recall']:<7.4f} {r['fpr']:<7.4f} {r['iou']:<7.4f}")
    cats = {}
    for r in all_results.values():
        cats.setdefault(r['category'], []).append(r)
    print(f"\n  {'Category (avg)':<28} {'F1':<7} {'guidF1':<7} {'oracF1':<7}")
    print('  ' + '-' * 56)
    cf = []; gf = []; of = []
    for c, rs in sorted(cats.items()):
        f1 = np.mean([x['f1'] for x in rs]); g1 = np.mean([x.get('guidance_f1', 0) for x in rs])
        o1 = np.mean([x.get('oracle_f1', 0) for x in rs]); cf.append(f1); gf.append(g1); of.append(o1)
        print(f"  {c:<28} {f1:<7.4f} {g1:<7.4f} {o1:<7.4f}")
    print('  ' + '-' * 56)
    if cf:
        print(f"  {'OVERALL (mean of cats)':<28} {np.mean(cf):<7.4f} {np.mean(gf):<7.4f} {np.mean(of):<7.4f}")
    return cats


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description='CPB+HoD framework (faithful MATLAB port, all-64)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--models_dir', default='models/cpb_paper')
    ap.add_argument('--guidance_models_dir', default='./models')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--mode', default='stage3', choices=['stage1', 'stage2', 'stage3'])
    ap.add_argument('--guidance', default='deeplab', choices=['zero', 'deeplab', 'pspnet'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--allow_fallback', action='store_true')
    ap.add_argument('--category', default=None)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--channel_mode', default='auto', choices=['auto', 'gray', 'rgb'])
    ap.add_argument('--af', type=float, default=ETA, help='eta: Gaussian threshold (2.5)')
    ap.add_argument('--sigema', type=float, default=EPSILON_SIM, help='stage-3 similarity threshold (0.8)')
    ap.add_argument('--bth_pixels', type=float, default=BTH_PIXELS_DEFAULT,
                    help='support-block is background if < this many guidance-fg px (MATLAB=7)')
    ap.add_argument('--ref', type=int, default=REF_SUPPORT, help='active supports per pixel (20)')
    ap.add_argument('--floor_min', type=float, default=DEFAULT_FLOOR_MIN)
    ap.add_argument('--no_hod_pool', action='store_true',
                    help='ABLATION: disable HoD spare-pair pool-walk (use only top-REF supports)')
    ap.add_argument('--n_frames', type=int, default=0, help='0 = all ROI frames')
    ap.add_argument('--smooth_guidance', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--viz', action='store_true')
    ap.add_argument('--viz_dir', default='hod_framework_viz')
    ap.add_argument('--viz_every', type=int, default=25)
    ap.add_argument('--viz_max', type=int, default=8)
    ap.add_argument('--output', default='hod_framework_results.json')
    args = ap.parse_args()

    root = args.dataset_root or find_root()
    if not root:
        raise SystemExit('ERROR: CDnet2014 not found — use --dataset_root')
    guidance = make_guidance(args.guidance, args.smooth_guidance,
                             models_dir=args.guidance_models_dir, allow_fallback=args.allow_fallback,
                             device=args.device)
    viz_root = os.path.abspath(args.viz_dir) if (args.viz or args.debug) else None
    if viz_root:
        os.makedirs(viz_root, exist_ok=True)
    cats = {args.category: ([args.scene] if args.scene else CDNET2014.get(args.category, []))} if args.category else CDNET2014

    print(f"\n{'='*74}\n  CPB + HoD FRAMEWORK (faithful MATLAB port, ALL-64)  mode={args.mode}  guidance={guidance.name}")
    print(f"  HoD pool-walk={'OFF(ablation)' if args.no_hod_pool else 'ON'}  BTH={args.bth_pixels}px  "
          f"af={args.af}  channels={args.channel_mode}  engine={'numba' if HAVE_NUMBA else 'python-fallback'}")
    print(f"  floor_min={args.floor_min}  frames/scene={'ALL' if args.n_frames == 0 else args.n_frames}\n  Dataset: {root}\n{'='*74}")

    all_results = {}
    for cat, scenes in cats.items():
        for sc in scenes:
            r = run_scene(args.models_dir, root, cat, sc, guidance, args, viz_root)
            if r:
                all_results[f'{cat}/{sc}'] = r
    if not all_results:
        raise SystemExit('No scenes produced results.')
    catmap = print_tables(all_results)
    if args.viz and _HAVE_MPL:
        save_summary_charts(viz_root, all_results); print(f"  Summary charts -> {viz_root}")
    payload = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'mode': args.mode, 'guidance': guidance.name,
               'hod_pool': not args.no_hod_pool, 'bth_pixels': args.bth_pixels, 'af': args.af,
               'channel_mode': args.channel_mode, 'floor_min': args.floor_min, 'n_frames': args.n_frames,
               'mean_scene_f1': float(np.mean([r['f1'] for r in all_results.values()])),
               'mean_scene_guidance_f1': float(np.mean([r.get('guidance_f1', 0) for r in all_results.values()])),
               'mean_scene_oracle_f1': float(np.mean([r.get('oracle_f1', 0) for r in all_results.values()])),
               'category_means': {c: float(np.mean([x['f1'] for x in rs])) for c, rs in catmap.items()},
               'results': all_results}
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()