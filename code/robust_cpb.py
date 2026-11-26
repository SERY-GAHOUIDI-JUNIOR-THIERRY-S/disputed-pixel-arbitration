#!/usr/bin/env python3
"""
CPB trainer with a CONSISTENT block statistic and optional ROBUST (foreground-rejecting) training.

Same algorithm as Zhou's MATLAB port (cpb_training.py):
  support score = MIN std over time of (I_p - Q_block_median), gamma = r/2 + 0.5, b = mean(I_p - Q),
  POOL best supports kept, sorted best first, own block excluded.
Two differences, both switchable:
  1. MAD sigma is computed from the SAME residual as b and the selection, (I_p - Q_median) - b.
     (The old robust_sigma_pass used the block MEAN; b and selection used the block MEDIAN.)
  2. robust=True: per-pixel rejection of foreground observations in the training frames:
       m = temporal median of the pixel, s = 1.4826 * MAD;  observation t is an outlier if
       |I_p(t) - m| > k * max(s, s_min).  Safeguard: if a pixel would lose more than
       max_reject of its frames, the "typical value" is ambiguous (possibly the person), so that
       pixel keeps all its frames.  The per-frame block statistic is the median of the block's
       retained pixels; a block-frame is valid if >= half of its 64 pixels are retained.
       Every per-pair quantity (selection score, r, b, sigma, MAD) uses only frames where BOTH
       the pixel and the support block are valid; a pair needs >= min_frac*T valid frames.
With robust=False and mad=False this reproduces train_kernel_fixed exactly (checked in __main__).
"""
import numpy as np
from numba import njit, prange

BS = 8
NEG = -1e30


def masks_and_blocks(G, robust, k=3.0, s_min=5.0, max_reject=0.4, max_frame_reject=1.0):
    """G [T,H,W] -> (pixel validity [T,H,W] bool, block median [T,ws,hs], block validity [T,ws,hs])."""
    T, H, W = G.shape
    if robust:
        m = np.median(G, 0)
        s = 1.4826 * np.median(np.abs(G - m[None]), 0)
        Vp = np.abs(G - m[None]) <= k * np.maximum(s, s_min)[None]
        rej = 1.0 - Vp.mean(0)
        Vp[:, rej > max_reject] = True                      # ambiguous pixel -> keep everything
        # frame safeguard: foreground is LOCAL. A frame in which more than max_frame_reject of the pixels
        # deviate is a GLOBAL change (illumination / camera), which the support model must learn -> keep it.
        frej = 1.0 - Vp.mean((1, 2))
        Vp[frej > max_frame_reject] = True
    else:
        Vp = np.ones(G.shape, bool)
    ws, hs = int(np.ceil(H / BS)), int(np.ceil(W / BS))
    padG = np.full((T, ws * BS, hs * BS), np.nan, np.float32); padG[:, :H, :W] = np.where(Vp, G, np.nan)
    blk = padG.reshape(T, ws, BS, hs, BS)
    nvalid = np.sum(~np.isnan(blk), axis=(2, 4))
    with np.errstate(all='ignore'):
        Q = np.nanmedian(blk.transpose(0, 1, 3, 2, 4).reshape(T, ws, hs, BS * BS), axis=3)
    npix = np.minimum(H - np.arange(ws)[:, None] * BS, BS) * np.minimum(W - np.arange(hs)[None, :] * BS, BS)
    Vb = nvalid >= np.ceil(npix[None] / 2.0)
    Q = np.where(Vb, Q, 0.0).astype(np.float32)
    return Vp, Q, Vb


@njit(cache=False, parallel=True, fastmath=False)
def train_masked(G, Vp, Q, Vb, pool, min_n, mad, sig_all, out):
    T, H, W = G.shape
    _, ws, hs = Q.shape
    for x in prange(H):
        sc_ = np.empty(pool); bi_ = np.empty(pool, np.int64); bj_ = np.empty(pool, np.int64)
        bb_ = np.empty(pool); bs_ = np.empty(pool); gm_ = np.empty(pool)
        res = np.empty(T)
        for y in range(W):
            for q in range(pool):
                sc_[q] = NEG; bi_[q] = -1; bj_[q] = -1; bb_[q] = 0.0; bs_[q] = 0.0; gm_[q] = 0.0
            sbx = x // BS; sby = y // BS
            filled = 0; worst = NEG; widx = 0
            for bi in range(ws):
                for bj in range(hs):
                    if bi == sbx and bj == sby:
                        continue
                    n = 0; sp = 0.0; sq = 0.0
                    for t in range(T):
                        if Vp[t, x, y] and Vb[t, bi, bj]:
                            n += 1; sp += G[t, x, y]; sq += Q[t, bi, bj]
                    if n < min_n:
                        continue
                    mp = sp / n; mq = sq / n
                    cov = 0.0; vp = 0.0; vq = 0.0; dsum = 0.0; d2 = 0.0
                    for t in range(T):
                        if Vp[t, x, y] and Vb[t, bi, bj]:
                            a = G[t, x, y] - mp; c = Q[t, bi, bj] - mq
                            cov += a * c; vp += a * a; vq += c * c
                            d = G[t, x, y] - Q[t, bi, bj]; dsum += d; d2 += d * d
                    den = (vp * vq) ** 0.5
                    r = cov / den if den > 1e-9 else 0.0
                    if r > 1.0: r = 1.0
                    elif r < -1.0: r = -1.0
                    b = dsum / n
                    var = d2 / n - b * b
                    if var < 0.0: var = 0.0
                    sig = var ** 0.5
                    s = -sig                                  # Zhou: minimum residual std
                    if filled < pool:
                        sc_[filled] = s; bi_[filled] = bi; bj_[filled] = bj; bb_[filled] = b; bs_[filled] = sig
                        gm_[filled] = r * 0.5 + 0.5; filled += 1
                        if filled == pool:
                            worst = sc_[0]; widx = 0
                            for q in range(1, pool):
                                if sc_[q] < worst:
                                    worst = sc_[q]; widx = q
                    elif s > worst:
                        sc_[widx] = s; bi_[widx] = bi; bj_[widx] = bj; bb_[widx] = b; bs_[widx] = sig
                        gm_[widx] = r * 0.5 + 0.5
                        worst = sc_[0]; widx = 0
                        for q in range(1, pool):
                            if sc_[q] < worst:
                                worst = sc_[q]; widx = q
            # sort best first (descending score)
            for a in range(1, pool):
                sv = sc_[a]; iv = bi_[a]; jv = bj_[a]; bv = bb_[a]; gv = bs_[a]; wv = gm_[a]; c = a - 1
                while c >= 0 and sc_[c] < sv:
                    sc_[c + 1] = sc_[c]; bi_[c + 1] = bi_[c]; bj_[c + 1] = bj_[c]
                    bb_[c + 1] = bb_[c]; bs_[c + 1] = bs_[c]; gm_[c + 1] = gm_[c]; c -= 1
                sc_[c + 1] = sv; bi_[c + 1] = iv; bj_[c + 1] = jv; bb_[c + 1] = bv; bs_[c + 1] = gv; gm_[c + 1] = wv
            for k in range(pool):
                if bi_[k] < 0:
                    out[x, y, k, 0] = 0; out[x, y, k, 1] = 0; out[x, y, k, 2] = 0.0; out[x, y, k, 3] = 0.0; out[x, y, k, 4] = 0.0
                    continue
                sig = bs_[k]
                if mad:                                       # MAD of the SAME residual, valid frames only
                    n = 0
                    for t in range(T):
                        if (sig_all or Vp[t, x, y]) and Vb[t, bi_[k], bj_[k]]:
                            res[n] = G[t, x, y] - Q[t, bi_[k], bj_[k]] - bb_[k]; n += 1
                    med = np.median(res[:n])
                    for t in range(n):
                        res[t] = abs(res[t] - med)
                    sig = 1.4826 * np.median(res[:n])
                out[x, y, k, 0] = bi_[k]; out[x, y, k, 1] = bj_[k]
                out[x, y, k, 2] = bb_[k]; out[x, y, k, 3] = sig; out[x, y, k, 4] = gm_[k]


def train_channel(G, robust=False, mad=True, pool=30, min_frac=0.3, k=3.0, s_min=5.0, max_reject=0.4, sig_all=False, max_frame_reject=1.0):
    """sig_all=True: robust selection/offset, but the MAD sigma uses ALL frames of the pixel (rejected ones too)."""
    """Returns a model dict compatible with wallflower_eval / the detection kernels, plus stats."""
    G = np.ascontiguousarray(G, np.float32)
    T, H, W = G.shape
    Vp, Q, Vb = masks_and_blocks(G, robust, k, s_min, max_reject, max_frame_reject)
    out = np.zeros((H, W, pool, 5), np.float32)
    train_masked(G, Vp, Q, Vb, pool, max(20, int(min_frac * T)), mad, sig_all, out)
    stats = dict(rejected=float(1.0 - Vp.mean()), blocks_invalid=float(1.0 - Vb.mean()))
    return {'DBLK_r': out, 'DBLK_g': out, 'DBLK_b': out, 'img_size': (H, W), 'params': {'poolnum': pool}}, stats


if __name__ == '__main__':
    # equivalence check: robust=False, mad=False must reproduce Zhou's train_kernel_fixed
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cpb_training as TR
    rng = np.random.RandomState(0)
    G = (rng.rand(40, 32, 48) * 20 + np.linspace(50, 150, 48)[None, None, :]).astype(np.float32)
    ref = TR.train_channel_fixed(G, 30, 0.0, 0.0, 'chk', 'median', 'minstd', 'matlab', 'std')
    new, _ = train_channel(G, robust=False, mad=False, min_frac=0.0)
    same_idx = np.mean((ref[..., 0] == new['DBLK_r'][..., 0]) & (ref[..., 1] == new['DBLK_r'][..., 1]))
    print(f'support choice identical: {100 * same_idx:.2f}%   max |b diff| {np.abs(ref[..., 2] - new["DBLK_r"][..., 2]).max():.2e}   '
          f'max |sigma diff| {np.abs(ref[..., 3] - new["DBLK_r"][..., 3]).max():.2e}   max |gamma diff| {np.abs(ref[..., 4] - new["DBLK_r"][..., 4]).max():.2e}')
