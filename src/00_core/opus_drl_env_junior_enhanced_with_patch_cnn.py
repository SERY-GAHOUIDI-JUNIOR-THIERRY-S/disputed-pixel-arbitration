#!/usr/bin/env python3
"""
================================================================================
DRL ENVIRONMENT (junior v4)  —  3-way gate + 23-D state + PATCH tensor  + RGB
================================================================================
Wraps the FROZEN framework (opus_final_hod_framework.py). Reuses the exact CPB
Stage-1/2/3 kernels and the exact models.

WHAT'S NEW vs v3:
  + TEMPORAL PERSISTENCE  (score_win_persist)
  + PATCH TENSOR  (patches())
  + RGB SUPPORT (v4b): the env now auto-detects an RGB CPB model (DBLK_r/g/b
    not all equal) exactly like HoDPipeline, and runs Stage-1/2/3 PER CHANNEL,
    OR-ing (np.maximum) the s1 / score / supp / s3 maps across R,G,B — byte-for-
    byte what opus_final_hod_framework.HoDPipeline does. A GRAY model (Dr==Dg==Db)
    takes the single-luma path, which is byte-identical to the previous version,
    so existing gray runs and trained agents are unaffected. The scalar STATE,
    the appearance/patch maps and STATE_DIM are unchanged (luma-based, as before).

STATE (23 scalar features): unchanged.
PATCH : [N, 3, P, P]  (score, guid_prob, gray/255) — unchanged.

3-WAY GATE (unchanged): {0: trust CPB, 1: trust guidance, 2: keep baseline}.
================================================================================
"""

import numpy as np
import cv2

try:
    import opus_final_hod_framework as F
except Exception:
    import opus_liang_hod_framework as F

BLOCK_SIZE = F.BLOCK_SIZE
GT_FG = F.GT_FG
GT_EXCLUDE = F.GT_EXCLUDE
LAMBDA_DEC = F.LAMBDA_DEC

PATCH_CHANNELS = 3          # score, guid_prob, gray/255


# ============================================================================
# CDnet metric
# ============================================================================
def frame_counts(pred, gt):
    return F.frame_counts(pred, gt)


def f1_from_counts(tp, fp, fn, tn):
    e = 1e-10
    pr = tp / (tp + fp + e); rc = tp / (tp + fn + e)
    return 2 * pr * rc / (pr + rc + e)


# ============================================================================
# channel detection (identical rule to HoDPipeline)
# ============================================================================
def _detect_channels(model):
    """Return (channels, gray_model) where channels is a list of (name, D):
    [('L', Dr)] for a gray model (Dr==Dg==Db), else [('R',Dr),('G',Dg),('B',Db)].
    Falls back to gray if the model has no separate DBLK_g/DBLK_b."""
    Dr = np.ascontiguousarray(model['DBLK_r'], np.float32)
    Dg = np.ascontiguousarray(model.get('DBLK_g', model['DBLK_r']), np.float32)
    Db = np.ascontiguousarray(model.get('DBLK_b', model['DBLK_r']), np.float32)
    gray_model = np.allclose(Dr, Dg) and np.allclose(Dr, Db)
    if gray_model:
        return [('L', Dr)], True
    return [('R', Dr), ('G', Dg), ('B', Db)], False


# ============================================================================
# Stand-alone Stage-1 (used by the trainer's in-order temporal pass)
# ============================================================================
def compute_stage1(model, frame_rgb_native, guid_native, af, floor, bth, ref, hod_pool=True):
    """Return the CPB score map at fit resolution (no full env build).
    RGB-aware: OR-s (np.maximum) the per-channel scores so the temporal-window
    features match the live env score exactly."""
    Hf, Wf = model['img_size']
    ws = int(np.ceil(Hf / BLOCK_SIZE)); hs = int(np.ceil(Wf / BLOCK_SIZE))
    channels, _ = _detect_channels(model)
    pool = channels[0][1].shape[2]
    if frame_rgb_native.shape[:2] != (Hf, Wf):
        fr = cv2.resize(frame_rgb_native, (Wf, Hf)).astype(np.float32)
        g = cv2.resize(guid_native, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
    else:
        fr = frame_rgb_native.astype(np.float32); g = guid_native
    guid_fg = (g > 127).astype(np.uint8)
    gcnt = F.block_fg_count(guid_fg, ws, hs)
    score_acc = None
    for nm, Dc in channels:
        chan = F.to_luma(fr) if nm == 'L' else fr[:, :, {'R': 0, 'G': 1, 'B': 2}[nm]].astype(np.float32)
        DBS = F.block_mean(chan, ws, hs)
        s1 = np.zeros((Hf, Wf), np.uint8); score = np.zeros((Hf, Wf), np.float32); supp = np.zeros((Hf, Wf), np.float32)
        F.k_stage1(chan, DBS, gcnt, guid_fg, Dc, s1, score, supp,
                   float(af), float(floor), LAMBDA_DEC, float(bth), int(ref), pool, bool(hod_pool))
        score_acc = score if score_acc is None else np.maximum(score_acc, score)
    return score_acc


# ============================================================================
# FrameEnv
# ============================================================================
class FrameEnv:
    FEATURE_NAMES = [
        # CPB
        'score', 'cpb_margin', 'supp', 'score_delta', 'score_win_mean', 'score_win_std',
        'score_win_slope', 'score_win_persist', 'score_ctx_mean', 'score_ctx_std', 'blob_size',
        # GUIDANCE
        'guid', 'guid_prob', 'guid_conf', 'soft_disagree', 'guid_dist', 'guid_density',
        'dist_strong_guid',
        # CONTEXT
        'motion', 'edge', 'texture', 'contrast', 'guid_reliability',
    ]
    STATE_DIM = len(FEATURE_NAMES)

    def __init__(self, model, af, floor, bth, ref, sigema, hod_pool=True):
        self.af = float(af); self.floor = float(floor); self.bth = float(bth)
        self.ref = int(ref); self.sigema = float(sigema); self.hod_pool = bool(hod_pool)
        self.Hf, self.Wf = model['img_size']
        self.ws = int(np.ceil(self.Hf / BLOCK_SIZE)); self.hs = int(np.ceil(self.Wf / BLOCK_SIZE))
        # RGB auto-detect (identical rule to HoDPipeline). self.D kept as the
        # first channel's dictionary for backward compatibility.
        self.channels, self.gray_model = _detect_channels(model)
        self.D = self.channels[0][1]
        self.pool = self.D.shape[2]
        self._prev_gray = None; self._prev_score = None

    # ---- build per-frame cache -----------------------------------------
    def set_frame(self, frame_rgb_native, guid_native, gt_native,
                  guid_prob_native=None, win_mean=None, win_std=None, win_slope=None, win_persist=None):
        native = frame_rgb_native.shape[:2]
        if native != (self.Hf, self.Wf):
            fr = cv2.resize(frame_rgb_native, (self.Wf, self.Hf)).astype(np.float32)
            g = cv2.resize(guid_native, (self.Wf, self.Hf), interpolation=cv2.INTER_NEAREST)
            gp = (cv2.resize(guid_prob_native, (self.Wf, self.Hf)).astype(np.float32)
                  if guid_prob_native is not None else None)
        else:
            fr = frame_rgb_native.astype(np.float32); g = guid_native
            gp = guid_prob_native.astype(np.float32) if guid_prob_native is not None else None
        self.native = native
        self.gt = gt_native
        self.gray = F.to_luma(fr)                                   # luma: appearance/patch/context features
        self.guid_fg = (g > 127).astype(np.uint8)
        self.guid_prob = gp if gp is not None else self.guid_fg.astype(np.float32)   # fallback: binary
        self.guid_reliability = float(self.guid_fg.mean())

        self.gcnt = F.block_fg_count(self.guid_fg, self.ws, self.hs)  # channel-independent (guidance)
        H, W = self.gray.shape

        # ---- CPB Stage-1 per channel, OR across channels (== HoDPipeline) ----
        # For a gray model this loops ONCE over the luma channel and is byte-
        # identical to the previous single-channel implementation.
        self._chans = []          # cached per channel: (intensity, DBS_c, DBM_c, D_c, s1_c)
        s1_acc = None; score_acc = None; supp_acc = None
        for nm, Dc in self.channels:
            chan = self.gray if nm == 'L' else fr[:, :, {'R': 0, 'G': 1, 'B': 2}[nm]].astype(np.float32)
            DBS_c = F.block_mean(chan, self.ws, self.hs)
            DBM_c = F.block_median(chan, self.ws, self.hs)
            s1c = np.zeros((H, W), np.uint8); scc = np.zeros((H, W), np.float32); spc = np.zeros((H, W), np.float32)
            F.k_stage1(chan, DBS_c, self.gcnt, self.guid_fg, Dc,
                       s1c, scc, spc, self.af, self.floor, LAMBDA_DEC,
                       self.bth, self.ref, self.pool, self.hod_pool)
            self._chans.append((chan, DBS_c, DBM_c, Dc, s1c))
            s1_acc = s1c if s1_acc is None else np.maximum(s1_acc, s1c)
            score_acc = scc if score_acc is None else np.maximum(score_acc, scc)
            supp_acc = spc if supp_acc is None else np.maximum(supp_acc, spc)
        # back-compat single-channel handles (== luma for gray, R plane for rgb)
        self.DBS, self.DBM, self.D = self._chans[0][1], self._chans[0][2], self._chans[0][3]
        self.score = score_acc; self.supp = supp_acc; self.s1_default = s1_acc

        # temporal window (fallbacks if not supplied)
        score = self.score
        self._win_mean = win_mean if win_mean is not None else score.copy()
        self._win_std = win_std if win_std is not None else np.zeros_like(score)
        self._win_slope = win_slope if win_slope is not None else np.zeros_like(score)
        self._win_persist = win_persist if win_persist is not None else (score > 0.5).astype(np.float32)

        self._make_feature_maps()

        # patch maps for the CNN branch: [C,H,W] in ~[0,1]
        self._patch_maps = np.stack([
            self.score.astype(np.float32),
            self.guid_prob.astype(np.float32),
            np.clip(self.gray / 255.0, 0.0, 1.0).astype(np.float32),
        ], axis=0)

        # baseline = FULL framework (Stages 1-2-3), per channel then OR — matches HoDPipeline
        self._baseline_mask = self._segment_baseline()
        self.base_pred_fit = (self._s3_fit == 255)          # baseline final label @ fit res (bool)
        self._baseline_f1 = self.f1(self._baseline_mask)

        # GT / valid at fit resolution (for the dense per-pixel reward)
        gtf = cv2.resize(gt_native, (self.Wf, self.Hf), interpolation=cv2.INTER_NEAREST)
        self.gtfg_fit = (gtf == GT_FG)
        self.valid_fit = np.ones(gtf.shape, bool)
        for v in GT_EXCLUDE:
            self.valid_fit &= (gtf != v)

        # 1-step temporal
        self._score_delta = (np.abs(score - self._prev_score)
                             if self._prev_score is not None else np.zeros_like(score))
        self._motion = (cv2.absdiff(self.gray, self._prev_gray) / 255.0
                        if self._prev_gray is not None else np.zeros_like(self.gray))
        self._prev_gray = self.gray.copy(); self._prev_score = score.copy()
        return self

    def _make_feature_maps(self):
        g = self.gray.astype(np.float32)
        sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        self._edge = np.clip(np.sqrt(sx * sx + sy * sy) / 255.0, 0, 4)
        mean = cv2.blur(g, (5, 5))
        self._texture = np.clip((cv2.blur(g * g, (5, 5)) - mean * mean) / (255.0 ** 2), 0, 1)
        gmax = cv2.dilate(g, np.ones((5, 5), np.uint8)); gmin = cv2.erode(g, np.ones((5, 5), np.uint8))
        self._contrast = (gmax - gmin) / 255.0
        fg = (self.guid_fg > 0).astype(np.uint8)
        inside = cv2.distanceTransform(fg, cv2.DIST_L2, 3) if fg.any() else np.zeros_like(g)
        outside = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3) if (1 - fg).any() else np.zeros_like(g)
        self._guid_dist = np.clip((inside + outside) / 20.0, 0, 1)
        self._guid_density = cv2.blur(fg.astype(np.float32), (9, 9))

        # spatial-context features
        sc = self.score.astype(np.float32)
        self._score_ctx_mean = cv2.blur(sc, (9, 9))
        m2 = cv2.blur(sc * sc, (9, 9))
        self._score_ctx_std = np.sqrt(np.clip(m2 - self._score_ctx_mean * self._score_ctx_mean, 0.0, None))
        fgm = (self.s1_default > 0).astype(np.uint8)
        ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(fgm, connectivity=8)
        sizes = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        area = float(fgm.shape[0] * fgm.shape[1])
        size_map = sizes[lbl]
        self._blob_size = np.clip(np.log1p(size_map) / np.log1p(area + 1.0), 0.0, 1.0).astype(np.float32)
        strong = (self.guid_prob > 0.6).astype(np.uint8)
        if strong.any():
            dstrong = cv2.distanceTransform(1 - strong, cv2.DIST_L2, 3)
            self._dist_strong = np.clip(dstrong / 30.0, 0.0, 1.0).astype(np.float32)
        else:
            self._dist_strong = np.ones_like(sc)

    # ---- disputes ------------------------------------------------------
    def dispute_ij(self):
        return np.where((self.s1_default > 0) != (self.guid_fg > 0))

    # ---- scalar state --------------------------------------------------
    def state(self, rows, cols):
        r, c = rows, cols
        gp = self.guid_prob[r, c]
        feats = np.stack([
            # CPB
            self.score[r, c],
            np.abs(self.score[r, c] - 0.5) * 2.0,
            np.clip(self.supp[r, c] / float(self.ref), 0, 1),
            self._score_delta[r, c],
            self._win_mean[r, c],
            self._win_std[r, c],
            np.clip(self._win_slope[r, c] * 5.0, -1.0, 1.0),
            self._win_persist[r, c],
            self._score_ctx_mean[r, c],
            self._score_ctx_std[r, c],
            self._blob_size[r, c],
            # GUIDANCE
            self.guid_fg[r, c].astype(np.float32),
            gp,
            np.abs(gp - 0.5) * 2.0,
            np.abs(self.score[r, c] - gp),
            self._guid_dist[r, c],
            self._guid_density[r, c],
            self._dist_strong[r, c],
            # CONTEXT
            self._motion[r, c],
            self._edge[r, c],
            self._texture[r, c],
            self._contrast[r, c],
            np.full(r.shape, self.guid_reliability, np.float32),
        ], axis=1).astype(np.float32)
        return feats

    # ---- patch tensor for the CNN branch -------------------------------
    def patches(self, rows, cols, k=4):
        """Return [N, C, P, P] (P=2k+1) patches from [score, guid_prob, gray] around
        each disputed pixel. Edge-padded so border pixels are handled cleanly."""
        P = 2 * k + 1
        maps = self._patch_maps                       # [C,H,W]
        C = maps.shape[0]
        padded = np.pad(maps, ((0, 0), (k, k), (k, k)), mode='edge')
        N = len(rows)
        out = np.empty((N, C, P, P), np.float32)
        rr = np.asarray(rows); cc = np.asarray(cols)
        for dy in range(P):
            ry = rr + dy
            for dx in range(P):
                out[:, :, dy, dx] = padded[:, ry, cc + dx].T    # [C,N] -> [N,C]
        return out

    # ---- baseline (full framework) -------------------------------------
    def _segment_baseline(self):
        """Full framework baseline (Stages 1-2-3): per channel, then OR across
        channels — byte-identical to HoDPipeline.run. For a gray model this runs
        once on the luma channel (== the old single-channel _segment_from_s1)."""
        H, W = self.gray.shape
        s3_acc = None
        for (chan, DBS_c, DBM_c, Dc, s1c) in self._chans:
            s2 = np.zeros((H, W), np.uint8)
            F.k_stage2(chan, DBS_c, self.gcnt, self.guid_fg, Dc, s1c, s2,
                       self.af, self.floor, self.bth, self.ref, self.pool)
            s3 = np.zeros((H, W), np.uint8)
            F.k_stage3(chan, DBM_c, self.gcnt, self.guid_fg, Dc, s2, s3,
                       self.sigema, self.bth, self.ref, self.pool)
            s3_acc = s3 if s3_acc is None else np.maximum(s3_acc, s3)
        self._s3_fit = s3_acc
        return self._to_native(s3_acc)

    # ---- legacy single-channel baseline (kept for back-compat; gray only) ----
    def _segment_from_s1(self, s1):
        """Legacy helper: Stages 2-3 on ONE channel (self.DBS/DBM/D). Kept so any
        external code that calls it on a gray model still works byte-identically.
        For RGB models use _segment_baseline (which loops all channels)."""
        H, W = self.gray.shape
        s2 = np.zeros((H, W), np.uint8)
        F.k_stage2(self.gray, self.DBS, self.gcnt, self.guid_fg, self.D, s1, s2,
                   self.af, self.floor, self.bth, self.ref, self.pool)
        s3 = np.zeros((H, W), np.uint8)
        F.k_stage3(self.gray, self.DBM, self.gcnt, self.guid_fg, self.D, s2, s3,
                   self.sigema, self.bth, self.ref, self.pool)
        self._s3_fit = s3
        return self._to_native(s3)

    # ---- gate segmentation ---------------------------------------------
    def segment_gate(self, labels, rows, cols):
        final = self.s1_default.copy()
        final[rows, cols] = (np.asarray(labels).astype(np.uint8) * 255)
        return self._to_native(final)

    def gate3_labels(self, actions, rows, cols):
        a = np.asarray(actions)
        cpb_lab = self.s1_default[rows, cols] > 0
        guid_lab = self.guid_fg[rows, cols] > 0
        base_lab = self.base_pred_fit[rows, cols]
        return np.where(a == 0, cpb_lab, np.where(a == 1, guid_lab, base_lab))

    def segment_gate3(self, actions, rows, cols):
        lab = self.gate3_labels(actions, rows, cols)
        final = self.s1_default.copy()
        final[rows, cols] = (lab.astype(np.uint8) * 255)
        return self._to_native(final)

    def _to_native(self, mask_fit):
        if mask_fit.shape != self.native:
            return cv2.resize(mask_fit, (self.native[1], self.native[0]), interpolation=cv2.INTER_NEAREST)
        return mask_fit

    # ---- scoring -------------------------------------------------------
    def f1(self, mask_native):
        if mask_native.shape != self.gt.shape:
            mask_native = cv2.resize(mask_native, (self.gt.shape[1], self.gt.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        tp, fp, fn, tn = frame_counts(mask_native, self.gt)
        return f1_from_counts(tp, fp, fn, tn)

    def counts(self, mask_native):
        if mask_native.shape != self.gt.shape:
            mask_native = cv2.resize(mask_native, (self.gt.shape[1], self.gt.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        return frame_counts(mask_native, self.gt)

    def baseline_f1(self):
        return self._baseline_f1

    def baseline_mask(self):
        return self._baseline_mask