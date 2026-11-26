#!/usr/bin/env python3
"""Pure CPB mask (guidance = zero): binary mask and continuous score at native resolution."""
import numpy as np
import cv2
import cpb_kernels as F


def detect_channels(model):
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


def pure_cpb_with_score(model, rgb_f32, gt_shape, af, floor, bth, ref):
    """Pure CPB (guidance = ZERO) returning BOTH the native-res binary mask AND the
    native-res continuous unsteady-fraction score (needed by the reliability gate)."""
    Hf, Wf = model['img_size']
    ws = int(np.ceil(Hf / 8)); hs = int(np.ceil(Wf / 8))
    chans, _ = detect_channels(model)
    pool = chans[0][1].shape[2]
    fr = cv2.resize(rgb_f32, (Wf, Hf)) if rgb_f32.shape[:2] != (Hf, Wf) else rgb_f32
    guid_fg = np.zeros((Hf, Wf), np.uint8)
    gcnt = F.block_fg_count(guid_fg, ws, hs)
    s1_acc = None; sc_acc = None
    for nm, Dc in chans:
        chan = F.to_luma(fr) if nm == 'L' else fr[:, :, {'R': 0, 'G': 1, 'B': 2}[nm]].astype(np.float32)
        DBS = F.block_mean(chan, ws, hs)
        s1 = np.zeros((Hf, Wf), np.uint8); sc = np.zeros((Hf, Wf), np.float32); su = np.zeros((Hf, Wf), np.float32)
        F.k_stage1(chan, DBS, gcnt, guid_fg, Dc, s1, sc, su,
                   float(af), float(floor), F.LAMBDA_DEC, float(bth), int(ref), pool, True)
        s1_acc = s1 if s1_acc is None else np.maximum(s1_acc, s1)
        sc_acc = sc if sc_acc is None else np.maximum(sc_acc, sc)
    s1_acc = cv2.medianBlur(s1_acc, 3)                     # Zhou medfilt2
    mask = cv2.resize(s1_acc, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(sc_acc, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask, score


def pure_cpb_mask(model, rgb_f32, gt_shape, af, floor, bth, ref):
    """Backwards-compatible wrapper: pure-CPB binary mask only."""
    return pure_cpb_with_score(model, rgb_f32, gt_shape, af, floor, bth, ref)[0]


