#!/usr/bin/env python3
"""Within-sequence aggregation variants for sparse-GT benchmarks (LIMU: GT every 15 frames, many frames empty).
rows: list of (tp, fp, fn, has_fg) per GT frame.
  pooled      : F from TP, FP, FN summed over all GT frames                                (what we reported)
  thesis_fg   : Zhou 2019 thesis Eq.5.4-5.7: P and R averaged over the test frames, F = 2PR/(P+R);
                test frames = GT frames that contain foreground (P_i skipped if nothing predicted)
  thesis_all  : same, but precision also averaged over EMPTY GT frames (P_i = 0 there whenever FP > 0)
  meanF_fg    : mean of per-frame F over GT frames that contain foreground
"""
import numpy as np


def metrics(rows):
    rows = [tuple(r) for r in rows]
    TP = sum(r[0] for r in rows); FP = sum(r[1] for r in rows); FN = sum(r[2] for r in rows)
    out = {'pooled': 2 * TP / max(2 * TP + FP + FN, 1)}
    fg = [r for r in rows if r[3]]
    Pf = [r[0] / (r[0] + r[1]) for r in fg if r[0] + r[1] > 0]
    Rf = [r[0] / (r[0] + r[2]) for r in fg if r[0] + r[2] > 0]
    P, R = (np.mean(Pf) if Pf else 0.0), (np.mean(Rf) if Rf else 0.0)
    out['thesis_fg'] = 2 * P * R / (P + R) if P + R > 0 else 0.0
    out['P_fg'], out['R_fg'] = P, R
    Pa = Pf + [0.0 for r in rows if not r[3] and r[1] > 0]
    Pa_ = np.mean(Pa) if Pa else 0.0
    out['thesis_all'] = 2 * Pa_ * R / (Pa_ + R) if Pa_ + R > 0 else 0.0
    out['meanF_fg'] = float(np.mean([2 * r[0] / max(2 * r[0] + r[1] + r[2], 1) for r in fg])) if fg else float('nan')
    return out


KEYS = ['pooled', 'thesis_fg', 'thesis_all', 'meanF_fg']
