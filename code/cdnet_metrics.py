#!/usr/bin/env python3
"""
CDnet 2014 metrics from per-frame counts (written by cdnet_cpb_eval.py). Every variant comes from the SAME masks.

Per video (frames inside the temporal ROI whose GT is released; GT 85/170 pixels ignored, 50 = background):
  official CDnet tool : TP, FP, FN, TN summed over the frames -> Re, Sp, FPR, FNR, PWC, Pr, F
  thesis_fg           : Zhou (2019) Eq. 5.2-5.7 -- P_i and R_i per frame, averaged over the frames that contain
                        foreground (P_i skipped when nothing is predicted), F = 2PR/(P+R)
  meanF_fg            : mean of the per-frame F over the frames that contain foreground
  A video without any foreground in its scored frames (only possible in a --max_frames test) has undefined F and is
  left out of the means (reported as nan), instead of counting as 0.
Across videos:
  category value  = mean of its videos (official CDnet convention)  |  cat_pooled = F from the category's summed counts
  OVERALL         = mean of the 11 category values (official)  |  mean of all videos  |  all pixels pooled

Usage:  python cdnet_metrics.py --run <mode folder or its counts folder> [--csv summary.csv]
"""
import os, glob, csv, argparse
import numpy as np

CATS = ['badWeather', 'baseline', 'cameraJitter', 'dynamicBackground', 'intermittentObjectMotion', 'lowFramerate',
        'nightVideos', 'PTZ', 'shadow', 'thermal', 'turbulence']
OFFICIAL = ['Re', 'Sp', 'FPR', 'FNR', 'PWC', 'Pr', 'F']
EXTRA = ['thesis_fg', 'meanF_fg']
LIANG = 'Liang et al. 2021, CPB on CDnet: printed overall 0.633 | mean of their category rows 0.707'
nan = float('nan')


def video_metrics(rows):
    """rows: [frame, tp, fp, fn, tn, gt_fg, valid] (only frames with valid > 0)."""
    a = np.asarray(rows, dtype=np.float64)
    TP, FP, FN, TN = (a[:, i].sum() for i in (1, 2, 3, 4))
    m = dict(TP=TP, FP=FP, FN=FN, TN=TN, n_frames=len(a), n_fg_frames=int((a[:, 5] > 0).sum()))
    m['Sp'] = TN / (TN + FP) if TN + FP > 0 else nan
    m['FPR'] = FP / (FP + TN) if FP + TN > 0 else nan
    m['PWC'] = 100.0 * (FN + FP) / max(TP + FN + FP + TN, 1)
    if TP + FN == 0:                                   # no foreground at all -> F undefined
        m.update(Re=nan, FNR=nan, Pr=nan, F=nan, thesis_fg=nan, meanF_fg=nan)
        return m
    m['Re'] = TP / (TP + FN); m['FNR'] = FN / (TP + FN)
    m['Pr'] = TP / (TP + FP) if TP + FP > 0 else 0.0
    m['F'] = 2 * m['Pr'] * m['Re'] / (m['Pr'] + m['Re']) if m['Pr'] + m['Re'] > 0 else 0.0
    fg = a[a[:, 5] > 0]
    Pf = [r[1] / (r[1] + r[2]) for r in fg if r[1] + r[2] > 0]
    Rf = [r[1] / (r[1] + r[3]) for r in fg if r[1] + r[3] > 0]
    p, r = (float(np.mean(Pf)) if Pf else 0.0), (float(np.mean(Rf)) if Rf else 0.0)
    m['thesis_fg'] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    m['meanF_fg'] = float(np.mean([2 * x[1] / max(2 * x[1] + x[2] + x[3], 1) for x in fg]))
    return m


def read_counts(path):
    with open(path, newline='') as fh:
        rows = [[int(x) for x in r] for r in list(csv.reader(fh))[1:]]
    return [r for r in rows if r[6] > 0]               # frames whose GT is all 85/170 (unreleased GT) are not test frames


def load_run(folder):
    d = os.path.join(folder, 'counts') if os.path.isdir(os.path.join(folder, 'counts')) else folder
    V = {}
    for p in sorted(glob.glob(os.path.join(d, '*.csv'))):
        name = os.path.basename(p)[:-4]
        if '__' not in name:
            continue
        cat, vid = name.split('__', 1)
        rows = read_counts(p)
        if rows:
            V[(cat, vid)] = video_metrics(rows)
    return V, d


def fmt(x, w=7, d=3):
    return f'{x:{w}.{d}f}' if np.isfinite(x) else f'{"nan":>{w}s}'


def video_line(name, m):
    return (f'{name:40s} F {fmt(m["F"], 6)} | P {fmt(m["Pr"], 5, 2)} R {fmt(m["Re"], 5, 2)} Sp {fmt(m["Sp"], 6, 4)} '
            f'FPR {fmt(m["FPR"], 6, 4)} FNR {fmt(m["FNR"], 5, 3)} PWC {fmt(m["PWC"], 6, 2)} | thesis {fmt(m["thesis_fg"], 6)} '
            f'meanF {fmt(m["meanF_fg"], 6)} | frames {m["n_frames"]} (fg {m["n_fg_frames"]})')


def nanmean(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else nan


def summarize(V, title=''):
    """Returns (text, category table dict, overall dict)."""
    L = []
    if title:
        L += ['=' * 118, title, '=' * 118]
    cats = [c for c in CATS if any(k[0] == c for k in V)] + sorted({k[0] for k in V} - set(CATS))
    cols = OFFICIAL + EXTRA
    head = f'{"":40s}' + ''.join(f'{c:>10s}' for c in cols) + f'{"catPool":>10s}{"frames":>8s}{"fg fr":>7s}'
    L.append(head)
    C = {}
    for c in cats:
        vs = sorted(k for k in V if k[0] == c)
        for k in vs:
            m = V[k]
            L.append(f'  {k[1]:38s}' + ''.join(fmt(m[x], 10, 4 if x in ('Sp', 'FPR') else 3) for x in cols) + f'{"":10s}'
                     f'{m["n_frames"]:8d}{m["n_fg_frames"]:7d}')
        C[c] = {x: nanmean([V[k][x] for k in vs]) for x in cols}
        S = np.sum([[V[k][s] for s in ('TP', 'FP', 'FN', 'TN')] for k in vs], 0)
        C[c]['catPool'] = 2 * S[0] / (2 * S[0] + S[1] + S[2]) if S[0] + S[2] > 0 else nan
        L.append(f'{c.upper()[:38]:40s}' + ''.join(fmt(C[c][x], 10, 4 if x in ('Sp', 'FPR') else 3) for x in cols)
                 + fmt(C[c]['catPool'], 10) + f'{len(vs):8d} videos')
        L.append('')
    S = np.sum([[V[k][s] for s in ('TP', 'FP', 'FN', 'TN')] for k in V], 0)
    O = {x: nanmean([C[c][x] for c in cats]) for x in cols + ['catPool']}
    Ov = {x: nanmean([V[k][x] for k in V]) for x in cols}
    L.append('-' * 118)
    L.append(f'{"OVERALL = mean of categories (official)":40s}' + ''.join(fmt(O[x], 10, 4 if x in ('Sp', 'FPR') else 3) for x in cols) + fmt(O['catPool'], 10))
    L.append(f'{"OVERALL = mean of videos":40s}' + ''.join(fmt(Ov[x], 10, 4 if x in ('Sp', 'FPR') else 3) for x in cols))
    pooled = 2 * S[0] / (2 * S[0] + S[1] + S[2]) if S[0] + S[2] > 0 else nan
    L.append(f'{"OVERALL = all pixels pooled F":40s}{fmt(pooled, 9)}')
    L.append('')
    L.append('F-measure, the three ways it is usually reported:')
    L.append(f'  official CDnet (per-video pooled -> category mean -> mean of {len(cats)} categories) : {fmt(O["F"], 6, 4)}')
    L.append(f'  thesis, Zhou 2019 (per-frame P, R averaged -> category mean -> mean of categories)  : {fmt(O["thesis_fg"], 6, 4)}')
    L.append(f'  pooled (all pixels of all videos)                                                  : {fmt(pooled, 6, 4)}')
    L.append(f'  mean of videos: official {fmt(Ov["F"], 6, 4)}  thesis {fmt(Ov["thesis_fg"], 6, 4)}   |  mean of category-pooled F {fmt(O["catPool"], 6, 4)}')
    undef = [k[1] for k in V if not np.isfinite(V[k]['F'])]
    if undef:
        L.append(f'NOTE: {len(undef)} videos have no foreground in the scored frames (F undefined, left out of the means): {", ".join(undef)}')
    if len(cats) < 11:
        L.append(f'NOTE: only {len(cats)} of 11 categories present -> overall values are not comparable with full-benchmark numbers.')
    L.append(LIANG)
    return '\n'.join(L), C, dict(official=O, videos=Ov, pooled=pooled)


def write_csv(V, path):
    cols = OFFICIAL + EXTRA
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['category', 'video'] + cols + ['TP', 'FP', 'FN', 'TN', 'n_frames', 'n_fg_frames'])
        for k in sorted(V):
            m = V[k]
            w.writerow(list(k) + [round(m[x], 6) if np.isfinite(m[x]) else '' for x in cols]
                       + [int(m[s]) for s in ('TP', 'FP', 'FN', 'TN')] + [m['n_frames'], m['n_fg_frames']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', '--counts', dest='run', required=True, help='mode folder (e.g. cdnet_runs/frozen) or its counts folder')
    ap.add_argument('--csv', '--out', dest='csv', default=None, help='optional per-video CSV')
    a = ap.parse_args()
    V, d = load_run(a.run)
    if not V:
        n = len(glob.glob(os.path.join(d, '*.csv')))
        print(f'No usable per-video counts in {os.path.abspath(d)} ({n} .csv files). Run cdnet_cpb_eval.py first and point --run to its mode folder.')
        return
    txt, _, _ = summarize(V, f'{len(V)} videos from {os.path.abspath(d)}')
    print(txt)
    if a.csv:
        write_csv(V, a.csv); print(f'\nper-video CSV written: {os.path.abspath(a.csv)}')


if __name__ == '__main__':
    main()
