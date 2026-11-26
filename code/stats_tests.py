#!/usr/bin/env python3
"""
Paired statistical comparisons for the controlled tables (LASIESTA, LIMU; WallFlower = win/loss only).

Metric: per-frame F (Zhou thesis) per sequence, dataset value = mean over sequences (as in the tables).
Frames inside a video are correlated, so frames are never treated as independent samples:
  * LASIESTA (20 sequences): exact paired Wilcoxon signed-rank test and sign test over SEQUENCES, Holm-corrected
    within each family (one family per arbiter); hierarchical bootstrap CI of the difference in the dataset mean:
    resample sequences with replacement, then a moving-block bootstrap of frames inside every drawn sequence
    (block = 10 scored frames), both systems evaluated on the SAME resampled frames (paired).
  * LIMU (3 sequences): too few sequences for a sequence-level test; within-sequence moving-block bootstrap
    (block = 10 GT frames = 150 video frames) of each sequence, CI of the difference in the mean over the 3
    sequences and per-sequence CIs.
  * WallFlower (one evaluated frame per sequence): no test possible; wins / losses per sequence.
"""
import os, sys, csv, json, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np
from scipy.stats import wilcoxon, binomtest

H = PATHS.RESULTS


def bar(i, n, label, t0, w=30):
    f = (i + 1) / n; el = time.time() - t0; eta = el / f - el
    sys.stdout.write(f'\r  {label:28s} [{"#" * int(w * f):{w}s}] {100 * f:5.1f}%  ETA {eta:5.0f}s'); sys.stdout.flush()
    if i + 1 == n:
        sys.stdout.write('\n')


def pr_arrays(rows):
    """rows: list of (tp, fp, fn, has_fg) -> P_t (nan if undefined), R_t (nan if no GT foreground)."""
    a = np.asarray(rows, float); tp, fp, fn, fg = a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(fg & (tp + fp > 0), tp / (tp + fp), np.nan)
        R = np.where(fg & (tp + fn > 0), tp / (tp + fn), np.nan)
    return P, R


def fstat(P, R):
    """Per-frame F of a sequence: F from the mean P and mean R (P over fg frames with a prediction, R over fg frames)."""
    with np.errstate(invalid='ignore'):
        Pm = np.nanmean(P, axis=-1); Rm = np.nanmean(R, axis=-1)
    Pm = np.nan_to_num(Pm); Rm = np.nan_to_num(Rm)
    return np.where(Pm + Rm > 0, 2 * Pm * Rm / np.maximum(Pm + Rm, 1e-12), 0.0)


def block_idx(T, L, rng):
    if T <= L:
        return rng.randint(0, T, T)
    starts = rng.randint(0, T - L + 1, int(np.ceil(T / L)))
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:T]


# ------------------------------------------------------------------------------------------------- loaders
def las_systems():
    def ext(sub, key):
        return lambda s: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/lasiesta_arbiters/{sub}/seq/{s}.json'))['rows'][key]]
    S = json.load(open(f'{H}/semantic_bgs/sbgs_lasiesta_rows.json'))
    sb = lambda k: (lambda s: [(r[1], r[2], r[3], r[5]) for r in S[k][s]])
    bg = lambda m: (lambda s: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/classical_bgs/{m}__lasiesta__{s}.json'))['result']])
    return {'Arbiter (DeepLab)': ext('deeplab', 'MLP'), 'Arbiter (PSPNet)': ext('pspnet', 'MLP'),
            'CPB': ext('deeplab', 'CPB'), 'DeepLabV3+ alone': ext('deeplab', 'guidance'), 'PSPNet alone': ext('pspnet', 'guidance'),
            'Liang framework (DeepLab)': ext('deeplab', 'framework'), 'Liang framework (PSPNet)': ext('pspnet', 'framework'),
            'Minimal fusion (DeepLab)': ext('fusion', 'MLP'),
            'SemanticBGS on CPB (DeepLab)': sb('SemanticBGS(CPB, deeplab)'), 'SemanticBGS on CPB (PSPNet)': sb('SemanticBGS(CPB, pspnet)'),
            'SemanticBGS on SuBSENSE (DeepLab)': sb('SemanticBGS(SuBSENSE, deeplab)'), 'SemanticBGS on SuBSENSE (PSPNet)': sb('SemanticBGS(SuBSENSE, pspnet)'),
            'PAWCS': bg('PAWCS'), 'SuBSENSE': bg('SuBSENSE'),
            'ZBS': (lambda s: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/zbs/ZBS__lasiesta__{s}.json'))['result']]),
            'ZBS-p1': (lambda s: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/zbs/ZBS-p1__lasiesta__{s}.json'))['result']])}


def limu_systems():
    def csvr(path, key):
        def g(s):
            with open(path.format(s=s), newline='') as fh:
                return [(int(r[f'{key}_tp']), int(r[f'{key}_fp']), int(r[f'{key}_fn']), r[f'{key}_has_fg'] == 'True') for r in csv.DictReader(fh)]
        return g
    dl, ps, fu = f'{H}/arbiter_deeplab/limu/limu_{{s}}_perframe.csv', f'{H}/arbiter_pspnet/limu/limu_{{s}}_perframe.csv', f'{H}/fusion_control/limu/limu_{{s}}_perframe.csv'
    out = {'Arbiter (DeepLab)': csvr(dl, 'MLP'), 'Arbiter (PSPNet)': csvr(ps, 'MLP'), 'CPB': csvr(dl, 'CPB'),
           'DeepLabV3+ alone': csvr(dl, 'guidance'), 'PSPNet alone': csvr(ps, 'guidance'),
           'Liang framework (DeepLab)': csvr(dl, 'framework'), 'Liang framework (PSPNet)': csvr(ps, 'framework'),
           'Minimal fusion (DeepLab)': csvr(fu, 'MLP')}
    sp = f'{H}/semantic_bgs_limu_rows/sbgs_limu_rows.json'
    if os.path.exists(sp):
        S = json.load(open(sp))
        for k, nm in (('SemanticBGS(CPB, deeplab)', 'SemanticBGS on CPB (DeepLab)'), ('SemanticBGS(CPB, pspnet)', 'SemanticBGS on CPB (PSPNet)')):
            if k in S and len(S[k]) == 3:
                out[nm] = (lambda s, k=k: [tuple(r) for r in S[k][s]])
    for m in ('SuBSENSE', 'PAWCS'):
        out[m] = (lambda s, m=m: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/classical_bgs/{m}__limu__{s}.json'))['result']])
    for m in ('ZBS', 'ZBS-p1'):
        out[m] = (lambda s, m=m: [(r[1], r[2], r[3], r[5]) for r in json.load(open(f'{H}/zbs/{m}__limu__{s}.json'))['result']])
    return out


# ------------------------------------------------------------------------------------------------- analysis
def analyse(dataset, ref, comps, B, block, seed=0):
    systems = las_systems() if dataset == 'lasiesta' else limu_systems()
    seqs = (sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{H}/lasiesta_arbiters/deeplab/seq/*.json')) if dataset == 'lasiesta'
            else ['CameraParameter', 'Intersection', 'LightSwitch'])
    names = [ref] + [c for c in comps if c in systems]
    PR = {}
    for s in seqs:
        arrs = [pr_arrays(systems[n](s)) for n in names]
        T = {len(a[0]) for a in arrs}
        assert len(T) == 1, f'{dataset} {s}: frame counts differ {T}'
        PR[s] = (np.stack([a[0] for a in arrs]), np.stack([a[1] for a in arrs]))
    F = np.array([[fstat(PR[s][0][k], PR[s][1][k]) for s in seqs] for k in range(len(names))])   # (systems, seqs)
    rng = np.random.RandomState(seed); boot = np.zeros((B, len(names))); bseq = np.zeros((B, len(names), len(seqs)))
    t0 = time.time()
    for b in range(B):
        draw = rng.randint(0, len(seqs), len(seqs)) if dataset == 'lasiesta' else np.arange(len(seqs))
        vals = []
        for j, si in enumerate(draw):
            P, R = PR[seqs[si]]; idx = block_idx(P.shape[1], block, rng)
            v = fstat(P[:, idx], R[:, idx]); vals.append(v)
            if dataset != 'lasiesta':
                bseq[b, :, j] = v
        boot[b] = np.mean(vals, 0)
        if b % 20 == 0 or b == B - 1:
            bar(b, B, f'{dataset}: bootstrap', t0)
    res = []
    for k, n in enumerate(names[1:], 1):
        d = F[0] - F[k]; db = boot[:, 0] - boot[:, k]
        r = dict(comp=n, ref_mean=float(F[0].mean()), comp_mean=float(F[k].mean()), diff=float(d.mean()),
                 ci=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], p_boot=float(2 * min((db <= 0).mean(), (db >= 0).mean())),
                 wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), per_seq=dict(zip(seqs, map(float, d))))
        if dataset == 'lasiesta':
            nz = d[d != 0]
            r['p_wilcoxon'] = float(wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='exact').pvalue) if len(nz) else 1.0
            r['p_sign'] = float(binomtest(r['wins'], r['wins'] + r['losses'], 0.5).pvalue) if r['wins'] + r['losses'] else 1.0
        else:
            sb = bseq[:, 0, :] - bseq[:, k, :]
            r['per_seq_ci'] = {s: [float(np.percentile(sb[:, j], 2.5)), float(np.percentile(sb[:, j], 97.5))] for j, s in enumerate(seqs)}
        res.append(r)
    if dataset == 'lasiesta':                                            # Holm within this family
        order = np.argsort([r['p_wilcoxon'] for r in res]); m = len(res); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * res[i]['p_wilcoxon'])); res[i]['p_holm'] = run
    return res


def show(dataset, ref, res):
    print(f'\n  {dataset.upper()}: {ref} vs each competitor (difference = {ref} - competitor, per-frame F, mean over sequences)')
    if dataset == 'lasiesta':
        print(f'  {"competitor":36s} {"theirs":>7s} {"diff":>8s} {"95% CI (hier. bootstrap)":>26s} {"W/L":>6s} {"p Wilcoxon":>11s} {"p Holm":>8s} {"p sign":>8s}')
        for r in res:
            print(f'  {r["comp"]:36s} {r["comp_mean"]:7.3f} {r["diff"]:+8.3f}   [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]      {r["wins"]:2d}/{r["losses"]:<2d} '
                  f'{r["p_wilcoxon"]:11.2g} {r["p_holm"]:8.2g} {r["p_sign"]:8.2g}')
    else:
        print(f'  {"competitor":36s} {"theirs":>7s} {"diff":>8s} {"95% CI (block bootstrap)":>26s} {"W/L":>5s}   per-sequence diff [95% CI]')
        for r in res:
            ps = '  '.join(f'{s[:5]} {r["per_seq"][s]:+.3f} [{r["per_seq_ci"][s][0]:+.3f},{r["per_seq_ci"][s][1]:+.3f}]' for s in r['per_seq'])
            print(f'  {r["comp"]:36s} {r["comp_mean"]:7.3f} {r["diff"]:+8.3f}   [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]      {r["wins"]}/{r["losses"]}   {ps}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['lasiesta', 'limu']); ap.add_argument('--ref', required=True)
    ap.add_argument('--comps', required=True); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--out', default=f'{H}/stats')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = analyse(a.dataset, a.ref, a.comps.split('|'), a.B, a.block)
    show(a.dataset, a.ref, res)
    tag = f'{a.dataset}__{a.ref.replace(" ", "_").replace("(", "").replace(")", "")}'
    json.dump(dict(dataset=a.dataset, ref=a.ref, B=a.B, block=a.block, results=res), open(os.path.join(a.out, tag + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
