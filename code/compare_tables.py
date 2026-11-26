#!/usr/bin/env python3
"""Builds the controlled-comparison tables (LIMU, WallFlower, LASIESTA): our systems + authors' released code +
BGSLibrary baselines, all under the same protocol, plus the 'as reported' literature block.  Writes markdown + json."""
import sys, os, json, glob, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np
import limu_metrics as LM
import lasiesta_cpb_eval as LCE

LIMU = ['CameraParameter', 'Intersection', 'LightSwitch']
WF6 = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees']
ZBS_NAMES = {'ZBS': 'ZBS (CVPR 2023), paper parameters', 'ZBS-p1': 'ZBS, CDnet-script group 1 (sensitivity)',
             'ZBS-p2': 'ZBS, CDnet-script group 2 (sensitivity)', 'ZBS-p3': 'ZBS, CDnet-script group 3 (sensitivity)'}
NAMES = {'SuBSENSE': 'SuBSENSE', 'PAWCS': 'PAWCS', 'LOBSTER': 'LOBSTER', 'ViBe': 'ViBe', 'MixtureOfGaussianV2': 'GMM (MOG2)',
         'KDE': 'KDE', 'PixelBasedAdaptiveSegmenter': 'PBAS'}
OURS = [('CPB', 'CPB (R6, frozen)'), ('guidance', 'DeepLabV3+ alone'), ('framework', 'CPB + DeepLabV3+: Liang framework, our re-impl.'),
        ('framework_blk', 'CPB + DeepLabV3+: Liang framework (block repl.), our re-impl.'), ('MLP', '**CPB + DeepLabV3+: framework + arbiter (ours)**'),
        ('oracle', 'CPB + DeepLabV3+: oracle arbiter (ceiling)')]
OURS_PSP = [('guidance', 'PSPNet alone'), ('framework', 'CPB + PSPNet: Liang framework, our re-impl.'),
            ('framework_blk', 'CPB + PSPNet: Liang framework (block repl.), our re-impl.'), ('MLP', '**CPB + PSPNet: framework + arbiter (ours)**'),
            ('oracle', 'CPB + PSPNet: oracle arbiter (ceiling)')]

SBGS_ROWS = [('SemanticBGS(CPB, deeplab)', 'SemanticBGS rules on CPB + DeepLabV3+'), ('SemanticBGS(SuBSENSE, deeplab)', 'SemanticBGS rules on SuBSENSE + DeepLabV3+'),
             ('SemanticBGS(CPB, pspnet)', 'SemanticBGS rules on CPB + PSPNet'), ('SemanticBGS(SuBSENSE, pspnet)', 'SemanticBGS rules on SuBSENSE + PSPNet')]


def lasiesta_metrics(rows):
    return LCE.seq_metrics([tuple(r) for r in rows])


def fmt(x):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bgs', default=os.path.join(PATHS.RESULTS, 'classical_bgs')); ap.add_argument('--limu', default=os.path.join(PATHS.RESULTS, 'arbiter_deeplab/limu/limu_results.json'))
    ap.add_argument('--wf', default=os.path.join(PATHS.RESULTS, 'arbiter_deeplab/wallflower/wallflower_results.json')); ap.add_argument('--venus', default=os.path.join(PATHS.RESULTS, 'authors_code'))
    ap.add_argument('--las_cpb', default=os.path.join(PATHS.RESULTS, 'cpb_lasiesta_ruleA_counts')); ap.add_argument('--out', default=os.path.join(PATHS.RESULTS, 'comparison'))
    ap.add_argument('--limu_psp', default=os.path.join(PATHS.RESULTS, 'arbiter_pspnet/limu/limu_results.json'))
    ap.add_argument('--limu_fus', default=os.path.join(PATHS.RESULTS, 'fusion_control/limu/limu_results.json'))
    ap.add_argument('--wf_fus', default=os.path.join(PATHS.RESULTS, 'fusion_control/wallflower/wallflower_results.json'))
    ap.add_argument('--las_ext', default=os.path.join(PATHS.RESULTS, 'lasiesta_arbiters'))
    ap.add_argument('--sbgs', default=os.path.join(PATHS.RESULTS, 'semantic_bgs'))
    ap.add_argument('--zbs', default=os.path.join(PATHS.RESULTS, 'zbs'))
    ap.add_argument('--wf_psp', default=os.path.join(PATHS.RESULTS, 'arbiter_pspnet/wallflower/wallflower_results.json'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    B = {}
    for p in glob.glob(os.path.join(a.bgs, '*.json')):
        d = json.load(open(p)); B.setdefault(d['method'], {}).setdefault(d['dataset'], {})[d['sequence']] = d['result']
    for p in glob.glob(os.path.join(a.zbs, '*.json')):
        d = json.load(open(p)); B.setdefault(d['method'], {}).setdefault(d['dataset'], {})[d['sequence']] = d['result']
    md = []; J = {}
    # ---------------- LIMU ----------------
    L = json.load(open(a.limu)); rows = []
    for k, nm in OURS:
        rows.append((nm, [L[s][k]['thesis_fg'] for s in LIMU], np.mean([L[s][k]['thesis_fg'] for s in LIMU]),
                     np.mean([L[s][k]['pooled'] for s in LIMU]), np.nanmean([L[s][k]['fpr_empty'] for s in LIMU])))
    if os.path.exists(a.limu_psp):
        Lp = json.load(open(a.limu_psp))
        for k, nm in OURS_PSP:
            rows.append((nm, [Lp[s][k]['thesis_fg'] for s in LIMU], np.mean([Lp[s][k]['thesis_fg'] for s in LIMU]),
                         np.mean([Lp[s][k]['pooled'] for s in LIMU]), np.nanmean([Lp[s][k]['fpr_empty'] for s in LIMU])))
    sp = os.path.join(a.sbgs, 'sbgs_limu.json')
    if os.path.exists(sp):
        Sb = json.load(open(sp))
        for k, nm in SBGS_ROWS:
            if k in Sb:
                rows.append((nm, [Sb[k][s]['thesis_fg'] for s in LIMU], np.mean([Sb[k][s]['thesis_fg'] for s in LIMU]),
                             np.mean([Sb[k][s]['pooled'] for s in LIMU]), np.nanmean([Sb[k][s]['fpr_empty'] for s in LIMU])))
    if os.path.exists(a.limu_fus):
        Lf = json.load(open(a.limu_fus))
        rows.append(('CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.)', [Lf[s]['MLP']['thesis_fg'] for s in LIMU], np.mean([Lf[s]['MLP']['thesis_fg'] for s in LIMU]),
                     np.mean([Lf[s]['MLP']['pooled'] for s in LIMU]), np.nanmean([Lf[s]['MLP']['fpr_empty'] for s in LIMU])))
    vf = {'CameraParameter': 'CamP_results.json', 'Intersection': 'InterDL_results.json', 'LightSwitch': 'lightswitch_results.json'}
    V = {s: json.load(open(os.path.join(a.venus, f))) for s, f in vf.items() if os.path.exists(os.path.join(a.venus, f))}
    if len(V) == 3:
        v = [V[s]['af3.5_guidance']['final']['native']['thesis_fg'] for s in LIMU]
        rows.insert(4, ("Liang framework, authors' released code", v, np.mean(v),
                        np.mean([V[s]['af3.5_guidance']['final']['native']['pooled'] for s in LIMU]), None))
    for meth, nm in list(NAMES.items()) + list(ZBS_NAMES.items()):
        r = B.get(meth, {}).get('limu', {})
        if len(r) == 3:
            nm = nm if meth in ZBS_NAMES else nm + ' (BGSLibrary)'
            m = {s: LM.metrics([(x[1], x[2], x[3], bool(x[5])) for x in r[s]]) for s in LIMU}
            fe = [np.mean([x[2] / max(x[6], 1) for x in r[s] if not x[5]]) for s in LIMU]
            rows.append((nm, [m[s]['thesis_fg'] for s in LIMU], np.mean([m[s]['thesis_fg'] for s in LIMU]),
                         np.mean([m[s]['pooled'] for s in LIMU]), float(np.mean(fe))))
    md += ['## LIMU (per-frame F; same frames, guidance and metric for every row)', '',
           '| System | CameraParameter | Intersection | LightSwitch | **Mean** | Pooled F (mean) | FPR empty-GT (mean) |', '|---|---|---|---|---|---|---|']
    for nm, v, mean, pooled, fe in rows:
        md.append(f'| {nm} | ' + ' | '.join(fmt(x) for x in v) + f' | **{fmt(mean)}** | {fmt(pooled)} | {fmt(fe)} |')
    md += ['', '*As reported (other protocols / aggregation not specified; reference only):* Liang et al. ICASSP/PR 2021: '
           'CPB + DeepLabV3+ 0.842 / 0.771 / 0.769 (0.793); CPB 0.655 / 0.678 / 0.663 (0.665); SuBSENSE 0.674 / 0.653 / 0.693 (0.675); '
           'KDE 0.646 / 0.648 / 0.675 (0.656); GMM 0.637 / 0.642 / 0.674 (0.652); PBAS 0.658 / 0.641 / 0.459 (0.586).', '']
    J['limu'] = [dict(system=r[0], per_seq=dict(zip(LIMU, map(float, r[1]))), mean=float(r[2])) for r in rows]
    # ---------------- WallFlower ----------------
    Wf = json.load(open(a.wf)); R = Wf['results']; rows = []
    for k, nm in OURS:
        rows.append((nm, [R[s][k]['score'] for s in WF6], R['MovedObject'][k]['score']))
    if os.path.exists(a.wf_psp):
        Rp = json.load(open(a.wf_psp))['results']
        for k, nm in OURS_PSP:
            rows.append((nm, [Rp[s][k]['score'] for s in WF6], Rp['MovedObject'][k]['score']))
    sp = os.path.join(a.sbgs, 'sbgs_wallflower.json')
    if os.path.exists(sp):
        Sb = json.load(open(sp))
        for k, nm in SBGS_ROWS:
            if k in Sb:
                rows.append((nm, [Sb[k][s]['score'] for s in WF6], Sb[k]['MovedObject']['score']))
    if os.path.exists(a.wf_fus):
        Rf = json.load(open(a.wf_fus))['results']
        rows.append(('CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.)', [Rf[s]['MLP']['score'] for s in WF6], Rf['MovedObject']['MLP']['score']))
    if Wf.get('venus'):
        vv = Wf['venus']; rows.insert(4, ("Liang framework, authors' released code (af 3.5)", [vv[s]['final_af3.5'] for s in WF6], vv['MovedObject']['final_af3.5']))
    for meth, nm in list(NAMES.items()) + list(ZBS_NAMES.items()):
        r = B.get(meth, {}).get('wallflower', {})
        if len(r) == 7:
            rows.append((nm if meth in ZBS_NAMES else nm + ' (BGSLibrary)', [r[s]['score'] for s in WF6], r['MovedObject']['score']))
    md += ['## WallFlower (F on the evaluated frame; MovedObject = specificity)', '',
           '| System | Bootstrap | Camouflage | FgAperture | LightSwitch | TimeOfDay | WavingTrees | **Mean F** | MovedObject |', '|---|---|---|---|---|---|---|---|---|']
    for nm, v, mo in rows:
        md.append(f'| {nm} | ' + ' | '.join(fmt(x) for x in v) + f' | **{fmt(np.mean(v))}** | {fmt(mo)} |')
    md += ['', '*As reported (reference only):* PR 2021: CPB 0.671; CPB + DeepLabV3+ 0.720; DeepLabV3+ alone 0.788; CPB + PSPNet 0.805; '
           'PSPNet alone 0.973; SuBSENSE 0.671; KDE 0.683; GMM 0.644; PBAS 0.562. ICPR 2020: CPB + STAM 0.793.', '']
    J['wallflower'] = [dict(system=r[0], per_seq=dict(zip(WF6, map(float, r[1]))), mean=float(np.mean(r[1])), movedobject=float(r[2])) for r in rows]
    # ---------------- LASIESTA ----------------
    seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.las_cpb, '*.csv')))
    cats = LCE.STATIC
    def las_row(nm, getrows):
        per = {}
        for s in seqs:
            r = getrows(s)
            if r is None:
                return None
            per[s] = lasiesta_metrics(r)
        cat = {c: np.mean([per[s]['thesis_fg'] for s in seqs if s.split('_')[1] == c]) for c in cats}
        return (nm, cat, np.mean([per[s]['thesis_fg'] for s in seqs]), np.mean([per[s]['pooled'] for s in seqs]),
                np.nanmean([per[s]['fpr_empty'] for s in seqs]))
    def cpb_rows(s):
        with open(os.path.join(a.las_cpb, s + '.csv'), newline='') as fh:
            return [[int(x) for x in r] for r in list(csv.reader(fh))[1:]]
    rows = [las_row('CPB (R6, frozen, rule A)', cpb_rows)]
    def ext_rows(sub, key):
        def g(s):
            p = os.path.join(a.las_ext, sub, 'seq', s + '.json')
            return json.load(open(p))['rows'][key] if os.path.exists(p) else None
        return g
    if os.path.isdir(os.path.join(a.las_ext, 'deeplab')):
        for k, nm in OURS[1:]:
            rows.append(las_row(nm, ext_rows('deeplab', k)))
    if os.path.isdir(os.path.join(a.las_ext, 'pspnet')):
        for k, nm in OURS_PSP:
            rows.append(las_row(nm, ext_rows('pspnet', k)))
    if os.path.isdir(os.path.join(a.las_ext, 'fusion')):
        rows.append(las_row('CPB + DeepLabV3+: minimal learned fusion (logistic on CPB vote, CPB score, guidance prob.)', ext_rows('fusion', 'MLP')))
    sp = os.path.join(a.sbgs, 'sbgs_lasiesta_rows.json')
    if os.path.exists(sp):
        Sb = json.load(open(sp))
        for k, nm in SBGS_ROWS:
            if k in Sb:
                rows.append(las_row(nm, lambda s, k=k: Sb[k].get(s)))
    for meth, nm in list(NAMES.items()) + list(ZBS_NAMES.items()):
        r = B.get(meth, {}).get('lasiesta', {})
        if len(r) == len(seqs):
            rows.append(las_row(nm if meth in ZBS_NAMES else nm + ' (BGSLibrary)', lambda s, r=r: r[s]))
    rows = [x for x in rows if x]
    md += [f'## LASIESTA (rule A; per-frame F per category = mean of its 2 sequences; {len(seqs)} sequences)', '',
           '| System | ' + ' | '.join(cats) + ' | **Mean** | Pooled F | FPR empty-GT |', '|---|' + '---|' * (len(cats) + 3)]
    for nm, cat, mean, pooled, fe in rows:
        md.append(f'| {nm} | ' + ' | '.join(fmt(cat[c]) for c in cats) + f' | **{fmt(mean)}** | {fmt(pooled)} | {fmt(fe)} |')
    md += ['',
           '*As reported (other protocols; reference only), per category SI CA OC IL MB BS CL RA SN SU, overall:* '
           'BSUV-Net 2.0 (CDnet-trained, supervised) 0.92 0.68 0.96 0.88 0.81 0.77 0.93 0.94 0.84 0.79 (0.85); '
           'Cuevas 0.88 0.84 0.78 0.65 0.93 0.66 0.93 0.87 0.78 0.72 (0.81); Haines 0.89 0.89 0.92 0.85 0.84 0.68 0.83 0.89 0.17 0.86 (0.78); '
           'Maddalena 0.95 0.86 0.95 0.21 0.91 0.40 0.97 0.90 0.81 0.88 (0.78).', '']
    J['lasiesta'] = [dict(system=r[0], per_category={c: float(r[1][c]) for c in cats}, mean=float(r[2]), pooled=float(r[3])) for r in rows]
    open(os.path.join(a.out, 'COMPARISON.md'), 'w').write('\n'.join(md))
    json.dump(J, open(os.path.join(a.out, 'comparison.json'), 'w'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
