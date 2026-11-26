#!/usr/bin/env python3
"""
Apply the FROZEN CDnet-trained MLP arbiter to an external dataset (LIMU now; WallFlower later).
Nothing is fitted or tuned here: frozen locked CPB (trained on the dataset's own training frames, no GT),
frozen DeepLab guidance cache, frozen arbiter + its validation threshold.

LIMU protocol: CPB trained on frames 0-399, processed at 160x120 with frame switching; every GT frame (every 15
frames from 500) scored at native 320x240. Main metric = per-frame F (Zhou thesis: mean P and R over GT frames
with foreground). Also pooled F, thesis_all (precision also over empty-GT frames) and the false-positive rate on
frames whose GT is empty.

Systems scored on the same frames, from the same model state:
  CPB alone | guidance alone | CPB stage 1 | framework (paper stage 2 + stage 3) | framework (block-repl.)
  | MLP arbiter | oracle arbiter (GT on disputed pixels; ceiling)
Example: python -u apply_arbiter.py limu --limu <LIMU folder> --guid <limu guidance cache> --model arbiter_runs/models/mlp_ALL+noA_final.npz
"""
import sys, os, csv, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import arbiter_mlp as AM
import limu_framework as LF
import limu_eval as L
import limu_metrics as LM
import wf_frame_switch as S
import wallflower_eval as W

P = AM.P
SYS = ['CPB', 'guidance', 'stage1', 'framework', 'framework_blk', 'MLP', 'oracle']
PAPER = {'CPB': {'CameraParameter': 0.6545, 'Intersection': 0.6778, 'LightSwitch': 0.6633},
         'CPB+DeepLabV3+': {'CameraParameter': 0.8419, 'Intersection': 0.7707, 'LightSwitch': 0.7688},
         'DeepLabV3+': {'CameraParameter': 0.6965, 'Intersection': 0.5624, 'LightSwitch': 0.2555}}


def limu_args():
    a = argparse.Namespace(ntrain=400, alpha=0.0, alpha_sig=0.0, sigema=0.8, trigger=0.7, state_diff=25.0, state_frac=0.5,
                           min_cluster=50, max_train=200, af=2.5, floor=3.0, bth=7.0, ref=20, median=1)
    a.robust_sigma, a.block, a.selection, a.gamma, a.robust_B, a.sig_all = 'mad', 'median', 'minstd', 'matlab', -1, 0
    a.trainer, a.robust, a.frame_guard, a.switch = 'consistent', 1, 0.5, True
    return a


def counts(m, gt):
    return (int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum()), bool(gt.any()))


def run_limu_seq(seq, a, arb, args):
    net, cols, mu, sd, thr, meta = arb
    t0 = time.time(); sq = L.Seq(args.limu, seq)
    A = S.RGBModel([sq.rgb(n) for n in range(a.ntrain)], a, 'A (frames 0-399)'); models = [A]
    other, info = LF.find_states_limu(sq, a); P('  ' + info)
    if other is not None:
        sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
        models.append(S.RGBModel([sq.rgb(n) for n in sel], a, 'B (other state)'))
    z = np.load(os.path.join(args.guid, f'{seq}.npz')); gidx = {int(n): i for i, n in enumerate(z['nums'])}
    GU, PR = z['guid'], z['prob']
    pscale = 255.0 if PR.dtype == np.uint8 else 1.0                    # LIMU cache: uint8 = round(255 * softmax)
    W, H = L.WORK; work = (H, W)
    seqn = range(a.ntrain, sq.N) if len(models) > 1 else sorted(sq.gts)
    rows = {k: [] for k in SYS}; frm = []; frm_bd = []; active = 0; switches = 0
    bar = AM.Bar(len(seqn), seq)
    for n in seqn:
        if n not in sq.frames:
            bar.update(); continue
        rgb = sq.rgb(n)
        if len(models) > 1:
            cpb = models[active].detect(rgb, work, a)
            if np.mean(cpb > 127) > a.trigger:
                fr = [float(np.mean(mm.detect(rgb, work, a) > 127)) for mm in models]; b = int(np.argmin(fr))
                if b != active:
                    active = b; switches += 1
        if n not in sq.gts or n not in gidx:
            bar.update(); continue
        M = models[active]
        gt = sq.gt(n)                                                   # native bool
        gw = cv2.resize(gt.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        gn = GU[gidx[n]] > 127
        gfg = cv2.resize(gn.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        gp = cv2.resize(PR[gidx[n]].astype(np.float32) / pscale, (W, H), interpolation=cv2.INTER_LINEAR)
        npv = n - args.guid_prev_step                                   # CDnet cache had guidance every 2nd frame
        gpp = (cv2.resize(PR[gidx[npv]].astype(np.float32) / pscale, (W, H), interpolation=cv2.INTER_LINEAR)
               if npv in gidx else gp)
        rgbp = sq.rgb(n - 1) if (n - 1) in sq.frames else rgb
        inf, rws = AM.frame_features(M, rgb, rgbp, gfg, gp, gpp, gw, a)
        s1 = inf['s1'] > 0
        mlp = s1.copy(); orc = s1.copy(); pfg = None; nd = 0; to_guid = 0.0; acc_d = {}
        if rws is not None:
            X = (rws['X'].astype(np.float32)[:, cols] - mu) / sd
            p = net.predict(X); dec = p >= thr; ys, xs = rws['ys'], rws['xs']
            mlp[ys, xs] = dec; orc[ys, xs] = rws['y'] > 0
            nd = len(ys); guid_lab = rws['guid'] > 0; to_guid = float((dec == guid_lab).mean())
            y = rws['y'] > 0
            acc_d = dict(mlp=float((dec == y).mean()), cpb=float(((rws['cpb'] > 0) == y).mean()), guid=float((guid_lab == y).mean()))
            frm_feats = rws['X'].astype(np.float32).mean(0)
        else:
            frm_feats = None
        up = lambda m: cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        masks = {'CPB': up(M.detect(rgb, work, a) > 127), 'guidance': gn, 'stage1': up(s1), 'framework': up(inf['s3p'] > 0),
                 'framework_blk': up(inf['s3b'] > 0), 'MLP': up(mlp), 'oracle': up(orc)}
        for k in SYS:
            rows[k].append(counts(masks[k], gt))
        if args.breakdown:
            gfb = gfg > 0
            reg = {'agree': up(s1 == gfb), 'A_cpbfg': up(s1 & ~gfb), 'B_guidfg': up(~s1 & gfb)}
            band = cv2.dilate(gt.astype(np.uint8), np.ones((5, 5), np.uint8)) != cv2.erode(gt.astype(np.uint8), np.ones((5, 5), np.uint8))
            bd = {}
            for rn, R in reg.items():
                for sn in ('MLP', 'oracle', 'framework'):
                    m = masks[sn]
                    bd[f'{rn}|{sn}'] = (int((m & gt & R).sum()), int((m & ~gt & R).sum()), int((~m & gt & R).sum()))
            err = (masks['MLP'] != gt) & ~reg['agree']
            bd['disp_err_px'] = int(err.sum()); bd['disp_err_band'] = int((err & band).sum())
            gtw = cv2.resize(gw, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            bd['res_ceiling'] = counts(gtw, gt)[:3]
            if rws is not None:
                y = rws['y'] > 0; e = dec != y; near = np.abs(p - thr) < 0.15
                bd['conf'] = (int(e.sum()), int((e & near).sum()), int((e & ((p > 0.9) | (p < 0.1))).sum()), len(y),
                              int((e & (rws['cpb'] > 0)).sum()), int((e & (rws['cpb'] == 0)).sum()))
                bd['err_feat'] = rws['X'][e].astype(np.float32).mean(0) if e.any() else None
                bd['ok_feat'] = rws['X'][~e].astype(np.float32).mean(0) if (~e).any() else None
            frm_bd.append(bd)
        frm.append(dict(bd=(frm_bd[-1] if args.breakdown else None), n=n, luma=float(AM.luma(rgb).mean()), disputes=nd, to_guid=to_guid, acc=acc_d, gt_fg=int(gt.sum()),
                        feats=frm_feats, guid_frac=float(gn.mean()), cpb_frac=float(s1.mean())))
        bar.update()
    bar.close()
    res = {k: LM.metrics(rows[k]) for k in SYS}
    for k in SYS:                                                       # FPR on empty-GT frames
        e = [r for r in rows[k] if not r[3]]
        res[k]['fpr_empty'] = float(np.mean([r[1] / (240 * 320) for r in e])) if e else float('nan')
        res[k]['n_empty'] = len(e)
    P(f'  {seq}: {len(frm)} GT frames ({sum(1 for f in frm if f["gt_fg"] > 0)} with foreground), models {len(models)}, '
      f'switches {switches}, {time.time() - t0:.0f}s')
    return res, rows, frm


def light_states(frm):
    l = np.array([f['luma'] for f in frm]); lo, hi = l.min(), l.max()
    thr = (lo + hi) / 2.0
    return l < thr, thr


def _thesis(rows):
    return LM.metrics(rows)['thesis_fg'] if rows else float('nan')


def breakdown_report(R, ROWS, FRM, seqs, meta, thr):
    """Where does the MLP lose per-frame F? Counterfactuals: replace the MLP's counts in one region by the oracle's."""
    P('\n' + '=' * 100 + '\n  ERROR BREAKDOWN of the MLP arbiter (per-frame F gained if the errors of one kind were removed)\n' + '=' * 100)
    P('  regions: agree = CPB and guidance agree (no arbiter can change it) | A = CPB fg / guidance bg | B = CPB bg / guidance fg')
    hdr = f'  {"counterfactual":44s}' + ''.join(f'{s[:14]:>16s}' for s in seqs) + f'{"MEAN":>9s}'
    P(hdr)
    def scen(s, fix):
        out = []
        for r, f in zip(ROWS[s]['MLP'], FRM[s]):
            tp, fp, fn, hf = r; bd = f['bd']
            for reg, what in fix:
                m = bd[f'{reg}|MLP']; o = bd[f'{reg}|oracle']
                if what == 'all':
                    tp, fp, fn = tp - m[0] + o[0], fp - m[1] + o[1], fn - m[2] + o[2]
                elif what == 'fp':
                    fp = fp - m[1] + o[1]
                elif what == 'fn':
                    tp, fn = tp - m[0] + o[0], fn - m[2] + o[2]
                elif what == 'perfect':                      # agreement region perfect too
                    tp, fp, fn = tp + m[2], fp - m[1], fn - m[2]
            out.append((tp, fp, fn, hf))
        return _thesis(out)
    rows_ = [('MLP as is', []), ('fix A (CPB-fg disputes) false alarms', [('A_cpbfg', 'fp')]),
             ('fix A misses (CPB right, MLP said bg)', [('A_cpbfg', 'fn')]),
             ('fix B (guidance-fg disputes) false alarms', [('B_guidfg', 'fp')]),
             ('fix B misses (guidance right, MLP said bg)', [('B_guidfg', 'fn')]),
             ('fix all disputes (= oracle)', [('A_cpbfg', 'all'), ('B_guidfg', 'all')]),
             ('oracle + agreement false alarms fixed', [('A_cpbfg', 'all'), ('B_guidfg', 'all'), ('agree', 'fp')]),
             ('oracle + agreement misses fixed', [('A_cpbfg', 'all'), ('B_guidfg', 'all'), ('agree', 'fn')])]
    base = {s: scen(s, []) for s in seqs}
    for nm, fix in rows_:
        v = [scen(s, fix) for s in seqs]
        d = '' if not fix else '   (gain ' + ', '.join(f'{x - base[s]:+.3f}' for x, s in zip(v, seqs)) + ')'
        P(f'  {nm:44s}' + ''.join(f'{x:16.4f}' for x in v) + f'{np.mean(v):9.4f}' + d)
    v = []
    for s in seqs:
        rr = [tuple(f['bd']['res_ceiling']) + (r[3],) for r, f in zip(ROWS[s]['MLP'], FRM[s])]
        v.append(_thesis(rr))
    P(f'  {"resolution ceiling (GT at 160x120, upsampled)":44s}' + ''.join(f'{x:16.4f}' for x in v) + f'{np.mean(v):9.4f}')
    P('\n  MLP errors on disputed pixels (working resolution)')
    P(f'  {"sequence":16s} {"error rate":>10s} {"near thr (+-0.15)":>18s} {"confident (p<.1/>.9)":>21s} {"errors on A (CPB fg)":>21s} {"on B (guid fg)":>15s} {"in GT boundary band":>20s}')
    for s in seqs:
        c = np.array([f['bd']['conf'] for f in FRM[s] if 'conf' in f['bd']]).sum(0)
        band = sum(f['bd']['disp_err_band'] for f in FRM[s]); tot = max(sum(f['bd']['disp_err_px'] for f in FRM[s]), 1)
        P(f'  {s:16s} {c[0] / c[3]:10.3f} {c[1] / max(c[0], 1):18.2f} {c[2] / max(c[0], 1):21.2f} {c[4] / max(c[0], 1):21.2f} {c[5] / max(c[0], 1):15.2f} {band / tot:20.2f}')
    P('\n  Mean feature value on MLP errors vs correct decisions (disputed pixels; raw units)')
    feats = AM.FEATURES
    sel = ['cpb_says_fg', 'cpb_score', 'resid_z', 'guid_prob', 'guid_bdist', 'cpb_occ', 'disagree_occ', 'frame_cpb_frac', 'frame_guid_frac', 'frame_dis_frac', 'frame_luma', 'motion']
    P(f'  {"feature":16s}' + ''.join(f'{s[:10] + " err":>15s}{s[:10] + " ok":>14s}' for s in seqs))
    for fn_ in sel:
        j = feats.index(fn_); line = f'  {fn_:16s}'
        for s in seqs:
            e = [f['bd']['err_feat'][j] for f in FRM[s] if f['bd'].get('err_feat') is not None]
            o = [f['bd']['ok_feat'][j] for f in FRM[s] if f['bd'].get('ok_feat') is not None]
            line += f'{np.mean(e):15.3f}{np.mean(o):14.3f}'
        P(line)
    P('\n  Frames that cost the MLP the most per-frame F (largest F gap to the oracle, frames with foreground)')
    for s in seqs:
        g = []
        for r, ro, f in zip(ROWS[s]['MLP'], ROWS[s]['oracle'], FRM[s]):
            if r[3]:
                fm = 2 * r[0] / max(2 * r[0] + r[1] + r[2], 1); fo = 2 * ro[0] / max(2 * ro[0] + ro[1] + ro[2], 1)
                g.append((fo - fm, f['n'], fm, fo, f['luma'], f['disputes']))
        g.sort(reverse=True)
        P(f'  {s}: ' + '; '.join(f'frame {n} F {fm:.2f}->{fo:.2f} (luma {l:.0f}, {d} disputes)' for _, n, fm, fo, l, d in g[:5]))


def cmd_limu(args):
    os.makedirs(args.out, exist_ok=True); AM.tee(os.path.join(args.out, 'limu_log.txt'))
    arb = AM.load_mlp(args.model, args.json)
    net, cols, mu, sd, thr, meta = arb
    a = limu_args(); L.WORK = (160, 120)
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} LIMU external test  model={os.path.basename(args.model)} '
      f'({len(meta["features"])} features, threshold {thr})  frozen CPB + frozen {args.guid_name} cache + frozen arbiter')
    R, ROWS, FRM = {}, {}, {}
    for seq in LF.SEQS:
        R[seq], ROWS[seq], FRM[seq] = run_limu_seq(seq, a, arb, args)
    seqs = LF.SEQS
    for agg, title in (('thesis_fg', 'PER-FRAME F (main metric, Zhou thesis)'), ('pooled', 'POOLED F (supplementary)'),
                       ('thesis_all', 'per-frame F with precision also on empty-GT frames'), ('fpr_empty', 'FALSE-POSITIVE RATE on empty-GT frames (lower = better)')):
        P(f'\n  {title}')
        P(f'  {"system":16s}' + ''.join(f'{s[:14]:>16s}' for s in seqs) + f'{"MEAN":>9s}')
        for k in SYS:
            v = [R[s][k][agg] for s in seqs]
            P(f'  {k:16s}' + ''.join(f'{x:16.4f}' for x in v) + f'{np.mean(v):9.4f}')
        if agg == 'thesis_fg':
            for nm, d in PAPER.items():
                v = [d[s] for s in seqs]
                P(f'  {"paper " + nm:16s}' + ''.join(f'{x:16.4f}' for x in v) + f'{np.mean(v):9.4f}')
    P(f'\n  P and R behind the per-frame F (mean over frames with foreground):')
    for k in ('CPB', 'guidance', 'framework', 'MLP', 'oracle'):
        P(f'  {k:16s}' + ''.join(f'   P {R[s][k]["P_fg"]:.3f} R {R[s][k]["R_fg"]:.3f}' for s in seqs))
    P('\n  ARBITER BEHAVIOUR on disputed pixels (mean over GT frames)')
    P(f'  {"sequence":16s} {"disputes/frame":>15s} {"sided w/ guidance":>18s} {"acc MLP":>8s} {"acc CPB":>8s} {"acc guid":>9s}')
    for s in seqs:
        f = [x for x in FRM[s] if x['disputes'] > 0]
        P(f'  {s:16s} {np.mean([x["disputes"] for x in FRM[s]]):15.0f} {np.mean([x["to_guid"] for x in f]):18.2f} '
          f'{np.mean([x["acc"]["mlp"] for x in f]):8.3f} {np.mean([x["acc"]["cpb"] for x in f]):8.3f} {np.mean([x["acc"]["guid"] for x in f]):9.3f}')
    P('\n  FEATURE SHIFT vs the CDnet training data (mean on LIMU disputes, in training standard deviations)')
    P(f'  {"feature":16s}' + ''.join(f'{s[:14]:>16s}' for s in seqs))
    zs = {}
    for s in seqs:
        F_ = np.array([x['feats'] for x in FRM[s] if x['feats'] is not None]).mean(0)
        zs[s] = (F_[cols] - mu) / sd
    for j, f in enumerate(meta['features']):
        v = [zs[s][j] for s in seqs]
        flag = '  <-- shifted' if max(abs(x) for x in v) > 1.0 else ''
        P(f'  {f:16s}' + ''.join(f'{x:16.2f}' for x in v) + flag)
    # LightSwitch: light on / off
    f = FRM['LightSwitch']; dark, lthr = light_states(f)
    P(f'\n  LIGHTSWITCH by lighting state (frame brightness threshold {lthr:.0f}): dark frames {int(dark.sum())}, bright {int((~dark).sum())}')
    P(f'  {"system":16s} {"dark: perframe F":>17s} {"P":>6s} {"R":>6s} {"FPR empty":>10s} | {"bright: perframe F":>19s} {"P":>6s} {"R":>6s} {"FPR empty":>10s}')
    for k in SYS:
        out = []
        for msk in (dark, ~dark):
            rr = [r for r, m in zip(ROWS['LightSwitch'][k], msk) if m]
            m_ = LM.metrics(rr) if rr else None
            e = [r for r in rr if not r[3]]
            fe = float(np.mean([r[1] / (240 * 320) for r in e])) if e else float('nan')
            out.append((m_['thesis_fg'], m_['P_fg'], m_['R_fg'], fe) if m_ else (np.nan,) * 4)
        P(f'  {k:16s} {out[0][0]:17.4f} {out[0][1]:6.3f} {out[0][2]:6.3f} {out[0][3]:10.4f} | {out[1][0]:19.4f} {out[1][1]:6.3f} {out[1][2]:6.3f} {out[1][3]:10.4f}')
    P(f'  arbiter on LightSwitch disputes: sided with guidance {np.mean([x["to_guid"] for x, d in zip(f, dark) if d and x["disputes"]]):.2f} (dark) / '
      f'{np.mean([x["to_guid"] for x, d in zip(f, dark) if not d and x["disputes"]]):.2f} (bright)')
    for s in seqs:
        with open(os.path.join(args.out, f'limu_{s}_perframe.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['frame', 'luma', 'disputes', 'sided_with_guidance'] + [f'{k}_{c}' for k in SYS for c in ('tp', 'fp', 'fn', 'has_fg')])
            for i, x in enumerate(FRM[s]):
                w.writerow([x['n'], f'{x["luma"]:.1f}', x['disputes'], f'{x["to_guid"]:.3f}'] + [v for k in SYS for v in ROWS[s][k][i]])
    AM.save_json(os.path.join(args.out, 'limu_results.json'), {s: {k: R[s][k] for k in SYS} for s in seqs})
    if args.breakdown:
        breakdown_report(R, ROWS, FRM, seqs, meta, thr)
    P(f'\n  -> {args.out}/limu_results.json, limu_<seq>_perframe.csv, limu_log.txt')


WF_SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']
WF_PAPER = {'CPB': [0.6518, 0.6112, 0.5900, 0.7157, 0.7564, 0.7033, 0.8922],
            'CPB+DeepLabV3+ (PR 2021)': [0.5927, 0.9867, 0.9237, 0.6886, 0.2604, 0.8682, 0.9838],
            'DeepLabV3+ (PR 2021)': [0.8327, 0.9941, 0.9884, 0.7375, 0.1909, 0.9850, 1.0],
            'CPB+PSPNet (PR 2021)': [0.5341, 0.9849, 0.9204, 0.7048, 0.8147, 0.8696, 0.9836],
            'CPB+STAM (ICPR 2020)': [0.7560, 0.6884, 0.9402, 0.9097, 0.7949, 0.6665, 0.9977]}


def cmd_wallflower(args):
    os.makedirs(args.out, exist_ok=True); AM.tee(os.path.join(args.out, 'wallflower_log.txt'))
    net, cols, mu, sd, thr, meta = AM.load_mlp(args.model, args.json)
    a = limu_args(); a.max_train = 200
    W.WF = args.wf
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} WALLFLOWER external test  model={os.path.basename(args.model)} '
      f'({len(meta["features"])} features, threshold {thr})  frozen CPB (official Train window) + frozen {args.guid_name} cache + frozen arbiter')
    R, BD, INFO = {}, {}, {}
    for seq in WF_SEQS:
        t0 = time.time(); d = os.path.join(args.wf, seq)
        tr, run, ev = W.script_windows(d)
        gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
        gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE); spec = seq == 'MovedObject'
        A = S.RGBModel([S.load_rgb(d, n) for n in tr], a, f'A (Train {tr[0]}-{tr[-1]})'); models = [A]
        other, info = S.find_states(d, tr[-1], a)
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
            models.append(S.RGBModel([S.load_rgb(d, n) for n in sel], a, 'B (other state)'))
        active = 0; switches = 0
        for n in run + [ev]:                                      # frozen: only the switching state evolves
            if len(models) > 1:
                rgb = S.load_rgb(d, n)
                if np.mean(models[active].detect(rgb, gt.shape, a) > 127) > a.trigger:
                    fr = [float(np.mean(mm.detect(rgb, gt.shape, a) > 127)) for mm in models]; b = int(np.argmin(fr))
                    if b != active:
                        active = b; switches += 1
        M = models[active]
        z = np.load(os.path.join(args.guid, f'{seq}.npz')); gidx = {int(n): i for i, n in enumerate(z['nums'])}
        pscale = 255.0 if z['prob'].dtype == np.uint8 else 1.0
        rs = lambda x, it: cv2.resize(x, (gt.shape[1], gt.shape[0]), interpolation=it) if x.shape[:2] != gt.shape else x
        gfg = rs((z['guid'][gidx[ev]] > 127).astype(np.uint8), cv2.INTER_NEAREST)
        gp = rs(z['prob'][gidx[ev]].astype(np.float32) / pscale, cv2.INTER_LINEAR)
        npv = ev - args.guid_prev_step
        gpp = rs(z['prob'][gidx[npv]].astype(np.float32) / pscale, cv2.INTER_LINEAR) if npv in gidx else gp
        rgb = S.load_rgb(d, ev); rgbp = S.load_rgb(d, ev - 1)
        gw = np.where(gt > 127, 255, 0).astype(np.uint8)
        inf, rws = AM.frame_features(M, rgb, rgbp, gfg, gp, gpp, gw, a)
        s1 = inf['s1'] > 0; mlp = s1.copy(); orc = s1.copy(); bd = {}
        if rws is not None:
            p = net.predict((rws['X'].astype(np.float32)[:, cols] - mu) / sd); dec = p >= thr; ys, xs = rws['ys'], rws['xs']
            mlp[ys, xs] = dec; orc[ys, xs] = rws['y'] > 0
            y = rws['y'] > 0; cpb_d = rws['cpb'] > 0; gd = rws['guid'] > 0
            bd = dict(n=len(y), to_guid=float((dec == gd).mean()), acc_mlp=float((dec == y).mean()),
                      acc_cpb=float((cpb_d == y).mean()), acc_guid=float((gd == y).mean()), gt_fg_share=float(y.mean()),
                      A=int(cpb_d.sum()), B=int((~cpb_d).sum()),
                      A_fp=int((dec & ~y & cpb_d).sum()), A_fn=int((~dec & y & cpb_d).sum()),
                      B_fp=int((dec & ~y & ~cpb_d).sum()), B_fn=int((~dec & y & ~cpb_d).sum()),
                      near=float((np.abs(p - thr) < 0.15)[dec != y].mean()) if (dec != y).any() else float('nan'))
        masks = {'CPB': M.detect(rgb, gt.shape, a) > 127, 'guidance': gfg > 0, 'stage1': s1, 'framework': inf['s3p'] > 0,
                 'framework_blk': inf['s3b'] > 0, 'MLP': mlp, 'oracle': orc}
        R[seq] = {}
        for k in SYS:
            sc, dd = W.score(masks[k].astype(np.uint8) * 255, gt, spec)
            R[seq][k] = dict(score=sc, P=dd['prec'], R=dd['rec'], fp=dd['fp'], fn=dd['fn'], tp=dd['tp'])
        # exact counterfactual breakdown (decisions and scoring at the same 160x120 resolution)
        if rws is not None and not spec:
            gtb = gt > 127; agree = s1 == (gfg > 0)
            def f_of(m):
                return W.score(m.astype(np.uint8) * 255, gt, False)[0]
            cf = {}
            for nm, sel_ in (('fix A false alarms', cpb_d & dec & ~y), ('fix A misses', cpb_d & ~dec & y),
                             ('fix B false alarms', ~cpb_d & dec & ~y), ('fix B misses', ~cpb_d & ~dec & y)):
                m2 = mlp.copy(); m2[ys[sel_], xs[sel_]] = y[sel_]; cf[nm] = f_of(m2) - R[seq]['MLP']['score']
            m3 = orc.copy(); m3[agree & ~gtb] = False; cf['agreement false alarms (beyond oracle)'] = f_of(m3) - R[seq]['oracle']['score']
            m4 = orc.copy(); m4[agree & gtb] = True; cf['agreement misses (beyond oracle)'] = f_of(m4) - R[seq]['oracle']['score']
            bd['cf'] = cf
        BD[seq] = bd; INFO[seq] = dict(models=len(models), switches=switches, frame_luma=float(AM.luma(rgb).mean()),
                                       frame_cpb_frac=float(s1.mean()), frame_guid_frac=float((gfg > 0).mean()))
        P(f'  {seq:18s} models {len(models)}, switches {switches}, disputes {bd.get("n", 0)}, '
          f'{"spec" if spec else "F"} MLP {R[seq]["MLP"]["score"]:.4f}  ({time.time() - t0:.0f}s)')
    six = WF_SEQS[:6]
    venus = None
    if args.venus and os.path.exists(args.venus):
        venus = json.load(open(args.venus))
    P('\n' + '=' * 118)
    P('  WALLFLOWER: F on the evaluated frame (6 sequences); MovedObject = specificity')
    P('=' * 118)
    P(f'  {"system":30s}' + ''.join(f'{s[:10]:>11s}' for s in six) + f'{"MEAN F":>9s}{"MovedObj":>10s}{"wins":>6s}')
    best_fixed = {s: max(R[s][k]['score'] for k in ('CPB', 'guidance', 'stage1', 'framework', 'framework_blk')) for s in six}
    rowsets = [(k, [R[s][k]['score'] for s in six], R['MovedObject'][k]['score']) for k in SYS]
    if venus:
        for af in ('3.5', '2.5'):
            rowsets.insert(-1, (f"authors' code (af {af})", [venus[s][f'final_af{af}'] for s in six], venus['MovedObject'][f'final_af{af}']))
        rowsets.insert(-1, ("authors' code CPB (af 3.5)", [venus[s]['cpb_af3.5'] for s in six], venus['MovedObject']['cpb_af3.5']))
    for k, v, mo in rowsets:
        w = sum(1 for s, x in zip(six, v) if x >= best_fixed[s] - 1e-9) if k == 'MLP' else ''
        P(f'  {k:30s}' + ''.join(f'{x:11.4f}' for x in v) + f'{np.mean(v):9.4f}{mo:10.4f}{str(w):>6s}')
    P('  ' + '-' * 114 + '\n  as reported (other implementations / guidance; for reference only):')
    for k, v in WF_PAPER.items():
        P(f'  {k:30s}' + ''.join(f'{x:11.4f}' for x in v[:6]) + f'{np.mean(v[:6]):9.4f}{v[6]:10.4f}')
    P('  "wins" = sequences where the MLP is >= the best fixed system (CPB, guidance, stage 1, both frameworks)')
    P('\n  P / R on the evaluated frame:')
    for k in ('CPB', 'guidance', 'framework', 'MLP', 'oracle'):
        P(f'  {k:16s}' + ''.join(f'  {s[:8]:>8s} P {R[s][k]["P"]:.2f} R {R[s][k]["R"]:.2f}' for s in six))
    P('\n  ARBITER BEHAVIOUR on disputed pixels of the evaluated frame')
    P(f'  {"sequence":18s} {"disputes":>9s} {"GT-fg share":>11s} {"sided w/ guid":>13s} {"acc MLP":>8s} {"acc CPB":>8s} {"acc guid":>9s} '
      f'{"A fp/fn":>12s} {"B fp/fn":>12s} {"errors near thr":>16s} {"frame CPB frac":>15s} {"luma":>6s}')
    for s in WF_SEQS:
        b = BD[s]
        if not b:
            P(f'  {s:18s} {0:9d}'); continue
        P(f'  {s:18s} {b["n"]:9d} {b["gt_fg_share"]:11.2f} {b["to_guid"]:13.2f} {b["acc_mlp"]:8.3f} {b["acc_cpb"]:8.3f} {b["acc_guid"]:9.3f} '
          f'{b["A_fp"]:6d}/{b["A_fn"]:<5d} {b["B_fp"]:6d}/{b["B_fn"]:<5d} {b["near"]:16.2f} {INFO[s]["frame_cpb_frac"]:15.3f} {INFO[s]["frame_luma"]:6.0f}')
    P('\n  EXACT BREAKDOWN (F gained by fixing one kind of MLP error; decisions and scoring at the same resolution)')
    keys = ['fix A false alarms', 'fix A misses', 'fix B false alarms', 'fix B misses',
            'agreement false alarms (beyond oracle)', 'agreement misses (beyond oracle)']
    P(f'  {"":40s}' + ''.join(f'{s[:10]:>11s}' for s in six))
    for k in keys:
        P(f'  {k:40s}' + ''.join(f'{BD[s]["cf"][k]:+11.4f}' if BD[s] and "cf" in BD[s] else f'{"-":>11s}' for s in six))
    AM.save_json(os.path.join(args.out, 'wallflower_results.json'), dict(results=R, breakdown=BD, info=INFO, venus=venus))
    P(f'\n  -> {args.out}/wallflower_results.json, wallflower_log.txt')


def cmd_lasiesta(args):
    """LASIESTA, rule A (main protocol): CPB trained on the first N = min(200, floor(0.4 L)) frames (the saved
    lasiesta_cpb_eval.py models are reused, or trained identically if missing), frames N+1..L scored at 352x288 with
    the grey 'uncertain' GT pixels ignored. Guidance cache: every frame, native size. Nothing is tuned."""
    import lasiesta_cpb_eval as LCE, cdnet_cpb_eval as E
    os.makedirs(os.path.join(args.out, 'seq'), exist_ok=True); AM.tee(os.path.join(args.out, 'lasiesta_log.txt'))
    net, cols, mu, sd, thr, meta = AM.load_mlp(args.model, args.json)
    L = E.locked_args(0.0); L.sigema = 0.8                                # same CPB / framework settings as the CDnet extraction
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} LASIESTA external test (rule A)  model={os.path.basename(args.model)} '
      f'({len(meta["features"])} features, threshold {thr})  frozen CPB + frozen {args.guid_name} cache + frozen arbiter')
    seqs = LCE.find_sequences(args.lasiesta)
    if args.only:
        seqs = {k: v for k, v in seqs.items() if k in args.only.split(',')}
    if not seqs:
        raise SystemExit(f'no LASIESTA sequence found under {args.lasiesta}')
    for k, (sid, (d, gtd)) in enumerate(seqs.items(), 1):
        outp = os.path.join(args.out, 'seq', f'{sid}.json')
        if os.path.exists(outp):
            P(f'[{k}/{len(seqs)}] {sid}: done -> skip'); continue
        gpath = os.path.join(args.guid, f'{sid}.npz')
        if not os.path.exists(gpath):
            P(f'[{k}/{len(seqs)}] {sid}: NO guidance cache {gpath} -> skip'); continue
        t0 = time.time()
        frames = sorted(glob.glob(os.path.join(d, '*.bmp')), key=LCE.num)
        gts = {LCE.num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
        Lf = len(frames); H, W = cv2.imread(frames[0]).shape[:2]; ww, wh = E.work_size(W, H, args.budget)
        rd = lambda p: cv2.cvtColor(cv2.resize(cv2.imread(p), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
        tr, sc, N = LCE.window(Lf, 'A', 200)
        ca = argparse.Namespace(out=args.cpb_out, budget=args.budget, max_train=200, rd=rd)
        models, cmeta, loaded = LCE.train_or_load(sid, 'A', ca, frames, tr, ww, wh)
        z = np.load(gpath); gidx = {int(n): i for i, n in enumerate(z['nums'])}; GU, PR = z['guid'], z['prob']
        pscale = 255.0 if PR.dtype == np.uint8 else 1.0
        P(f'[{k}/{len(seqs)}] {sid}  {W}x{H} -> {ww}x{wh}  {Lf} frames  train {N}  score {Lf - N}  CPB models {len(models)} '
          f'({"loaded" if loaded else "trained"})  guidance frames {len(gidx)}')
        wsz = lambda x, it: cv2.resize(x, (ww, wh), interpolation=it)
        rows = {s_: [] for s_ in SYS}; disp = []; active = 0; switches = 0
        bar = AM.Bar(Lf - N, sid)
        for i in range(tr[-1] + 1, Lf):
            n = LCE.num(frames[i]); rgb = rd(frames[i])
            m = models[active].detect(rgb, (wh, ww), L)
            if len(models) > 1 and np.mean(m > 127) > L.trigger:
                fr = [float(np.mean(mm.detect(rgb, (wh, ww), L) > 127)) for mm in models]; b = int(np.argmin(fr))
                if b != active:
                    active, switches = b, switches + 1
                    m = models[active].detect(rgb, (wh, ww), L)
            fg, valid = LCE.read_gt(gts.get(n, ''))
            if fg is None or n not in gidx:
                bar.update(); continue
            gn = GU[gidx[n]] > 127
            if gn.shape != fg.shape:
                gn = cv2.resize(gn.astype(np.uint8), (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            gw = np.where(fg, 255, 0).astype(np.uint8); gw[~valid] = 85              # grey = ignored (CDnet 'unknown' code)
            gw = wsz(gw, cv2.INTER_NEAREST)
            gfg = wsz(gn.astype(np.uint8), cv2.INTER_NEAREST)
            gp = wsz(PR[gidx[n]].astype(np.float32) / pscale, cv2.INTER_LINEAR)
            npv = n - args.guid_prev_step
            gpp = wsz(PR[gidx[npv]].astype(np.float32) / pscale, cv2.INTER_LINEAR) if npv in gidx else gp
            rgbp = rd(frames[i - 1]) if i > 0 else rgb
            inf, rws = AM.frame_features(models[active], rgb, rgbp, gfg, gp, gpp, gw, L)
            s1 = inf['s1'] > 0; mlp = s1.copy(); orc = s1.copy(); nd = 0; tg = float('nan')
            if rws is not None:
                p = net.predict((rws['X'].astype(np.float32)[:, cols] - mu) / sd); dec = p >= thr
                mlp[rws['ys'], rws['xs']] = dec; orc[rws['ys'], rws['xs']] = rws['y'] > 0
                nd = len(dec); tg = float((dec == (rws['guid'] > 0)).mean())
            up = lambda x: cv2.resize(x.astype(np.uint8), (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            masks = {'CPB': up(m > 127), 'guidance': gn, 'stage1': up(s1), 'framework': up(inf['s3p'] > 0),
                     'framework_blk': up(inf['s3b'] > 0), 'MLP': up(mlp), 'oracle': up(orc)}
            for s_ in SYS:
                pm = masks[s_]
                rows[s_].append((n, int((pm & fg & valid).sum()), int((pm & ~fg & valid).sum()), int((~pm & fg & valid).sum()),
                                 int((~pm & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())))
            disp.append((n, nd, tg)); bar.update()
        bar.close()
        AM.save_json(outp, dict(sequence=sid, rows=rows, disputes=disp, switches=switches, n_train=N))
        r = {s_: LCE.seq_metrics(rows[s_]) for s_ in SYS}
        P(f'  per-frame F: ' + '  '.join(f'{s_} {r[s_]["thesis_fg"]:.3f}' for s_ in SYS) + f'   ({(time.time() - t0) / 60:.1f} min)')
    # ---------------- summary over the sequences that are done
    R = {}
    for p in sorted(glob.glob(os.path.join(args.out, 'seq', '*.json'))):
        dd = json.load(open(p)); R[dd['sequence']] = {s_: LCE.seq_metrics(dd['rows'][s_]) for s_ in SYS}
    if not R:
        return
    cats = [c for c in LCE.STATIC if any(sq.split('_')[1] == c for sq in R)]
    for agg, title in (('thesis_fg', 'PER-FRAME F (main metric)'), ('pooled', 'POOLED F (supplementary)'),
                       ('fpr_empty', 'FALSE-POSITIVE RATE on empty-GT frames')):
        P(f'\n  {title}: per category (mean of its sequences) and mean over {len(R)} sequences')
        P(f'  {"system":14s}' + ''.join(f'{c:>7s}' for c in cats) + f'{"MEAN":>8s}')
        for s_ in SYS:
            v = [np.nanmean([R[sq][s_][agg] for sq in R if sq.split('_')[1] == c]) for c in cats]
            P(f'  {s_:14s}' + ''.join(f'{x:7.3f}' for x in v) + f'{np.nanmean([R[sq][s_][agg] for sq in R]):8.4f}')
    P('\n  per sequence (per-frame F):')
    P(f'  {"sequence":10s}' + ''.join(f'{s_[:13]:>14s}' for s_ in SYS))
    for sq in R:
        P(f'  {sq:10s}' + ''.join(f'{R[sq][s_]["thesis_fg"]:14.4f}' for s_ in SYS))
    AM.save_json(os.path.join(args.out, 'lasiesta_results.json'), R)
    P(f'\n  -> {args.out}/lasiesta_results.json, seq/<ID>.json (per-frame counts), lasiesta_log.txt')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    l = sub.add_parser('limu')
    l.add_argument('--limu', required=True); l.add_argument('--guid', required=True)
    l.add_argument('--model', required=True); l.add_argument('--json', default=None)
    l.add_argument('--out', default='arbiter_external'); l.add_argument('--guid_prev_step', type=int, default=2)
    l.add_argument('--guid_name', default='DeepLabV3+', help='name of the guidance model (log text only)')
    l.add_argument('--breakdown', action='store_true', help='also decompose the remaining error (agreement vs disputes, ...)')
    w = sub.add_parser('wallflower')
    w.add_argument('--wf', required=True); w.add_argument('--guid', required=True)
    w.add_argument('--model', required=True); w.add_argument('--json', default=None)
    w.add_argument('--guid_name', default='DeepLabV3+', help='name of the guidance model (log text only)')
    w.add_argument('--venus', default=None, help="optional json with the authors' released code results")
    w.add_argument('--out', default='arbiter_external'); w.add_argument('--guid_prev_step', type=int, default=2)
    z = sub.add_parser('lasiesta')
    z.add_argument('--lasiesta', required=True); z.add_argument('--guid', required=True)
    z.add_argument('--model', required=True); z.add_argument('--json', default=None)
    z.add_argument('--cpb_out', default='lasiesta_runs', help='folder of lasiesta_cpb_eval.py (holds trained_models/)')
    z.add_argument('--guid_name', default='DeepLabV3+'); z.add_argument('--budget', type=float, default=160 * 120)
    z.add_argument('--out', default='arbiter_external_lasiesta'); z.add_argument('--guid_prev_step', type=int, default=2)
    z.add_argument('--only', default='')
    args = ap.parse_args()
    {'limu': cmd_limu, 'wallflower': cmd_wallflower, 'lasiesta': cmd_lasiesta}[args.cmd](args)


if __name__ == '__main__':
    main()
