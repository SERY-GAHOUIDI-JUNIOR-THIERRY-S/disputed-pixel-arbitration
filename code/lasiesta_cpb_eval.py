#!/usr/bin/env python3
"""
LASIESTA -- locked robust CPB (R6, frozen), trained per sequence and saved, then scored. Progress bars, resumable.

Protocol (fixed 17 Nov 2026, before any LASIESTA result; methodology doc section 13):
  * the 20 static-camera sequences I_SI, I_CA, I_OC, I_IL, I_MB, I_BS, O_CL, O_RA, O_SN, O_SU (01 and 02)
  * rule A (main): CPB trained on the first N = min(200, floor(0.4 L)) frames (no GT), frames N+1..L scored
    rule B (supplementary, transductive): CPB trained on up to --max_train frames sampled over the whole sequence,
    every frame scored
  * locked CPB = the same as CDnet / LIMU / WallFlower: per-channel RGB, consistent + robust training, frame guard 0.5,
    MAD sigma, floor 3, eta 2.5, lambda 0.5, K 20, 3x3 median, OR over channels, frame-level switching (trigger 0.7)
  * working size from the 19,200-pixel budget (352x288 -> 152x128); masks upsampled (nearest) to 352x288 for scoring
  * GT: foreground = red / green / yellow (objects) and white (object that stopped); black = background;
    grey (128,128,128, uncertain contour) ignored
  * metrics per sequence: per-frame F (Zhou thesis, main), pooled F, per-frame F with precision on empty frames,
    mean per-frame F, false-positive rate on empty-GT frames; then mean over sequences and per category

Folder layout expected under --root (any nesting): <ID>/<ID>-<n>.bmp and <ID>-GT/<ID>-GT_<n>.png
Outputs under --out (default lasiesta_runs):
  trained_models/<ID>__rule<A|B>.npz (+ .json)   the trained CPB, reused by every later run (trained ONCE)
  rule<A|B>[_testN]/counts/<ID>.csv  (frame, tp, fp, fn, tn, has_fg, valid)   log.txt   summary.txt

Examples (run from the folder that holds cdnet_cpb_eval.py):
  python lasiesta_cpb_eval.py --root "<LASIESTA>" --list
  python -u lasiesta_cpb_eval.py --root "<LASIESTA>" --rule A --only I_SI_01,O_RA_02 --max_frames 50     (quick test)
  python -u lasiesta_cpb_eval.py --root "<LASIESTA>" --rule A                                             (full run)
  python -u lasiesta_cpb_eval.py --root "<LASIESTA>" --rule B                                             (supplementary)
"""
import sys, os, re, csv, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import wf_frame_switch as S
import cdnet_cpb_eval as E


class LM:
    """Per-frame aggregations (same code as limu_metrics.py, inlined so this script needs no extra file).
    rows: (tp, fp, fn, has_fg) per scored frame."""
    @staticmethod
    def metrics(rows):
        rows = [tuple(r) for r in rows]
        TP = sum(r[0] for r in rows); FP = sum(r[1] for r in rows); FN = sum(r[2] for r in rows)
        out = {'pooled': 2 * TP / max(2 * TP + FP + FN, 1)}
        fg = [r for r in rows if r[3]]
        Pf = [r[0] / (r[0] + r[1]) for r in fg if r[0] + r[1] > 0]
        Rf = [r[0] / (r[0] + r[2]) for r in fg if r[0] + r[2] > 0]
        Pm, Rm = (np.mean(Pf) if Pf else 0.0), (np.mean(Rf) if Rf else 0.0)
        out['thesis_fg'] = 2 * Pm * Rm / (Pm + Rm) if Pm + Rm > 0 else 0.0
        out['P_fg'], out['R_fg'] = Pm, Rm
        Pa = Pf + [0.0 for r in rows if not r[3] and r[1] > 0]
        Pa_ = np.mean(Pa) if Pa else 0.0
        out['thesis_all'] = 2 * Pa_ * Rm / (Pa_ + Rm) if Pa_ + Rm > 0 else 0.0
        out['meanF_fg'] = float(np.mean([2 * r[0] / max(2 * r[0] + r[1] + r[2], 1) for r in fg])) if fg else float('nan')
        return out

P = E.P
Bar = E.Bar
SEQ_RE = re.compile(r'^[IO]_[A-Z]{2}_\d{2}$')
STATIC = ('SI', 'CA', 'OC', 'IL', 'MB', 'BS', 'CL', 'RA', 'SN', 'SU')
CAT_NAMES = {'SI': 'simple', 'CA': 'camouflage', 'OC': 'occlusions', 'IL': 'illumination', 'MB': 'modified bg',
             'BS': 'bootstrap', 'CL': 'cloudy', 'RA': 'rainy', 'SN': 'snowy', 'SU': 'sunny'}
num = lambda p: int(re.findall(r'(\d+)', os.path.basename(p))[-1])


def find_sequences(root):
    seqs = {}
    for d, dirs, files in os.walk(root, followlinks=True):
        name = os.path.basename(d)
        if SEQ_RE.match(name) and name.split('_')[1] in STATIC and any(f.lower().endswith('.bmp') for f in files):
            gtd = None
            for cand in (name + '-GT', name + '_GT'):
                for base in (os.path.dirname(d), d):
                    if os.path.isdir(os.path.join(base, cand)):
                        gtd = os.path.join(base, cand)
            if gtd and name not in seqs:
                seqs[name] = (d, gtd)
    return dict(sorted(seqs.items()))


def read_gt(path):
    g = cv2.imread(path)
    if g is None:
        return None, None
    grey = (g == 128).all(2)
    fg = ~((g == 0).all(2) | grey)
    return fg, ~grey


def window(L, rule, max_train):
    if rule == 'A':
        N = min(200, int(0.4 * L))
        return list(range(N)), list(range(N, L)), N
    tr = sorted(set(np.linspace(0, L - 1, min(max_train, L)).round().astype(int).tolist()))
    return tr, list(range(L)), len(tr)


def protocol(a, rule):
    return dict(rule=rule, budget=a.budget, max_train=a.max_train if rule == 'B' else None,
                cpb='locked robust CPB R6 (wf_frame_switch --preset final), frozen')


def train_or_load(sid, rule, a, frames, tr, ww, wh):
    mp = os.path.join(a.out, 'trained_models', f'{sid}__rule{rule}.npz')
    if os.path.exists(mp) and os.path.exists(mp[:-4] + '.json'):
        models, meta = E.load_models(mp)
        if meta.get('protocol') == protocol(a, rule) and meta.get('work') == [ww, wh]:
            return models, meta, True
        P(f'  saved model {sid} rule {rule} has another protocol -> retraining')
    bar = Bar(len(tr), 'train frames'); imgs = []
    for i in tr:
        imgs.append(a.rd(frames[i])); bar.update()
    bar.close()
    L = E.locked_args(); t0 = time.time()
    P(f'  training CPB on {len(tr)} frames ...')
    A = S.RGBModel(imgs, L, 'A (training window)'); models = [A]
    feats = np.stack([cv2.resize(f, (40, 30), interpolation=cv2.INTER_AREA).ravel() for f in imgs])
    lab, C = S.kmeans2(feats); cur = lab[-1]
    other = [i for i in range(len(imgs)) if lab[i] != cur]
    cimg = C.reshape(2, 30, 40, 3).mean(3); frac = float(np.mean(np.abs(cimg[0] - cimg[1]) > L.state_diff))
    second = bool(frac > L.state_frac and len(other) >= L.min_cluster)
    if second:
        sel = [other[i] for i in np.linspace(0, len(other) - 1, min(L.max_train, len(other))).round().astype(int)]
        models.append(S.RGBModel([imgs[i] for i in sel], L, 'B (other state)'))
    meta = dict(protocol=protocol(a, rule), work=[ww, wh], n_train=len(tr), train_frames=[num(frames[i]) for i in tr],
                second_state=second, t_train=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    E.save_models(models, mp, meta)
    P(f'  trained in {meta["t_train"]:.0f}s, saved -> {os.path.relpath(mp)}')
    return models, meta, False


def run_seq(sid, d, gtd, rule, a, outdir, k, n):
    csvp = os.path.join(outdir, 'counts', sid + '.csv')
    if os.path.exists(csvp):
        P(f'[{k}/{n}] {sid}: already scored -> skip'); return
    t0 = time.time()
    frames = sorted(glob.glob(os.path.join(d, '*.bmp')), key=num)
    gts = {num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
    L = len(frames); H, W = cv2.imread(frames[0]).shape[:2]; ww, wh = E.work_size(W, H, a.budget)
    a.rd = lambda p: cv2.cvtColor(cv2.resize(cv2.imread(p), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    tr, sc, N = window(L, rule, a.max_train)
    P(f'[{k}/{n}] {sid}  {W}x{H} -> {ww}x{wh}  {L} frames | rule {rule}: train {N} frames, score {len(sc)}')
    models, meta, loaded = train_or_load(sid, rule, a, frames, tr, ww, wh)
    P(f'  {"loaded saved" if loaded else "trained new"} CPB ({meta["n_train"]} frames, second state {int(meta["second_state"])})')
    Lr = E.locked_args(0.0)
    run = list(range(tr[-1] + 1, L)) if rule == 'A' else list(range(L))
    test = set(sc[:a.max_frames] if a.max_frames else sc)
    active, switches, rows = 0, 0, []
    bar = Bar(len(run), 'score frames')
    for i in run:
        rgb = a.rd(frames[i])
        m = models[active].detect(rgb, (wh, ww), Lr)
        if len(models) > 1 and np.mean(m > 127) > Lr.trigger:
            fr = [float(np.mean(mm.detect(rgb, (wh, ww), Lr) > 127)) for mm in models]; b = int(np.argmin(fr))
            if b != active:
                active, switches = b, switches + 1
                m = models[active].detect(rgb, (wh, ww), Lr)
        if i in test:
            fg, valid = read_gt(gts.get(num(frames[i]), ''))
            if fg is not None:
                pm = cv2.resize((m > 127).astype(np.uint8), (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                rows.append((num(frames[i]), int((pm & fg & valid).sum()), int((pm & ~fg & valid).sum()),
                             int((~pm & fg & valid).sum()), int((~pm & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())))
        bar.update()
        if a.max_frames and len(rows) >= a.max_frames:
            break
    bar.close()
    os.makedirs(os.path.dirname(csvp), exist_ok=True); tmp = E.tmp_name(csvp)
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frame', 'tp', 'fp', 'fn', 'tn', 'has_fg', 'valid']); w.writerows(rows)
    E.safe_replace(tmp, csvp)
    m = seq_metrics(rows)
    P(f'  per-frame F {m["thesis_fg"]:.4f} (P {m["P_fg"]:.3f} R {m["R_fg"]:.3f}) | pooled {m["pooled"]:.4f} | '
      f'FPR empty {m["fpr_empty"]:.4f} ({m["n_empty"]} empty frames) | switches {switches} | {(time.time() - t0) / 60:.1f} min')


def seq_metrics(rows):
    m = LM.metrics([(r[1], r[2], r[3], bool(r[5])) for r in rows])
    e = [r for r in rows if not r[5]]
    m['fpr_empty'] = float(np.mean([r[2] / max(r[6], 1) for r in e])) if e else float('nan')
    m['n_empty'] = len(e); m['n_frames'] = len(rows)
    return m


def summarize(outdir, seqs):
    res = {}
    for sid in seqs:
        p = os.path.join(outdir, 'counts', sid + '.csv')
        if os.path.exists(p):
            with open(p, newline='') as fh:
                rows = [[int(x) for x in r] for r in list(csv.reader(fh))[1:]]
            if rows:
                res[sid] = seq_metrics(rows)
    if not res:
        return ''
    keys = ['thesis_fg', 'pooled', 'thesis_all', 'meanF_fg', 'fpr_empty']
    lines = [f'{"sequence":10s} {"perframe F":>10s} {"P":>6s} {"R":>6s} {"pooled F":>9s} {"pf F+empty":>10s} {"mean F":>7s} {"FPR empty":>10s} {"frames":>7s}']
    for sid, m in res.items():
        lines.append(f'{sid:10s} {m["thesis_fg"]:10.4f} {m["P_fg"]:6.3f} {m["R_fg"]:6.3f} {m["pooled"]:9.4f} {m["thesis_all"]:10.4f} '
                     f'{m["meanF_fg"]:7.4f} {m["fpr_empty"]:10.4f} {m["n_frames"]:7d}')
    cats = {}
    for sid, m in res.items():
        cats.setdefault(sid.split('_')[1], []).append(m)
    lines.append('\nper category (mean of its sequences):')
    for c in [c for c in STATIC if c in cats]:
        lines.append(f'  {c} {CAT_NAMES[c]:13s} ' + '  '.join(f'{k} {np.nanmean([m[k] for m in cats[c]]):.4f}' for k in ('thesis_fg', 'pooled', 'fpr_empty')))
    lines.append(f'\nMEAN over {len(res)} sequences: ' + '  '.join(f'{k} {np.nanmean([m[k] for m in res.values()]):.4f}' for k in keys))
    lines.append(f'MEAN over {len(cats)} categories: ' + '  '.join(f'{k} {np.nanmean([np.nanmean([m[k] for m in v]) for v in cats.values()]):.4f}' for k in keys))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='LASIESTA folder (searched recursively)')
    ap.add_argument('--out', default='lasiesta_runs')
    ap.add_argument('--rule', choices=['A', 'B', 'both'], default='A')
    ap.add_argument('--only', default='', help='comma list of sequence ids (default: all 20 static-camera sequences)')
    ap.add_argument('--max_frames', type=int, default=0, help='quick test: score only the first N scored frames')
    ap.add_argument('--max_train', type=int, default=400, help='rule B only: frames sampled over the sequence')
    ap.add_argument('--budget', type=float, default=160 * 120)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        raise SystemExit(f'ERROR: --root not found: {os.path.abspath(a.root)}')
    seqs = find_sequences(a.root)
    if a.only:
        seqs = {k: v for k, v in seqs.items() if k in [x.strip() for x in a.only.split(',')]}
    print(f'{len(seqs)} LASIESTA static-camera sequences found under {os.path.abspath(a.root)}', flush=True)
    if not seqs:
        raise SystemExit('  -> expected folders like I_SI_01 (with .bmp) next to I_SI_01-GT (with .png)')
    if a.list:
        for k, (d, g) in seqs.items():
            print(f'  {k}: {len(glob.glob(os.path.join(d, "*.bmp")))} frames, {len(glob.glob(os.path.join(g, "*.png")))} GT  ({d})')
        return
    rules = ['A', 'B'] if a.rule == 'both' else [a.rule]
    for rule in rules:
        outdir = os.path.join(a.out, f'rule{rule}' + (f'_test{a.max_frames}' if a.max_frames else ''))
        os.makedirs(outdir, exist_ok=True)
        E.LOGS[:] = [open(os.path.join(outdir, 'log.txt'), 'a', encoding='utf-8')]
        P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")}  LASIESTA CPB  rule {rule}  sequences={len(seqs)}  '
          f'max_frames={a.max_frames or "all"}  out={os.path.abspath(outdir)}')
        t0 = time.time()
        try:
            for k, (sid, (d, g)) in enumerate(seqs.items(), 1):
                run_seq(sid, d, g, rule, a, outdir, k, len(seqs))
        except KeyboardInterrupt:
            P('\nSTOPPED by user. Finished sequences are saved; run the same command again to continue.')
        txt = summarize(outdir, list(seqs))
        if txt:
            txt = f'SUMMARY  LASIESTA CPB (frozen R6)  rule {rule}  {os.path.abspath(outdir)}\n' + txt
            P('\n' + txt)
            open(os.path.join(outdir, 'summary.txt'), 'w', encoding='utf-8').write(txt + '\n')
        P(f'rule {rule} done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
