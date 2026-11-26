#!/usr/bin/env python3
"""
Semantic Background Subtraction (SemanticBGS; Braham, Pierard, Van Droogenbroeck, ICIP 2017) under OUR protocol.

Per pixel x and frame t, with the semantic probability p_t(x) and the BGS decision B_t(x):
  rule 1   p_t(x) <= tau_BG                    ->  background
  rule 2   p_t(x) - M_t(x) >= tau_FG           ->  foreground
  else     B_t(x)
  M_0 = p_0 (first frame of the run); if D_t(x) = background, M_{t+1}(x) = p_t(x) with probability alpha.
Paper's default (fixed) thresholds: tau_BG = 0.00366, tau_FG = 0.88627, alpha = 0.00024. They were derived on
CDnet 2014 by the authors, so nothing is tuned on LIMU / WallFlower / LASIESTA.

Semantic probability: the paper sums the softmax probabilities of the 12 ADE20K classes
{person, car, cushion, box, book, boat, bus, truck, bottle, van, bag, bicycle} (PSPNet). Our guidance caches store
the MAX of those 12 probabilities (same classes, same ADE20K networks), a lower bound of the sum that equals it
whenever one class dominates; this approximation is stated in the paper.

BGS methods it is applied to, every frame in order (as designed: an online post-processing of the BGS output):
  CPB      the same frozen locked CPB (R6) and model switching as every other CPB row (apply_arbiter.py)
  SuBSENSE BGSLibrary 3.3.1 defaults, run from the first frame (as in bgs_baselines.py)
Scored frames and metrics: identical to the other rows (LIMU GT frames, per-frame F etc.; WallFlower evaluated frame).

  python -u semantic_bgs.py --dataset limu --limu data/limu --guid deeplab=caches/limu_deeplab,pspnet=caches/limu_pspnet --out sbgs_runs
"""
import sys, os, re, glob, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np, cv2

TAU_BG, TAU_FG, ALPHA = 0.00366, 0.88627, 0.00024
LIMU_ROWS = {}


class SBGS:
    def __init__(self, seed=0):
        self.M = None; self.rng = np.random.RandomState(seed)

    def step(self, B, p):
        if self.M is None:
            self.M = p.copy()
        D = B.copy()
        D[p <= TAU_BG] = False
        D[(p - self.M) >= TAU_FG] = True
        upd = (~D) & (self.rng.random_sample(p.shape) < ALPHA)
        self.M[upd] = p[upd]
        return D


def load_cache(path, shape):
    z = np.load(path); pscale = 255.0 if z['prob'].dtype == np.uint8 else 1.0
    idx = {int(n): i for i, n in enumerate(z['nums'])}
    P = z['prob']
    def prob(n):
        p = P[idx[n]].astype(np.float32) / pscale
        return cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) if p.shape != shape else p
    return idx, prob


def counts(m, gt):
    return (int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum()), bool(gt.any()))


# ------------------------------------------------------------------ BGS streams: yield (frame number, native bool mask)
def limu_cpb_stream(sq, seq, a):
    import limu_framework as LF, limu_eval as L, wf_frame_switch as S
    A = S.RGBModel([sq.rgb(n) for n in range(a.ntrain)], a, 'A'); models = [A]
    other, info = LF.find_states_limu(sq, a)
    if other is not None:
        sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
        models.append(S.RGBModel([sq.rgb(n) for n in sel], a, 'B'))
    W, H = L.WORK; work = (H, W); active = 0
    for n in range(a.ntrain, sq.N):
        if n not in sq.frames:
            continue
        rgb = sq.rgb(n); m = models[active].detect(rgb, work, a)
        if len(models) > 1 and np.mean(m > 127) > a.trigger:
            fr = [float(np.mean(mm.detect(rgb, work, a) > 127)) for mm in models]; b = int(np.argmin(fr))
            if b != active:
                active = b; m = models[active].detect(rgb, work, a)
        # the arbiter tables score CPB with the model active at that frame; the switch decision is taken on the
        # previous active model exactly as in apply_arbiter.py, then the new active model gives the mask
        yield n, cv2.resize((m > 127).astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST) > 0


def subsense_stream(paths):
    import pybgs
    os.makedirs('config', exist_ok=True); bg = pybgs.SuBSENSE()
    for n, p in paths:
        m = bg.apply(cv2.imread(p))
        yield n, (m.max(2) if m.ndim == 3 else m) > 127


def run_limu(args, guids):
    import limu_eval as L, limu_metrics as LM, apply_arbiter as AA
    a = AA.limu_args(); L.WORK = (160, 120); out = {}
    for seq in ['CameraParameter', 'Intersection', 'LightSwitch']:
        sq = L.Seq(args.limu, seq); caches = {g: load_cache(os.path.join(p, f'{seq}.npz'), (240, 320)) for g, p in guids.items()}
        for base in args.bases.split(','):
            t0 = time.time()
            stream = (limu_cpb_stream(sq, seq, a) if base == 'CPB' else
                      subsense_stream([(n, sq.frames[n]) for n in sorted(sq.frames)]))
            sb = {g: SBGS(0) for g in guids}; rows = {g: [] for g in guids}; rows_base = []
            for n, B in stream:
                for g, (idx, prob) in caches.items():
                    if n not in idx:
                        continue
                    D = sb[g].step(B, prob(n))
                    if n in sq.gts and n >= 400:
                        rows[g].append(counts(D, sq.gt(n)))
                if n in sq.gts and n >= 400:
                    rows_base.append(counts(B, sq.gt(n)))
            for g in guids:
                LIMU_ROWS.setdefault(f'SemanticBGS({base}, {g})', {})[seq] = rows[g]
                json.dump(LIMU_ROWS, open(os.path.join(args.out, 'sbgs_limu_rows.json'), 'w'))
                r = LM.metrics(rows[g]); e = [x for x in rows[g] if not x[3]]
                r['fpr_empty'] = float(np.mean([x[1] / (240 * 320) for x in e])) if e else float('nan')
                out.setdefault(f'SemanticBGS({base}, {g})', {})[seq] = r
            rb = LM.metrics(rows_base); out.setdefault(f'{base} (stream check)', {})[seq] = rb
            print(f'  {seq:16s} {base:9s} ' + '  '.join(f'{g} {out[f"SemanticBGS({base}, {g})"][seq]["thesis_fg"]:.4f}' for g in guids)
                  + f'  | base alone {rb["thesis_fg"]:.4f}  ({len(rows_base)} GT frames, {time.time() - t0:.0f}s)', flush=True)
    return out


def run_wf(args, guids):
    import wallflower_eval as W, wf_frame_switch as S, apply_arbiter as AA
    a = AA.limu_args(); a.max_train = 200; W.WF = args.wf; out = {}
    for seq in AA.WF_SEQS:
        d = os.path.join(args.wf, seq); tr, run, ev = W.script_windows(d)
        gt = cv2.imread([os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0], cv2.IMREAD_GRAYSCALE)
        spec = seq == 'MovedObject'; caches = {g: load_cache(os.path.join(p, f'{seq}.npz'), gt.shape) for g, p in guids.items()}
        frames = sorted(set(tr + run + [ev]))
        for base in args.bases.split(','):
            if base == 'CPB':
                A = S.RGBModel([S.load_rgb(d, n) for n in tr], a, 'A'); models = [A]
                other, info = S.find_states(d, tr[-1], a)
                if other is not None:
                    sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
                    models.append(S.RGBModel([S.load_rgb(d, n) for n in sel], a, 'B'))
                def stream():
                    active = 0
                    for n in frames:
                        rgb = S.load_rgb(d, n); m = models[active].detect(rgb, gt.shape, a)
                        if n > tr[-1] and len(models) > 1 and np.mean(m > 127) > a.trigger:
                            fr = [float(np.mean(mm.detect(rgb, gt.shape, a) > 127)) for mm in models]; b = int(np.argmin(fr))
                            if b != active:
                                active = b; m = models[active].detect(rgb, gt.shape, a)
                        yield n, m > 127
                st = stream()
            else:
                st = subsense_stream([(n, os.path.join(d, f'b{n:05d}.bmp')) for n in frames])
            sb = {g: SBGS(0) for g in guids}; res = {}
            for n, B in st:
                for g, (idx, prob) in caches.items():
                    if n in idx:
                        D = sb[g].step(B, prob(n))
                        if n == ev:
                            res[g] = D
                if n == ev:
                    res['_base'] = B
            for g in guids:
                sc, dd = W.score(res[g].astype(np.uint8) * 255, gt, spec)
                out.setdefault(f'SemanticBGS({base}, {g})', {})[seq] = dict(score=sc, P=dd['prec'], R=dd['rec'])
            sc, _ = W.score(res['_base'].astype(np.uint8) * 255, gt, spec)
            out.setdefault(f'{base} (stream check)', {})[seq] = dict(score=sc)
            print(f'  {seq:18s} {base:9s} ' + '  '.join(f'{g} {out[f"SemanticBGS({base}, {g})"][seq]["score"]:.4f}' for g in guids)
                  + f'  | base alone {sc:.4f}', flush=True)
    return out


def run_las(args, guids):
    import lasiesta_cpb_eval as LCE, cdnet_cpb_eval as E
    out = {}; seqs = LCE.find_sequences(args.lasiesta)
    if args.only:
        seqs = {k: v for k, v in seqs.items() if k in args.only.split(',')}
    L = E.locked_args(0.0)
    for sid, (d, gtd) in seqs.items():
        t0 = time.time()
        frames = sorted(glob.glob(os.path.join(d, '*.bmp')), key=LCE.num)
        gts = {LCE.num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
        Lf = len(frames); H, W = cv2.imread(frames[0]).shape[:2]; ww, wh = E.work_size(W, H, 160 * 120)
        tr, sc, N = LCE.window(Lf, 'A', 200)
        caches = {g: load_cache(os.path.join(p, f'{sid}.npz'), (H, W)) for g, p in guids.items()}
        for base in args.bases.split(','):
            if base == 'CPB':
                rd = lambda p: cv2.cvtColor(cv2.resize(cv2.imread(p), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
                ca = argparse.Namespace(out=args.cpb_out, budget=160 * 120, max_train=200, rd=rd)
                models, _, _ = LCE.train_or_load(sid, 'A', ca, frames, tr, ww, wh)
                def stream():
                    active = 0
                    for i, p in enumerate(frames):
                        rgb = rd(p); m = models[active].detect(rgb, (wh, ww), L)
                        if i >= N and len(models) > 1 and np.mean(m > 127) > L.trigger:
                            fr = [float(np.mean(mm.detect(rgb, (wh, ww), L) > 127)) for mm in models]; b = int(np.argmin(fr))
                            if b != active:
                                active = b; m = models[active].detect(rgb, (wh, ww), L)
                        yield LCE.num(p), cv2.resize((m > 127).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                st = stream()
            else:
                st = subsense_stream([(LCE.num(p), p) for p in frames])
            sb = {g: SBGS(0) for g in guids}; rows = {g: [] for g in guids}; rb = []
            first_scored = LCE.num(frames[N]) if N < Lf else 10 ** 9
            for n, B in st:
                fg = valid = None
                if n >= first_scored:
                    fg, valid = LCE.read_gt(gts.get(n, ''))
                for g, (idx, prob) in caches.items():
                    if n not in idx:
                        continue
                    D = sb[g].step(B, prob(n))
                    if fg is not None:
                        rows[g].append((n, int((D & fg & valid).sum()), int((D & ~fg & valid).sum()), int((~D & fg & valid).sum()),
                                        int((~D & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())))
                if fg is not None:
                    rb.append((n, int((B & fg & valid).sum()), int((B & ~fg & valid).sum()), int((~B & fg & valid).sum()),
                               int((~B & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())))
            for g in guids:
                out.setdefault(f'SemanticBGS({base}, {g})', {})[sid] = rows[g]
            out.setdefault(f'{base} (stream check)', {})[sid] = rb
            print(f'  {sid:9s} {base:9s} ' + '  '.join(f'{g} {LCE.seq_metrics(rows[g])["thesis_fg"]:.4f}' for g in guids)
                  + f'  | base alone {LCE.seq_metrics(rb)["thesis_fg"]:.4f}  ({len(rb)} frames, {time.time() - t0:.0f}s)', flush=True)
        json.dump(out, open(os.path.join(args.out, 'sbgs_lasiesta_rows.json'), 'w'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['limu', 'wallflower', 'lasiesta'])
    ap.add_argument('--limu', default=PATHS.LIMU); ap.add_argument('--wf', default=PATHS.WALLFLOWER)
    ap.add_argument('--guid', required=True, help='name=folder,name=folder')
    ap.add_argument('--lasiesta', default=PATHS.LASIESTA); ap.add_argument('--cpb_out', default=PATHS.CPB_LASIESTA)
    ap.add_argument('--only', default='')
    ap.add_argument('--bases', default='CPB,SuBSENSE'); ap.add_argument('--out', default='sbgs_runs')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    guids = dict(x.split('=') for x in a.guid.split(','))
    print(f'### SemanticBGS {a.dataset}  tau_BG {TAU_BG} tau_FG {TAU_FG} alpha {ALPHA}  guidance {list(guids)}  bases {a.bases}', flush=True)
    if a.dataset == 'lasiesta':
        run_las(a, guids); print('->', os.path.join(a.out, 'sbgs_lasiesta_rows.json')); return
    res = run_limu(a, guids) if a.dataset == 'limu' else run_wf(a, guids)
    json.dump(res, open(os.path.join(a.out, f'sbgs_{a.dataset}.json'), 'w'), indent=1)
    print('->', os.path.join(a.out, f'sbgs_{a.dataset}.json'))


if __name__ == '__main__':
    main()
