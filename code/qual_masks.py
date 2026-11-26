#!/usr/bin/env python3
"""
Qualitative masks for the paper: for chosen frames, the exact masks of every system as scored in the tables
(same frozen CPB with frame switching, same guidance caches, same frozen arbiters, SemanticBGS with its full
temporal state). Output: one npz per (dataset, sequence) with input, GT and all masks at native resolution, and
the per-frame F of each mask (checked against the scored per-frame counts).
"""
import sys, os, glob, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np, cv2
import arbiter_mlp as AM
import apply_arbiter as AA
import semantic_bgs as SB

MODELS = PATHS.MODELS
CACHES = PATHS.CACHES
F1 = lambda m, g, v: 2 * (m & g & v).sum() / max(2 * (m & g & v).sum() + (m & ~g & v).sum() + (~m & g & v).sum(), 1)


class Guid:
    def __init__(self, path, shape):
        z = np.load(path); self.idx = {int(n): i for i, n in enumerate(z['nums'])}; self.G, self.P = z['guid'], z['prob']
        self.sc = 255.0 if self.P.dtype == np.uint8 else 1.0; self.shape = shape

    def mask(self, n):
        g = self.G[self.idx[n]] > 127
        return cv2.resize(g.astype(np.uint8), (self.shape[1], self.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 if g.shape != self.shape else g

    def prob(self, n, shape):
        p = self.P[self.idx[n]].astype(np.float32) / self.sc
        return cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) if p.shape != shape else p


def arbitrate(M, rgb, rgbp, gd, n, work, gwork, L, arb, prev_step=2):
    """Return stage-1, framework (paper Stage 2) and arbiter masks at working size for one guidance."""
    net, cols, mu, sd, thr, meta = arb
    gfg = cv2.resize(gd.G[gd.idx[n]].astype(np.uint8) // 255, (work[1], work[0]), interpolation=cv2.INTER_NEAREST)
    gp = gd.prob(n, work); gpp = gd.prob(n - prev_step, work) if (n - prev_step) in gd.idx else gp
    inf, rws = AM.frame_features(M, rgb, rgbp, gfg, gp, gpp, gwork, L)
    s1 = inf['s1'] > 0; mlp = s1.copy()
    if rws is not None:
        p = net.predict((rws['X'].astype(np.float32)[:, cols] - mu) / sd); mlp[rws['ys'], rws['xs']] = p >= thr
    return s1, inf['s3p'] > 0, mlp


def run(dataset, seq, targets, out):
    t0 = time.time(); arbs = {k: AM.load_mlp(v) for k, v in MODELS.items()}
    import wf_frame_switch as S
    if dataset == 'limu':
        import limu_eval as LE, limu_framework as LF
        a = AA.limu_args(); LE.WORK = (160, 120); L = a
        sq = LE.Seq(PATHS.LIMU, seq); native = (240, 320); work = (120, 160)
        models = [S.RGBModel([sq.rgb(n) for n in range(a.ntrain)], a, 'A')]
        other, _ = LF.find_states_limu(sq, a)
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
            models.append(S.RGBModel([sq.rgb(n) for n in sel], a, 'B'))
        frames = [n for n in range(a.ntrain, sq.N) if n in sq.frames]
        rgb_of = sq.rgb; img_of = lambda n: cv2.imread(sq.frames[n])
        gt_of = lambda n: (sq.gt(n), np.ones(native, bool)); switch_from = a.ntrain
    elif dataset == 'wallflower':
        import wallflower_eval as W
        a = AA.limu_args(); a.max_train = 200; W.WF = PATHS.WALLFLOWER; L = a
        d = os.path.join(PATHS.WALLFLOWER, seq); tr, runf, ev = W.script_windows(d)
        g8 = cv2.imread([os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0], 0)
        native = g8.shape; work = native
        models = [S.RGBModel([S.load_rgb(d, n) for n in tr], a, 'A')]
        other, _ = S.find_states(d, tr[-1], a)
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
            models.append(S.RGBModel([S.load_rgb(d, n) for n in sel], a, 'B'))
        frames = sorted(set(tr + runf + [ev])); rgb_of = lambda n: S.load_rgb(d, n)
        img_of = lambda n: cv2.imread(os.path.join(d, f'b{n:05d}.bmp')); gt_of = lambda n: (g8 > 127, np.ones(native, bool))
        switch_from = tr[-1] + 1; targets = [ev]
    else:
        import lasiesta_cpb_eval as LCE, cdnet_cpb_eval as E
        sd_, gtd = LCE.find_sequences(PATHS.LASIESTA)[seq]
        fl = sorted(glob.glob(os.path.join(sd_, '*.bmp')), key=LCE.num); byn = {LCE.num(p): p for p in fl}
        gts = {LCE.num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
        H, Wd = cv2.imread(fl[0]).shape[:2]; ww, wh = E.work_size(Wd, H, 160 * 120); native = (H, Wd); work = (wh, ww)
        L = E.locked_args(0.0); L.sigema = 0.8
        rd = lambda p: cv2.cvtColor(cv2.resize(cv2.imread(p), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
        tr, sc, N = LCE.window(len(fl), 'A', 200)
        ca = argparse.Namespace(out=PATHS.CPB_LASIESTA, budget=160 * 120, max_train=200, rd=rd)
        models, _, _ = LCE.train_or_load(seq, 'A', ca, fl, tr, ww, wh)
        frames = [LCE.num(p) for p in fl]; rgb_of = lambda n: rd(byn[n]); img_of = lambda n: cv2.imread(byn[n])
        gt_of = lambda n: LCE.read_gt(gts[n]); switch_from = LCE.num(fl[N]); a = L
    gds = {k: Guid(os.path.join(v, f'{seq}.npz'), native) for k, v in CACHES[dataset].items()}
    sbgs = {k: SB.SBGS(0) for k in gds}; active = 0; last = max(targets); res = {}
    for n in frames:
        if n > last:
            break
        rgb = rgb_of(n); m = models[active].detect(rgb, work, a)
        if n >= switch_from and len(models) > 1 and np.mean(m > 127) > a.trigger:
            fr = [float(np.mean(mm.detect(rgb, work, a) > 127)) for mm in models]; b = int(np.argmin(fr))
            if b != active:
                active = b; m = models[active].detect(rgb, work, a)
        up = lambda x: cv2.resize(x.astype(np.uint8), (native[1], native[0]), interpolation=cv2.INTER_NEAREST) > 0 if x.shape != native else x
        B = up(m > 127)
        D = {k: (sbgs[k].step(B, gds[k].prob(n, native)) if n in gds[k].idx else None) for k in gds}
        if n in targets:
            fg, valid = gt_of(n)
            gw = np.where(fg, 255, 0).astype(np.uint8); gw[~valid] = 85
            gwork = cv2.resize(gw, (work[1], work[0]), interpolation=cv2.INTER_NEAREST)
            prev = [x for x in frames if x < n]; rgbp = rgb_of(prev[-1]) if prev else rgb
            r = dict(input=img_of(n), gt=fg, valid=valid, CPB=B)
            for k in gds:
                s1, fw, mlp = arbitrate(models[active], rgb, rgbp, gds[k], n, work, gwork, L, arbs[k])
                r[f'guid_{k}'] = gds[k].mask(n); r[f'fw_{k}'] = up(fw); r[f'arb_{k}'] = up(mlp); r[f'sbgs_{k}'] = D[k]
            r['F'] = {k: float(F1(v, fg, valid)) for k, v in r.items() if isinstance(v, np.ndarray) and v.dtype == bool and k not in ('gt', 'valid')}
            res[n] = r
            print(f'  {dataset} {seq} frame {n}: ' + '  '.join(f'{k} {v:.3f}' for k, v in r['F'].items()), flush=True)
    os.makedirs(out, exist_ok=True)
    for n, r in res.items():
        np.savez_compressed(os.path.join(out, f'{dataset}__{seq}__{n}.npz'), **{k: v for k, v in r.items() if k != 'F'}, F=json.dumps(r['F']))
    print(f'  done {seq} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset'); ap.add_argument('--seq'); ap.add_argument('--frames', default='')
    ap.add_argument('--out', default='qualitative_masks')
    a = ap.parse_args()
    run(a.dataset, a.seq, [int(x) for x in a.frames.split(',') if x], a.out)
