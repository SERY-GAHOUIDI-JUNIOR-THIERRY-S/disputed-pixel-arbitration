#!/usr/bin/env python3
"""
================================================================================
wf_framework.py -- FULL PIPELINE on WallFlower: locked enhanced CPB + semantic guidance + 3-stage communication
================================================================================
CPB = the locked configuration of wf_frame_switch.py --preset final (consistent + robust training,
frame guard, frame-level switching, online update alpha 0.01), unchanged.

Guidance enters in two places (Liang et al. 2021, Section 3.3):
  * Run frames (online maintenance): on frames that have guidance, the update is GATED -- no update
    where guidance says foreground, and support blocks covered by guidance foreground (>= bth pixels)
    are skipped (k_stage1_online). Frames without guidance use the locked CPB update unchanged.
  * Evaluated frame, per RGB channel, OR-combined:
      stage 1  CPB with guidance-vetoed supports (block with >= bth guidance-fg pixels discarded) + 3x3 median
      stage 2  paper Eq. 6-8: on disputed pixels, supports with gamma >= gamma_bar are replaced by reserved
               candidates and Eq. 5 is re-run (k_stage2_paper, copied from faithful_framework.py);
               omega uses the block MEDIAN, consistent with how b was trained
      stage 3  foreground similarity r_F < eps * r_B (k_stage3), eps = 0.8
Frame-level switching keeps the locked rule (unguided CPB foreground share > 0.7).

--n_guided N : guidance only on the last N frames before/including the evaluated frame (30 = quick test);
               0 = every Run frame (the full protocol).
--guidance   : cache (real DeepLabV3+/PSPNet .npz) | zero | oracle_eval (GT of the evaluated frame: a ceiling).
Reports, on the evaluated frame: CPB (locked), guidance only, stage1, stage1+2, stage1+2+3.
================================================================================
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from numba import njit, prange
import cpb_kernels as F
import wallflower_eval as W
import wf_frame_switch as S


def P(*a):
    print(*a, flush=True)


# ============================================================================
# FAITHFUL STAGE 2  (paper Eq 6-8: candidate-block replacement, then re-run Eq 5) -- from faithful_framework.py
# ============================================================================
@njit(cache=False, parallel=True, fastmath=False)
def k_stage2_paper(gray, DBS, gcnt, guid_fg, D, cpb_in, out, af, floor, lam, BTH, REF, pool):
    H, W = gray.shape
    ws, hs = DBS.shape
    for x in prange(H):
        for y in range(W):
            out[x, y] = cpb_in[x, y]
            cfg = cpb_in[x, y] > 0
            gfg = guid_fg[x, y] > 0
            if cfg == gfg:
                continue                              # agreement pixel: untouched
            case1 = cfg and (not gfg)                 # CPB fg, guidance bg
            # ---- pass 1: gamma_bar over the relevant support set ----
            Ksum = 0.0; sum_rel = 0.0; cntM = 0
            for k in range(pool):
                if cntM >= REF:
                    break
                gam = D[x, y, k, 4]
                if gam <= 0.0:
                    continue
                bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                if bi < 0: bi = 0
                elif bi >= ws: bi = ws - 1
                if bj < 0: bj = 0
                elif bj >= hs: bj = hs - 1
                if gcnt[bi, bj] >= BTH:               # Stage-1 veto
                    continue
                cntM += 1
                sig = D[x, y, k, 3]
                if sig < floor: sig = floor
                diff = gray[x, y] - DBS[bi, bj] - D[x, y, k, 2]
                uns = (diff > af * sig) or (diff < -af * sig)     # omega_m (Eq 4): unsteady
                if case1:
                    if uns:
                        sum_rel += gam; Ksum += 1.0              # relevant = unsteady
                else:
                    if not uns:
                        sum_rel += gam; Ksum += 1.0              # relevant = steady
            if cntM == 0 or Ksum == 0.0:
                continue
            gbar = sum_rel / Ksum
            # ---- pass 2: replace supports with gamma >= gamma_bar, re-run Eq 5 ----
            total = 0.0; changed = 0.0; used_cand = 0; rk = 0
            for k in range(pool):
                if rk >= REF:
                    break
                gam = D[x, y, k, 4]
                if gam <= 0.0:
                    continue
                bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
                if bi < 0: bi = 0
                elif bi >= ws: bi = ws - 1
                if bj < 0: bj = 0
                elif bj >= hs: bj = hs - 1
                if gcnt[bi, bj] >= BTH:
                    continue
                rk += 1
                sig = D[x, y, k, 3]
                if sig < floor: sig = floor
                diff = gray[x, y] - DBS[bi, bj] - D[x, y, k, 2]
                uns = (diff > af * sig) or (diff < -af * sig)
                if gam >= gbar:
                    # ---- replace with the next unused reserved candidate ----
                    rep = False; crank = 0
                    for k2 in range(pool):
                        gam2 = D[x, y, k2, 4]
                        if gam2 <= 0.0:
                            continue
                        bi2 = int(D[x, y, k2, 0]); bj2 = int(D[x, y, k2, 1])
                        if bi2 < 0: bi2 = 0
                        elif bi2 >= ws: bi2 = ws - 1
                        if bj2 < 0: bj2 = 0
                        elif bj2 >= hs: bj2 = hs - 1
                        if gcnt[bi2, bj2] >= BTH:
                            continue
                        crank += 1
                        if crank <= REF:
                            continue                     # skip originals
                        if crank - REF <= used_cand:
                            continue                     # skip already-used candidates
                        used_cand += 1
                        sig2 = D[x, y, k2, 3]
                        if sig2 < floor: sig2 = floor
                        diff2 = gray[x, y] - DBS[bi2, bj2] - D[x, y, k2, 2]
                        uns2 = (diff2 > af * sig2) or (diff2 < -af * sig2)
                        total += gam2
                        if uns2: changed += gam2
                        rep = True
                        break
                    if not rep:                           # no candidate left -> keep original
                        total += gam
                        if uns: changed += gam
                else:
                    total += gam
                    if uns: changed += gam
            if total > 1e-9:
                out[x, y] = 255 if changed > lam * total else 0


# ============================================================================
# FRAMEWORK ON ONE FRAME (per RGB channel, OR-combined; same channel logic as the locked CPB)
# ============================================================================
def frame_stages(model_rgb, rgb, guid_fg, args):
    """Returns dict of 0/255 masks: s1 (CPB, guided supports), s2 (paper stage 2), s3 (stage 3), all after
    the per-channel 3x3 median used by the locked CPB; OR over R,G,B."""
    acc = {}
    for c, m in enumerate(model_rgb.models):
        D = np.ascontiguousarray(m['DBLK_r'], np.float32)
        H, Wd = m['img_size']; ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8)); pool = D.shape[2]
        g = np.ascontiguousarray(rgb[:, :, c], np.float32)
        gcnt = F.block_fg_count(guid_fg, ws, hs)
        DBM = F.block_median(g, ws, hs)                      # consistent with training (b = mean(I_p - Q_median))
        s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
        F.k_stage1(g, DBM, gcnt, guid_fg, D, s1, sc, su, float(args.af), float(args.floor), F.LAMBDA_DEC,
                   float(args.bth), int(args.ref), pool, True)
        if args.median:
            s1 = cv2.medianBlur(s1, 3)
        s2 = np.zeros((H, Wd), np.uint8)
        k_stage2_paper(g, DBM, gcnt, guid_fg, D, s1, s2, float(args.af), float(args.floor), F.LAMBDA_DEC,
                       float(args.bth), int(args.ref), pool)
        s3 = np.zeros((H, Wd), np.uint8)
        F.k_stage3(g, DBM, gcnt, guid_fg, D, s2, s3, float(args.sigema), float(args.bth), int(args.ref), pool)
        s2m = np.zeros((H, Wd), np.uint8)      # authors' MATLAB BlockReplacement.m (NaE/NaB)
        F.k_stage2(g, DBM, gcnt, guid_fg, guid_fg.astype(np.float32), D, s1, s2m, float(args.af), float(args.floor),
                   float(args.bth), int(args.ref), pool, 1.0, 0.0)
        s3m = np.zeros((H, Wd), np.uint8)
        F.k_stage3(g, DBM, gcnt, guid_fg, D, s2m, s3m, float(args.sigema), float(args.bth), int(args.ref), pool)
        for k, v in (('s1', s1), ('s2', s2), ('s3', s3), ('s2m', s2m), ('s3m', s3m)):
            acc[k] = v if k not in acc else np.maximum(acc[k], v)
    return acc


def guided_update(model_rgb, rgb, guid_fg, args):
    """One causal online step per channel, gated by guidance: no update where guidance says foreground,
    and support blocks covered by guidance foreground (>= bth pixels) are neither used nor updated."""
    for c, m in enumerate(model_rgb.models):
        D = np.ascontiguousarray(m['DBLK_r'], np.float32)
        H, Wd = m['img_size']; ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8)); pool = D.shape[2]
        g = np.ascontiguousarray(rgb[:, :, c], np.float32)
        gcnt = F.block_fg_count(guid_fg, ws, hs)
        s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
        F.k_stage1_online(g, F.block_median(g, ws, hs), gcnt, guid_fg, D, s1, sc, su,
                          float(args.af), float(args.floor), F.LAMBDA_DEC, float(args.bth), int(args.ref),
                          pool, True, float(args.alpha), float(args.alpha_sig))
        m['DBLK_r'] = D; m['DBLK_g'] = D; m['DBLK_b'] = D


# ============================================================================
# GUIDANCE SOURCES
# ============================================================================
class Guidance:
    """kind = cache (real DeepLabV3+/PSPNet cache from make_guidance_wallflower.py), zero, oracle_eval
    (ground truth on the evaluated frame only: a CEILING / plumbing check, never a result)."""
    def __init__(self, kind, seq, d, ev, cache_dir=None):
        self.kind, self.ev = kind, ev
        self.table = {}
        if kind == 'cache':
            z = np.load(os.path.join(cache_dir, f'{seq}.npz'))
            gall = z['guid']                       # decompress ONCE (indexing z['guid'] inside the loop re-reads it per frame)
            self.table = {int(n): gall[i] for i, n in enumerate(z['nums'])}
        elif kind == 'oracle_eval':
            gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
            self.table = {ev: cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)}

    def get(self, n, shape):
        if self.kind == 'zero':
            return np.zeros(shape, np.uint8)
        g = self.table.get(n)
        if g is None:
            return None
        if g.shape != shape:
            g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (g > 127).astype(np.uint8)


# ============================================================================
# ONE SEQUENCE
# ============================================================================
def evaluate(seq, args):
    d = f'{W.WF}/{seq}'
    tr, run, ev = W.script_windows(d)
    gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
    gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
    is_spec = seq == 'MovedObject'
    t0 = time.time()
    A = S.RGBModel([S.load_rgb(d, n) for n in tr], args, f'A (official Train {tr[0]}-{tr[-1]})')
    models = [A]
    if args.switch:
        other, info = S.find_states(d, tr[-1], args)
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(args.max_train, len(other))).round().astype(int)]
            models.append(S.RGBModel([S.load_rgb(d, n) for n in sel], args, f'B (other state {sel[0]}-{sel[-1]})'))
    G = Guidance(args.guidance, seq, d, ev, args.cache_dir)
    seqframes = run + [ev]
    guided = set(seqframes[-args.n_guided:]) if args.n_guided > 0 else set(seqframes)
    active = 0; switches = []; n_gated = 0; gfrac = []
    zero = np.zeros(gt.shape, np.uint8)
    for n in seqframes:
        rgb = S.load_rgb(d, n)
        gmask = G.get(n, gt.shape) if n in guided else None
        if gmask is not None:
            gfrac.append(float(gmask.mean()))
        # frame-level switching: identical rule to the locked CPB (unguided CPB foreground share)
        fgf = float(np.mean(models[active].detect(rgb, gt.shape, args) > 127))
        if len(models) > 1 and fgf > args.trigger:
            fr = [float(np.mean(mm.detect(rgb, gt.shape, args) > 127)) for mm in models]
            best = int(np.argmin(fr))
            if best != active:
                switches.append((n, models[active].name[0], models[best].name[0])); active = best
        if n == ev:
            break
        if args.alpha > 0:
            if gmask is not None and args.gate_update:
                guided_update(models[active], rgb, gmask, args); n_gated += 1
            else:
                models[active].update(rgb, args)             # == locked CPB update (no guidance)
    rgb = S.load_rgb(d, ev)
    gmask = G.get(ev, gt.shape)
    cpb = models[active].detect(rgb, gt.shape, args)         # locked CPB mask (reference)
    res = {'CPB (locked)': cpb}
    if gmask is not None:
        st = frame_stages(models[active], rgb, gmask, args)
        res.update({'guidance only': gmask * 255, 'stage1 (guided supports)': st['s1'],
                    'stage1+2': st['s2'], 'stage1+2+3 (framework)': st['s3'],
                    'stage1+2 MATLAB': st['s2m'], 'stage1+2+3 MATLAB': st['s3m']})
    out = {}
    for k, msk in res.items():
        s, dd = W.score(msk, gt, is_spec)
        out[k] = (s, dd['prec'], dd['rec'])
    P(f'\n=== {seq}  [{args.guidance}]  guidance on the last {len(guided)} frames (Run+eval); '
      f'gated updates {n_gated}; switches {switches if switches else "none"}; '
      f'mean guidance fg {100 * np.mean(gfrac) if gfrac else 0:.1f}%  ({time.time() - t0:.0f}s)')
    for k, (s, p, r) in out.items():
        P(f'    {k:26s} {"spec" if is_spec else "F"}={s:.4f}  P={p:.2f} R={r:.2f}')
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        tiles = [cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)] + \
                [cv2.cvtColor(res[k].astype(np.uint8), cv2.COLOR_GRAY2BGR) for k in res]
        cv2.imwrite(os.path.join(args.save_dir, f'{seq}_{args.guidance}.png'),
                    cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    return {k: v[0] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guidance', choices=['cache', 'zero', 'oracle_eval'], default='cache')
    ap.add_argument('--cache_dir', default=None, help='folder with <Sequence>.npz from make_guidance_wallflower.py')
    ap.add_argument('--n_guided', type=int, default=30, help='guidance on the last N frames (Run+eval); 0 = all Run frames')
    ap.add_argument('--gate_update', type=int, default=1, help='1 = guidance-gated online update on guided frames')
    ap.add_argument('--sigema', type=float, default=0.8, help='stage-3 epsilon (paper)')
    ap.add_argument('--seqs', default=','.join(S.SEQS))
    ap.add_argument('--wf_root', default=None)
    ap.add_argument('--save_dir', default=None)
    # locked CPB (identical defaults to wf_frame_switch.py --preset final)
    ap.add_argument('--trainer', default='consistent'); ap.add_argument('--robust', type=int, default=1)
    ap.add_argument('--robust_B', type=int, default=-1); ap.add_argument('--sig_all', type=int, default=0)
    ap.add_argument('--frame_guard', type=float, default=0.5); ap.add_argument('--switch', type=int, default=1)
    ap.add_argument('--trigger', type=float, default=0.7); ap.add_argument('--state_diff', type=float, default=25.0)
    ap.add_argument('--state_frac', type=float, default=0.5); ap.add_argument('--min_cluster', type=int, default=50)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--alpha', type=float, default=0.01); ap.add_argument('--alpha_sig', type=float, default=0.0)
    ap.add_argument('--af', type=float, default=2.5); ap.add_argument('--floor', type=float, default=3.0)
    ap.add_argument('--bth', type=float, default=7.0); ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--median', type=int, default=1); ap.add_argument('--robust_sigma', default='mad')
    ap.add_argument('--block', default='median'); ap.add_argument('--selection', default='minstd'); ap.add_argument('--gamma', default='matlab')
    a = ap.parse_args()
    if a.wf_root:
        W.WF = a.wf_root
    if a.guidance == 'cache' and not a.cache_dir:
        raise SystemExit('--guidance cache needs --cache_dir')
    P('EFFECTIVE ARGS: ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())) + f'  WF={W.WF}')
    if a.guidance == 'oracle_eval':
        P('NOTE: oracle_eval uses the GROUND TRUTH of the evaluated frame as guidance -> ceiling / plumbing check, NOT a result.')
    res = {}
    for seq in a.seqs.split(','):
        res[seq] = evaluate(seq, a)
    six = [s for s in S.SEQS[:6] if s in res]
    if six:
        keys = list(res[six[0]].keys())
        P(f'\nSUMMARY [{a.guidance}, guidance on last {a.n_guided or "all"} frames]   paper CPB 0.6714 | paper CPB+guidance: see Table 3')
        P('  ' + f'{"":26s}' + ''.join(f'{s[:9]:>10s}' for s in six) + f'{"OVERALL":>10s}' + ('  MovedObj' if 'MovedObject' in res else ''))
        for k in keys:
            vals = [res[s].get(k, float('nan')) for s in six]
            mo = f'{res["MovedObject"].get(k, float("nan")):10.3f}' if 'MovedObject' in res else ''
            P(f'  {k:26s}' + ''.join(f'{v:10.3f}' for v in vals) + f'{np.nanmean(vals):10.3f}' + mo)


if __name__ == '__main__':
    main()
