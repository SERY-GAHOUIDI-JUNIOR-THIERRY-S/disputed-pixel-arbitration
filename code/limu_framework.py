#!/usr/bin/env python3
"""
LIMU (rebuilt): CPB + DeepLabV3+ guidance + 3-stage communication, scored PER FRAME with every aggregation.
Protocol: train on the first 400 frames (Liang et al.), process at 160x120, score every GT frame at 320x240.
--cpb robust : locked enhanced CPB (consistent + robust training, frame guard 0.5, frame-level switching), RGB OR
--cpb zhou   : Zhou-port CPB (min-std, median-of-medians, MAD, floor 3), RGB OR, no switching
--alpha 0    : frozen model (default, = the LIMU protocol we used); > 0: causal online update over every frame
One pass gives, on every GT frame and from the same model state:
  CPB | guidance only | stage1 | stage1+2 paper (Eq.6-8) | +3 | stage1+2 MATLAB (BlockReplacement NaE/NaB) | +3
Per-frame rows are written to <out>/<label>_<seq>_perframe.csv.
"""
import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np, cv2
import cpb_kernels as F
import wf_frame_switch as S
import wf_framework as WF
import limu_eval as L
import limu_metrics as LM

PAPER_CPB = {'CameraParameter': 0.6545, 'Intersection': 0.6778, 'LightSwitch': 0.6633}
SEQS = ['CameraParameter', 'Intersection', 'LightSwitch']
KEYS = ['CPB', 'guidance only', 'stage1', 'stage1+2 paper', 'stage1+2+3 paper', 'stage1+2 matlab', 'stage1+2+3 matlab']


def P(*a):
    print(*a, flush=True)


def stages(model_rgb, rgb, gfg, a):
    acc = {}
    for c, m in enumerate(model_rgb.models):
        D = np.ascontiguousarray(m['DBLK_r'], np.float32)
        H, Wd = m['img_size']; ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8)); pool = D.shape[2]
        g = np.ascontiguousarray(rgb[:, :, c], np.float32)
        gcnt = F.block_fg_count(gfg, ws, hs); DBM = F.block_median(g, ws, hs)
        s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
        F.k_stage1(g, DBM, gcnt, gfg, D, s1, sc, su, a.af, a.floor, F.LAMBDA_DEC, a.bth, a.ref, pool, True)
        s1 = cv2.medianBlur(s1, 3)
        s2p = np.zeros_like(s1); WF.k_stage2_paper(g, DBM, gcnt, gfg, D, s1, s2p, a.af, a.floor, F.LAMBDA_DEC, a.bth, a.ref, pool)
        s3p = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2p, s3p, a.sigema, a.bth, a.ref, pool)
        s2m = np.zeros_like(s1); F.k_stage2(g, DBM, gcnt, gfg, gfg.astype(np.float32), D, s1, s2m, a.af, a.floor, a.bth, a.ref, pool, 1.0, 0.0)
        s3m = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2m, s3m, a.sigema, a.bth, a.ref, pool)
        for k, v in (('stage1', s1), ('stage1+2 paper', s2p), ('stage1+2+3 paper', s3p),
                     ('stage1+2 matlab', s2m), ('stage1+2+3 matlab', s3m)):
            acc[k] = v if k not in acc else np.maximum(acc[k], v)
    return acc


def find_states_limu(sq, a):
    feats = np.stack([cv2.resize(sq.rgb(n), (40, 30), interpolation=cv2.INTER_AREA).ravel() for n in range(a.ntrain)])
    lab, C = S.kmeans2(feats); cur = lab[-1]
    other = [n for n in range(a.ntrain) if lab[n] != cur]
    cimg = C.reshape(2, 30, 40, 3).mean(3)
    frac = float(np.mean(np.abs(cimg[0] - cimg[1]) > a.state_diff))
    ok = frac > a.state_frac and len(other) >= a.min_cluster
    return (other if ok else None), f'k-means on 0-{a.ntrain - 1}: {len(other)} frames in other cluster, {100 * frac:.0f}% pixels differ -> {"2nd state" if ok else "single model"}'


def run_seq(seq, a):
    t0 = time.time(); sq = L.Seq(a.limu, seq)
    A = S.RGBModel([sq.rgb(n) for n in range(a.ntrain)], a, 'A (frames 0-399)'); models = [A]
    if a.switch:
        other, info = find_states_limu(sq, a); P('  ' + info)
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(a.max_train, len(other))).round().astype(int)]
            models.append(S.RGBModel([sq.rgb(n) for n in sel], a, 'B (other state)'))
    G = None
    if a.guid_dir:
        z = np.load(os.path.join(a.guid_dir, f'{seq}.npz')); gall = z['guid']
        G = {int(n): gall[i] for i, n in enumerate(z['nums'])}
    sequential = len(models) > 1 or a.alpha > 0
    frames = range(a.ntrain, sq.N) if sequential else sorted(sq.gts)
    work = (L.WORK[1], L.WORK[0]); active = 0; rows = {k: [] for k in KEYS}; switches = 0
    for n in frames:
        if n not in sq.frames:
            continue
        isgt = n in sq.gts
        if not (isgt or sequential):
            continue
        rgb = sq.rgb(n)
        cpb = models[active].detect(rgb, work, a)
        if len(models) > 1 and np.mean(cpb > 127) > a.trigger:
            fr = [float(np.mean(mm.detect(rgb, work, a) > 127)) for mm in models]; b = int(np.argmin(fr))
            if b != active:
                active = b; switches += 1; cpb = models[active].detect(rgb, work, a)
        if isgt:
            gt = sq.gt(n); up = lambda m: cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            masks = {'CPB': up(cpb > 127)}
            if G is not None and n in G:
                gn = G[n] > 127
                gw = cv2.resize(gn.astype(np.uint8), L.WORK, interpolation=cv2.INTER_NEAREST)
                masks['guidance only'] = gn
                for k, v in stages(models[active], rgb, gw, a).items():
                    masks[k] = up(v > 127)
            for k, m in masks.items():
                rows[k].append((int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum()), bool(gt.any()), n))
        if a.alpha > 0:
            models[active].update(rgb, a)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for k, r in rows.items():
        if not r:
            continue
        with open(f'{a.out}/{a.label}_{seq}_{k.replace(" ", "_").replace("+", "p")}_perframe.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['tp', 'fp', 'fn', 'has_fg', 'frame']); w.writerows(r)
        res[k] = LM.metrics([x[:4] for x in r])
    P(f'\n=== {seq}  [{a.label}]  switches={switches}  ({time.time() - t0:.0f}s)   paper CPB {PAPER_CPB[seq]:.4f}')
    for k, m in res.items():
        P(f'    {k:20s} ' + '  '.join(f'{x}={m[x]:.3f}' for x in LM.KEYS) + f'  (P_fg {m["P_fg"]:.2f} R_fg {m["R_fg"]:.2f})')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limu', default=PATHS.LIMU); ap.add_argument('--guid_dir', default=None)
    ap.add_argument('--cpb', choices=['robust', 'zhou'], default='robust'); ap.add_argument('--label', default=None)
    ap.add_argument('--out', default='limu_runs'); ap.add_argument('--seqs', default=','.join(SEQS))
    ap.add_argument('--ntrain', type=int, default=400)
    ap.add_argument('--alpha', type=float, default=0.0); ap.add_argument('--alpha_sig', type=float, default=0.0)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--trigger', type=float, default=0.7); ap.add_argument('--state_diff', type=float, default=25.0)
    ap.add_argument('--state_frac', type=float, default=0.5); ap.add_argument('--min_cluster', type=int, default=50)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--af', type=float, default=2.5); ap.add_argument('--floor', type=float, default=3.0)
    ap.add_argument('--bth', type=float, default=7.0); ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--median', type=int, default=1)
    ap.add_argument('--work', default='160,120', help='CPB working size W,H (320,240 = LIMU native)')
    a = ap.parse_args()
    L.WORK = tuple(int(v) for v in a.work.split(','))
    a.robust_sigma, a.block, a.selection, a.gamma, a.robust_B, a.sig_all = 'mad', 'median', 'minstd', 'matlab', -1, 0
    if a.cpb == 'robust':
        a.trainer, a.robust, a.frame_guard, a.switch = 'consistent', 1, 0.5, True
    else:
        a.trainer, a.robust, a.frame_guard, a.switch = 'zhou', 0, 1.0, False
    a.label = a.label or f'{a.cpb}_a{a.alpha:g}' + ('' if L.WORK == (160, 120) else f'_w{L.WORK[0]}')
    P('ARGS ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())))
    res = {s: run_seq(s, a) for s in a.seqs.split(',')}
    keys = [k for k in KEYS if all(k in res[s] for s in res)]
    for agg in LM.KEYS:
        P(f'\nOVERALL [{agg}]  ' + ''.join(f'{s[:12]:>14s}' for s in res) + f'{"MEAN":>9s}')
        for k in keys:
            v = [res[s][k][agg] for s in res]
            P(f'  {k:20s}' + ''.join(f'{x:14.3f}' for x in v) + f'{np.mean(v):9.3f}')
    P('  paper CPB                 0.655         0.678         0.663    0.665 | paper framework (DeepLab) overall 0.794')


if __name__ == '__main__':
    main()
