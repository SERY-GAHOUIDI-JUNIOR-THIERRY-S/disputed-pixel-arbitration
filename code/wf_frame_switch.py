#!/usr/bin/env python3
"""
================================================================================
WallFlower: (1) update-starvation diagnostics  and  (2) frame-level model switching
================================================================================
Base CPB = the locked baseline: Zhou-MATLAB port, RGB (per-channel models, masks OR-combined),
MAD sigma, sigma floor 3, 3x3 median per channel, eta 2.5, lambda 0.5, official script.txt Train
window, corrected online update over the official Run frames (alpha 0.01, sigma frozen).

(1) --diag : for every Run frame (and the evaluated frame) of the chosen sequences, print
    brightness, foreground fraction, share of pixels eligible for the update (classified
    background), and the mean change of the offset b. Ground truth exists only for the evaluated
    frame, so per-frame F cannot be computed.

(2) --switch : frame-level background maintenance as in Toyama et al. 1999 (WallFlower), applied
    with ONE rule to every sequence, no ground truth anywhere:
      a. training period = frames 0 .. end of the official Train window (the original paper: the
         training period of LightSwitch "sees a room with the lights both on and off");
      b. k-means (k=2) on the training-period frames (downsampled colour images);
      c. a second scene state is accepted only if the two cluster centres are genuinely different
         (> --state_frac of pixels differ by more than --state_diff grey levels) and the smaller
         cluster holds >= --min_cluster frames; otherwise no switching (identical to the baseline);
      d. model A = the official Train-window model (the current state, as in the baseline);
         model B = CPB trained on up to 200 frames of the OTHER cluster, sampled evenly;
      e. at every Run frame and the evaluated frame: detect with the active model; if the
         foreground fraction exceeds --trigger (0.7, as in WallFlower), detect with every stored
         model and make the one with the FEWEST foreground pixels active;
      f. the active model is then updated on that frame (alpha 0.01), exactly as in the baseline.

(3) --trainer consistent --robust 1 --frame_guard 0.5 : robust training (robust_cpb.py).
    Per-pixel rejection of foreground observations in the training frames, consistent block-median
    statistics, and the frame guard (a frame with > 50% rejected pixels is a global change -> kept).

LOCKED ENHANCED CPB (WallFlower overall 0.6803):  python wf_frame_switch.py --preset final
  == --trainer consistent --robust 1 --switch --frame_guard 0.5 (robust training for BOTH models A and B)
================================================================================
"""
import sys, os, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import cpb_kernels as F
import wallflower_eval as W
import robust_cpb as RC

PAPER = {'Bootstrap': 0.6518, 'Camouflage': 0.6112, 'ForegroundAperture': 0.5900, 'LightSwitch': 0.7157,
         'TimeOfDay': 0.7564, 'WavingTrees': 0.7033, 'MovedObject': 0.8922}
SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']


def P(*a):
    print(*a, flush=True)


class RGBModel:
    """Three per-channel CPB models (R, G, B) trained on the same frames."""
    def __init__(self, frames_rgb, args, name):
        self.name = name
        self.models = []; rej = []
        for c in range(3):
            st = np.stack([f[:, :, c] for f in frames_rgb]).astype(np.float32)
            if args.trainer == 'zhou':          # locked baseline trainer (old MAD pass on block MEAN)
                self.models.append(W.train_model(st, args.robust_sigma, args.block, args.selection, args.gamma))
            else:                               # consistent block-median trainer, optional robust training
                rob = args.robust if (name[0] == 'A' or args.robust_B < 0) else args.robust_B
                m, stt = RC.train_channel(st, robust=bool(rob), mad=(args.robust_sigma == 'mad'), sig_all=bool(args.sig_all),
                                          max_frame_reject=args.frame_guard)
                self.models.append(m); rej.append(stt['rejected'])
        if rej:
            P(f'    model {name[0]}: trainer=consistent robust={rob} sig_all={args.sig_all} frame_guard={args.frame_guard}  rejected training observations '
              f'{100 * np.mean(rej):.1f}%')

    def detect(self, rgb, shape, args):
        masks = [W.detect_chan(m, rgb[:, :, c], shape, args.af, args.floor, args.bth, args.ref, bool(args.median))
                 for c, m in enumerate(self.models)]
        return np.maximum.reduce(masks)

    def update(self, rgb, args):
        """One causal online step per channel; returns (eligible fraction, mean |delta b|)."""
        elig, db = [], []
        for c, m in enumerate(self.models):
            D = np.ascontiguousarray(m['DBLK_r'], np.float32); b0 = D[:, :, :20, 2].copy()
            H, Wd = m['img_size']; ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8)); pool = D.shape[2]
            g = np.ascontiguousarray(rgb[:, :, c], np.float32)
            guid = np.zeros((H, Wd), np.uint8); gcnt = F.block_fg_count(guid, ws, hs)
            s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
            F.k_stage1_online(g, F.block_median(g, ws, hs), gcnt, guid, D, s1, sc, su,
                              float(args.af), float(args.floor), F.LAMBDA_DEC, float(args.bth), int(args.ref),
                              pool, True, float(args.alpha), float(args.alpha_sig))
            m['DBLK_r'] = D; m['DBLK_g'] = D; m['DBLK_b'] = D
            elig.append(float(np.mean(s1 == 0))); db.append(float(np.mean(np.abs(D[:, :, :20, 2] - b0))))
        return float(np.mean(elig)), float(np.mean(db))


def load_rgb(d, n):
    return W.load_rgb_gray(f'{d}/b{n:05d}.bmp')[0]


def kmeans2(X, iters=50, seed=0):
    """Plain k-means, k=2, on row vectors X; deterministic farthest-point init."""
    rng = np.random.RandomState(seed)
    c0 = X[rng.randint(len(X))]
    c1 = X[np.argmax(((X - c0) ** 2).sum(1))]
    C = np.stack([c0, c1]).astype(np.float64)
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(2)
        new = d.argmin(1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for k in range(2):
            if np.any(lab == k):
                C[k] = X[lab == k].mean(0)
    return lab, C


def find_states(d, train_end, args):
    """k-means on the training period 0..train_end; returns (other_cluster_frames or None, info)."""
    nums = [n for n in range(0, train_end + 1) if os.path.exists(f'{d}/b{n:05d}.bmp')]
    feats = np.stack([cv2.resize(load_rgb(d, n), (40, 30), interpolation=cv2.INTER_AREA).ravel() for n in nums])
    lab, C = kmeans2(feats)
    cur = lab[-1]                                   # state of the latest training frame = current state
    other = [n for n, l in zip(nums, lab) if l != cur]
    cimg = C.reshape(2, 30, 40, 3).mean(3)          # centre images (grey)
    diff_frac = float(np.mean(np.abs(cimg[0] - cimg[1]) > args.state_diff))
    info = (f'training period 0-{train_end} ({len(nums)} frames): clusters {np.sum(lab == cur)} (current) / '
            f'{len(other)} (other); centre brightness {cimg[cur].mean():.1f} vs {cimg[1 - cur].mean():.1f}; '
            f'{100 * diff_frac:.0f}% of pixels differ by >{args.state_diff:g}')
    accepted = diff_frac > args.state_frac and len(other) >= args.min_cluster
    return (other if accepted else None), info + ('  -> SECOND STATE ACCEPTED' if accepted else '  -> no second state (single model)')


def evaluate(seq, args):
    d = f'{W.WF}/{seq}'
    tr, run, ev = W.script_windows(d)
    gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
    gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
    is_spec = seq == 'MovedObject'
    t0 = time.time()
    A = RGBModel([load_rgb(d, n) for n in tr], args, f'A (official Train {tr[0]}-{tr[-1]})')
    models = [A]
    if args.switch:
        other, info = find_states(d, tr[-1], args)
        P(f'  {info}')
        if other is not None:
            sel = [other[i] for i in np.linspace(0, len(other) - 1, min(args.max_train, len(other))).round().astype(int)]
            models.append(RGBModel([load_rgb(d, n) for n in sel], args, f'B (other state, {len(sel)} frames from {sel[0]}-{sel[-1]})'))
            P(f'  stored models: {models[0].name} | {models[1].name}')
    active = 0
    switches = []
    for n in run + [ev]:
        rgb = load_rgb(d, n)
        m = models[active].detect(rgb, gt.shape, args)
        fgf = float(np.mean(m > 127))
        line = f'    frame {n:5d}  luma {W.F.to_luma(rgb).mean():6.1f}  fg {100 * fgf:5.1f}%  active={models[active].name[0]}'
        if args.switch and len(models) > 1 and fgf > args.trigger:
            fr = [float(np.mean(mm.detect(rgb, gt.shape, args) > 127)) for mm in models]
            best = int(np.argmin(fr))
            line += f'  TRIGGER: fg per model ' + ', '.join(f'{mm.name[0]}={100 * f:.1f}%' for mm, f in zip(models, fr))
            if best != active:
                switches.append((n, models[active].name[0], models[best].name[0]))
                active = best
                line += f'  -> SWITCH to {models[active].name[0]}'
        if n == ev:
            if args.diag:
                P(line + '   <- evaluated frame')
            break
        if args.alpha > 0:
            el, db = models[active].update(rgb, args)
            line += f'  update-eligible {100 * el:5.1f}%  mean|db| {db:.3f}'
        if args.diag:
            P(line)
    mask = models[active].detect(load_rgb(d, ev), gt.shape, args)
    s, dd = W.score(mask, gt, is_spec)
    P(f'  >> {seq}: {"spec" if is_spec else "F"}={s:.4f}  P={dd["prec"]:.2f} R={dd["rec"]:.2f}  '
      f'switches={switches if switches else "none"}  paper={PAPER[seq]:.4f}  ({time.time() - t0:.0f}s)')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', default=','.join(SEQS))
    ap.add_argument('--diag', action='store_true', help='print per-frame starvation diagnostics')
    ap.add_argument('--switch', action='store_true', help='frame-level model switching (WallFlower)')
    ap.add_argument('--trigger', type=float, default=0.7, help='foreground fraction that triggers a model check (WallFlower: 0.7)')
    ap.add_argument('--state_diff', type=float, default=25.0, help='grey-level difference between cluster centres')
    ap.add_argument('--state_frac', type=float, default=0.5, help='fraction of pixels that must differ to accept a 2nd state')
    ap.add_argument('--min_cluster', type=int, default=50, help='minimum frames in the second state')
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--trainer', choices=['zhou', 'consistent'], default='zhou',
                    help='zhou = locked baseline trainer; consistent = block-median MAD (+ optional robust training)')
    ap.add_argument('--robust', type=int, default=0, help='1 = reject foreground observations per pixel during training')
    ap.add_argument('--robust_B', type=int, default=-1, help='-1 = same as --robust; 0/1 = override for the second-state model B')
    ap.add_argument('--frame_guard', type=float, default=1.0,
                    help='robust training: a frame with more than this share of rejected pixels is a global change -> kept whole (1.0 = off)')
    ap.add_argument('--sig_all', type=int, default=0, help='1 = robust selection/offset but MAD sigma over ALL frames')
    ap.add_argument('--alpha', type=float, default=0.01); ap.add_argument('--alpha_sig', type=float, default=0.0)
    ap.add_argument('--af', type=float, default=2.5); ap.add_argument('--floor', type=float, default=3.0)
    ap.add_argument('--bth', type=float, default=7.0); ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--median', type=int, default=1); ap.add_argument('--robust_sigma', default='mad')
    ap.add_argument('--block', default='median'); ap.add_argument('--selection', default='minstd'); ap.add_argument('--gamma', default='matlab')
    ap.add_argument('--wf_root', default=None, help='WallFlower folder (default: $WALLFLOWER_ROOT or ../data/wallflower)')
    ap.add_argument('--preset', choices=['none', 'final'], default='none',
                    help='final = the locked enhanced CPB: --trainer consistent --robust 1 --switch --frame_guard 0.5')
    a = ap.parse_args()
    if a.preset == 'final':
        a.trainer, a.robust, a.switch, a.frame_guard, a.robust_B, a.sig_all = 'consistent', 1, True, 0.5, -1, 0
    if a.wf_root:
        W.WF = a.wf_root
    P('EFFECTIVE ARGS: ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())) + f'  WF={W.WF}')
    P(f'CONFIG: trainer={a.trainer} robust={a.robust} switch={a.switch} trigger={a.trigger} alpha={a.alpha} RGB MAD floor={a.floor} median={a.median} '
      f'eta={a.af} official script.txt windows')
    res = {}
    for seq in a.seqs.split(','):
        P(f'\n=== {seq}')
        res[seq] = evaluate(seq, a)
    six = [s for s in SEQS[:6] if s in res]
    if len(six) == 6:
        P('\nSUMMARY  ' + '  '.join(f'{s}={res[s]:.3f}' for s in six)
          + f'   OVERALL={np.mean([res[s] for s in six]):.4f}   (paper 0.6714)'
          + (f'   MovedObject spec={res["MovedObject"]:.3f}' if 'MovedObject' in res else ''))


if __name__ == '__main__':
    main()
