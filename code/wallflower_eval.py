#!/usr/bin/env python3
"""
Wallflower evaluation of PURE CPB (no guidance), following Liang et al.'s protocol:
  - each sequence has ONE hand-segmented ground-truth frame (the test frame),
  - train CPB on the 200 frames immediately preceding the test frame,
  - score ONLY the single test frame.
Metric: F-measure for the 6 standard sequences; specificity for MovedObject
(its ground truth has no foreground).

Reuses the exact validated CPB training kernel (train_channel_fixed) and the exact
pure-CPB detector (pure_cpb_mask) from the project. Also runs the ONLINE update
(k_stage1_online, guidance OFF) by adapting over the 200-frame window before scoring.
"""
import sys, os, glob, re, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import cpb_kernels as F
from cpb_mask import pure_cpb_mask
import cpb_training as TR
import cpb_training_base as T0

WF = os.environ.get('WALLFLOWER_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'wallflower'))
SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']
FMEAS = set(SEQS) - {'MovedObject'}     # MovedObject -> specificity
POOL = T0.POOLNUM
NTRAIN = 200
SCALE = 1.0
TARGET = None   # (H, W) exact resize, e.g. (256,256) to match Zhou

def load_rgb_gray(path):
    im = cv2.imread(path)
    if TARGET is not None:
        im = cv2.resize(im, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_LINEAR)
    elif SCALE != 1.0:
        im = cv2.resize(im, (int(round(im.shape[1]*SCALE)), int(round(im.shape[0]*SCALE))),
                        interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb, F.to_luma(rgb)

import re as _re
def script_windows(d):
    """Official WallFlower protocol from <seq>/script.txt -> (train_nums, run_nums, eval_idx)."""
    txt = open(os.path.join(d, 'script.txt')).read()
    tr = _re.search(r'(\d+)\s*-\s*(\d+)\s+Train', txt)
    rn = _re.search(r'(\d+)\s*-\s*(\d+)\s+Run', txt)
    ev = _re.search(r'(\d+)\s+Evaluate', txt)
    a, b = int(tr.group(1)), int(tr.group(2))
    r0, r1 = (int(rn.group(1)), int(rn.group(2))) if rn else (b + 1, int(ev.group(1)))
    train = [n for n in range(a, b + 1) if os.path.exists(os.path.join(d, f'b{n:05d}.bmp'))]
    run = [n for n in range(max(r0, b + 1), r1) if os.path.exists(os.path.join(d, f'b{n:05d}.bmp'))]
    return train, run, int(ev.group(1))


def seq_info(d):
    g = glob.glob(d + '/hand_segmented_*.BMP') + glob.glob(d + '/hand_segmented_*.bmp')
    idx = int(re.findall(r'(\d+)', os.path.basename(g[0]))[-1])
    return idx, g[0]

def train_model(gray_stack, robust='std', block='median', selection='minstd', gamma='matlab'):
    H, W = gray_stack.shape[1:]
    D = TR.train_channel_fixed(np.ascontiguousarray(gray_stack), POOL, 0.0, 0.0, 'L',
                               block, selection, gamma, robust)
    return {'DBLK_r': D, 'DBLK_g': D, 'DBLK_b': D, 'img_size': (H, W), 'params': {'poolnum': POOL}}

def detect_mask(model, rgb_f32, gt_shape, af, floor, bth, ref, median=True):
    """Pure-CPB detection mirroring pure_cpb_mask but with a 3x3-median toggle."""
    Hf, Wf = model['img_size']
    ws = int(np.ceil(Hf/8)); hs = int(np.ceil(Wf/8))
    D = np.ascontiguousarray(model['DBLK_r'], np.float32); pool = D.shape[2]
    fr = cv2.resize(rgb_f32, (Wf, Hf)) if rgb_f32.shape[:2] != (Hf, Wf) else rgb_f32
    chan = F.to_luma(fr)
    guid = np.zeros((Hf, Wf), np.uint8); gcnt = F.block_fg_count(guid, ws, hs)
    DBS = F.block_mean(chan, ws, hs)
    s1 = np.zeros((Hf, Wf), np.uint8); sc = np.zeros((Hf, Wf), np.float32); su = np.zeros((Hf, Wf), np.float32)
    F.k_stage1(chan, DBS, gcnt, guid, D, s1, sc, su,
               float(af), float(floor), F.LAMBDA_DEC, float(bth), int(ref), pool, True)
    if median:
        s1 = cv2.medianBlur(s1, 3)
    return cv2.resize(s1, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)

def detect_chan(model, chan_native, gt_shape, af, floor, bth, ref, median=True):
    Hf, Wf = model['img_size']; ws = int(np.ceil(Hf/8)); hs = int(np.ceil(Wf/8))
    D = np.ascontiguousarray(model['DBLK_r'], np.float32); pool = D.shape[2]
    chan = cv2.resize(chan_native, (Wf, Hf)) if chan_native.shape != (Hf, Wf) else chan_native
    chan = np.ascontiguousarray(chan, np.float32)
    guid = np.zeros((Hf, Wf), np.uint8); gcnt = F.block_fg_count(guid, ws, hs)
    DBS = F.block_mean(chan, ws, hs)
    s1 = np.zeros((Hf, Wf), np.uint8); sc = np.zeros((Hf, Wf), np.float32); su = np.zeros((Hf, Wf), np.float32)
    F.k_stage1(chan, DBS, gcnt, guid, D, s1, sc, su,
               float(af), float(floor), F.LAMBDA_DEC, float(bth), int(ref), pool, True)
    if median:
        s1 = cv2.medianBlur(s1, 3)
    return cv2.resize(s1, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)

def online_adapt(model, d, tr_nums, af, floor, bth, ref, alpha, alpha_sig, chan=None):
    """chan=None -> luma model; chan=0/1/2 -> the R/G/B channel model (RGB path)."""
    m = pickle.loads(pickle.dumps(model))
    D = np.ascontiguousarray(m['DBLK_r'], np.float32)
    H, W = m['img_size']; ws = int(np.ceil(H/8)); hs = int(np.ceil(W/8)); pool = D.shape[2]
    guid = np.zeros((H, W), np.uint8); gcnt = F.block_fg_count(guid, ws, hs)
    for n in tr_nums:                                   # temporal order, causal
        rgb_n, g = load_rgb_gray(f'{d}/b{n:05d}.bmp')
        if chan is not None:
            g = np.ascontiguousarray(rgb_n[:, :, chan], np.float32)
        DBS = F.block_median(g, ws, hs)     # same block statistic as training (median)
        s1 = np.zeros((H, W), np.uint8); sc = np.zeros((H, W), np.float32); su = np.zeros((H, W), np.float32)
        F.k_stage1_online(g, DBS, gcnt, guid, D, s1, sc, su,
                          float(af), float(floor), F.LAMBDA_DEC, float(bth), int(ref),
                          pool, True, float(alpha), float(alpha_sig))
    m['DBLK_r'] = D; m['DBLK_g'] = D; m['DBLK_b'] = D
    return m

def score(mask, gt, is_spec):
    fg = mask > 127; gtfg = gt > 127
    tp = int((fg & gtfg).sum()); fp = int((fg & ~gtfg).sum())
    fn = int((~fg & gtfg).sum()); tn = int((~fg & ~gtfg).sum())
    e = 1e-10
    pr = tp/(tp+fp+e); rc = tp/(tp+fn+e); f1 = 2*pr*rc/(pr+rc+e)
    spec = tn/(tn+fp+e)
    return (spec if is_spec else f1), dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=pr, rec=rc, spec=spec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor', type=float, default=3.0)
    ap.add_argument('--bth', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--alpha_sig', type=float, default=0.05)
    ap.add_argument('--train_mode', choices=['first', 'pre', 'script'], default='first',
                    help="first = frames 0-199; pre = 200 before test; script = official script.txt Train window")
    ap.add_argument('--robust_sigma', choices=['std', 'mad'], default='std')
    ap.add_argument('--scale', type=float, default=1.0, help='upscale factor for frames (res to Zhou scale)')
    ap.add_argument('--seqs', default=None, help='comma list to restrict sequences')
    ap.add_argument('--skip_online', action='store_true')
    ap.add_argument('--online_full', action='store_true',
                    help='adapt online over ALL frames [0, test) up to the test frame (drift tracking)')
    ap.add_argument('--online_stride', type=int, default=1)
    ap.add_argument('--online_run', action='store_true',
                    help='adapt online over the official script.txt Run frames (needs --train_mode script)')
    ap.add_argument('--clean_frac', type=float, default=1.0,
                    help='keep only the cleanest fraction of training frames (drop busiest)')
    ap.add_argument('--selection', choices=['minstd', 'pearson'], default='minstd')
    ap.add_argument('--block', choices=['median', 'mean'], default='median')
    ap.add_argument('--gamma', choices=['matlab', 'raw'], default='matlab')
    ap.add_argument('--median', type=int, default=1, help='1 = apply Zhou 3x3 median; 0 = none')
    ap.add_argument('--target_hw', default=None, help='exact resize HxW, e.g. 256x256 (Zhou)')
    ap.add_argument('--rgb', action='store_true', help='Zhou 3-channel R/G/B OR-combine (frozen and online)')
    args = ap.parse_args()
    global SCALE, TARGET; SCALE = args.scale
    if args.target_hw:
        TARGET = tuple(int(v) for v in args.target_hw.lower().split('x'))
    seq_list = args.seqs.split(',') if args.seqs else SEQS
    print(f"params: af={args.af} floor={args.floor} bth={args.bth} ref={args.ref} "
          f"alpha={args.alpha} alpha_sig={args.alpha_sig}  POOL={POOL}\n")
    rows = []
    for seq in seq_list:
        d = f'{WF}/{seq}'; idx, gtp = seq_info(d)
        is_spec = seq not in FMEAS
        run_nums = None
        if args.train_mode == 'script':
            tr_nums, run_nums, _ = script_windows(d)
        elif args.train_mode == 'first':
            tr_nums = [n for n in range(0, NTRAIN) if os.path.exists(f'{d}/b{n:05d}.bmp')]
        else:
            tr_nums = [n for n in range(idx - NTRAIN, idx) if os.path.exists(f'{d}/b{n:05d}.bmp')]
        frames = [load_rgb_gray(f'{d}/b{n:05d}.bmp') for n in tr_nums]   # (rgb, luma) each
        luma_stack = np.stack([f[1] for f in frames]).astype(np.float32)
        keep = np.arange(len(luma_stack))
        if args.clean_frac < 1.0 and len(luma_stack) > 20:
            med = np.median(luma_stack, 0)
            activity = np.abs(luma_stack - med[None]).mean(axis=(1, 2))
            keep = np.sort(np.argsort(activity)[:max(20, int(round(args.clean_frac * len(luma_stack))))])
        rgb_t, _ = load_rgb_gray(f'{d}/b{idx:05d}.bmp')
        gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
        if args.rgb:
            masks = []; masks_on = []
            if args.online_run and run_nums is not None:
                on_nums = run_nums[::args.online_stride]
            elif args.online_full:
                on_nums = [n for n in range(0, idx, args.online_stride) if os.path.exists(f'{d}/b{n:05d}.bmp')]
            else:
                on_nums = tr_nums
            for c in range(3):
                cs = np.stack([f[0][:, :, c] for f in frames]).astype(np.float32)[keep]
                mdl = train_model(cs, args.robust_sigma, args.block, args.selection, args.gamma)
                masks.append(detect_chan(mdl, rgb_t[:, :, c], gt.shape,
                                         args.af, args.floor, args.bth, args.ref, bool(args.median)))
                if not args.skip_online:          # per-channel online update (RGB path)
                    mdl_on = online_adapt(mdl, d, on_nums, args.af, args.floor, args.bth, args.ref,
                                          args.alpha, args.alpha_sig, chan=c)
                    masks_on.append(detect_chan(mdl_on, rgb_t[:, :, c], gt.shape,
                                                args.af, args.floor, args.bth, args.ref, bool(args.median)))
            m_fz = np.maximum.reduce(masks)
            s_fz, d_fz = score(m_fz, gt, is_spec)
            if masks_on:
                s_on, d_on = score(np.maximum.reduce(masks_on), gt, is_spec)
            else:
                s_on, d_on = s_fz, d_fz
        else:
            stack = luma_stack[keep]
            model = train_model(stack, args.robust_sigma, args.block, args.selection, args.gamma)
            m_fz = detect_mask(model, rgb_t, gt.shape, args.af, args.floor, args.bth, args.ref, bool(args.median))
            s_fz, d_fz = score(m_fz, gt, is_spec)
            if args.skip_online:
                s_on, d_on = s_fz, d_fz
            else:
                if args.online_run and run_nums is not None:
                    on_nums = run_nums[::args.online_stride]
                elif args.online_full:
                    on_nums = [n for n in range(0, idx, args.online_stride) if os.path.exists(f'{d}/b{n:05d}.bmp')]
                else:
                    on_nums = tr_nums
                model_on = online_adapt(model, d, on_nums, args.af, args.floor, args.bth, args.ref,
                                        args.alpha, args.alpha_sig)
                m_on = detect_mask(model_on, rgb_t, gt.shape, args.af, args.floor, args.bth, args.ref, bool(args.median))
                s_on, d_on = score(m_on, gt, is_spec)
        tag = 'spec' if is_spec else 'F1'
        print(f"{seq:20s} test={idx:4d} train={len(tr_nums)}  "
              f"frozen_{tag}={s_fz:.4f}  online_{tag}={s_on:.4f}   "
              f"(froz P={d_fz['prec']:.2f} R={d_fz['rec']:.2f} | onl P={d_on['prec']:.2f} R={d_on['rec']:.2f})")
        rows.append((seq, is_spec, s_fz, s_on))
    # overall: mean F-measure over the 6 F-measure sequences (paper's "Overall")
    f_fz = [r[2] for r in rows if not r[1]]; f_on = [r[3] for r in rows if not r[1]]
    mo = [r for r in rows if r[1]][0]
    print("\n--- SUMMARY ---")
    print(f"Overall F-measure (6 seqs)   frozen={np.mean(f_fz):.4f}   online={np.mean(f_on):.4f}")
    print(f"MovedObject specificity       frozen={mo[2]:.4f}   online={mo[3]:.4f}")
    print(f"\nPaper CPB overall F-measure = 0.6714  (target)")

if __name__ == '__main__':
    main()
