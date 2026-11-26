#!/usr/bin/env python3
"""
Classic unsupervised background-subtraction baselines (BGSLibrary 3.3.1, default parameters) run under OUR protocol
on LIMU, WallFlower and LASIESTA, so they can sit in the controlled comparison table.

Methods (BGSLibrary class names): SuBSENSE, PAWCS, LOBSTER, ViBe, MixtureOfGaussianV2 (GMM, Zivkovic), KDE,
PixelBasedAdaptiveSegmenter (PBAS). Each is used as designed: online and adaptive, default parameters, native
resolution, colour input. Output mask: foreground = value > 127 (shadow labels 127, if any, count as background).

Protocol (identical frames and metrics to CPB / framework / arbiter):
  LIMU        every frame from 0 to the end in order (frames 0-399 = the CPB training window act as warm-up);
              every GT frame (every 15 frames from 500) scored at native 320x240; per-frame F (Zhou thesis) + pooled F
              + false-positive rate on empty-GT frames
  WallFlower  official script.txt window: Train + Run frames in order (warm-up), then the Evaluate frame is scored;
              F on the 6 sequences with foreground, specificity on MovedObject
  LASIESTA    20 static-camera sequences, frames 1..L in order; rule A: frames N+1..L scored, N = min(200, floor(0.4 L));
              GT foreground = objects + stopped objects, grey ignored; per-frame F, pooled F, false-positive rate on empty frames
Resumable: one JSON per (method, dataset, sequence) under --out.

  python -u bgs_baselines.py --methods SuBSENSE,ViBe --datasets limu,wallflower,lasiesta --limu <LIMU> --wf <WallFlower> --lasiesta <LASIESTA>
"""
import sys, os, re, glob, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PATHS
import numpy as np, cv2
import pybgs
import wallflower_eval as W
import lasiesta_cpb_eval as LCE

METHODS = ['SuBSENSE', 'PAWCS', 'LOBSTER', 'ViBe', 'MixtureOfGaussianV2', 'KDE', 'PixelBasedAdaptiveSegmenter']
LIMU_SEQS = ['CameraParameter', 'Intersection', 'LightSwitch']
WF_SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']
num = lambda p: int(re.findall(r'(\d+)', os.path.basename(p))[-1])


def make(method):
    os.makedirs('config', exist_ok=True)                    # BGSLibrary stores its default parameters here
    return getattr(pybgs, method)()


def fgmask(m):
    if m.ndim == 3:
        m = m.max(2)
    return m > 127


def counts(pm, fg, valid):
    return [int((pm & fg & valid).sum()), int((pm & ~fg & valid).sum()), int((~pm & fg & valid).sum()),
            int((~pm & ~fg & valid).sum()), int((fg & valid).any()), int(valid.sum())]


def run_limu(method, seq, root):
    d = os.path.join(root, seq)
    frames = {num(p): p for p in glob.glob(os.path.join(d, '*.jpg'))}
    gts = {int(re.findall(r'(\d+)_gt', os.path.basename(p))[-1]): p for p in glob.glob(os.path.join(d, '*_gt.bmp'))}
    a = make(method); rows = []
    for n in sorted(frames):
        m = fgmask(a.apply(cv2.imread(frames[n])))
        if n in gts and n >= 400:
            g = cv2.imread(gts[n], 0) > 127
            pm = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            rows.append([n] + counts(pm, g, np.ones_like(g)))
    return rows


def run_wf(method, seq, root):
    d = os.path.join(root, seq); tr, run, ev = W.script_windows(d)
    gtp = [os.path.join(d, f) for f in os.listdir(d) if f.lower().startswith('hand_segmented')][0]
    gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE); a = make(method); m = None
    for n in tr + run + [ev]:
        m = fgmask(a.apply(cv2.imread(os.path.join(d, f'b{n:05d}.bmp'))))
    s, dd = W.score(m.astype(np.uint8) * 255, gt, seq == 'MovedObject')
    return dict(score=s, P=dd['prec'], R=dd['rec'], tp=dd['tp'], fp=dd['fp'], fn=dd['fn'], tn=dd['tn'])


def run_lasiesta(method, seq, d, gtd):
    frames = sorted(glob.glob(os.path.join(d, '*.bmp')), key=num)
    gts = {num(p): p for p in glob.glob(os.path.join(gtd, '*.png'))}
    L = len(frames); N = min(200, int(0.4 * L)); a = make(method); rows = []
    for i, p in enumerate(frames):
        m = fgmask(a.apply(cv2.imread(p)))
        if i >= N:
            fg, valid = LCE.read_gt(gts.get(num(p), ''))
            if fg is not None:
                rows.append([num(p)] + counts(m, fg, valid))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', default=','.join(METHODS)); ap.add_argument('--datasets', default='limu,wallflower,lasiesta')
    ap.add_argument('--limu', default=PATHS.LIMU); ap.add_argument('--wf', default=PATHS.WALLFLOWER)
    ap.add_argument('--lasiesta', default=PATHS.LASIESTA); ap.add_argument('--out', default='bgs_runs')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    jobs = []
    for meth in a.methods.split(','):
        for ds in a.datasets.split(','):
            if ds == 'limu':
                jobs += [(meth, ds, s, (a.limu,)) for s in LIMU_SEQS]
            elif ds == 'wallflower':
                jobs += [(meth, ds, s, (a.wf,)) for s in WF_SEQS]
            elif ds == 'lasiesta':
                jobs += [(meth, ds, s, v) for s, v in LCE.find_sequences(a.lasiesta).items()]
    t0 = time.time()
    for k, (meth, ds, s, args) in enumerate(jobs, 1):
        outp = os.path.join(a.out, f'{meth}__{ds}__{s}.json')
        if os.path.exists(outp):
            continue
        t1 = time.time()
        res = (run_limu(meth, s, *args) if ds == 'limu' else run_wf(meth, s, *args) if ds == 'wallflower'
               else run_lasiesta(meth, s, *args))
        tmp = outp + '.tmp'
        json.dump(dict(method=meth, dataset=ds, sequence=s, result=res), open(tmp, 'w')); os.replace(tmp, outp)
        print(f'[{k}/{len(jobs)}] {meth:28s} {ds:10s} {s:18s} {time.time() - t1:6.0f}s   total {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
