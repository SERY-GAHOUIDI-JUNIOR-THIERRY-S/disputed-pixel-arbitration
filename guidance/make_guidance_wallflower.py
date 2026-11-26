#!/usr/bin/env python3
"""
make_guidance_wallflower.py -- ADE20K semantic guidance caches for WallFlower (GPU, Kaggle/Colab)
================================================================================================
NO TRAINING. As in Liang et al. 2021, the segmentation network (DeepLabV3+ or PSPNet) is used
exactly as released, PRE-TRAINED on ADE20K, inference only. WallFlower has one ground-truth
frame per sequence, so fine-tuning on it would be both impossible and test-set leakage.

Same conventions as the CDnet caches you already used (make_guidance.py):
  * foreground = the paper's 12 ADE20K classes (fixed 0-indexed MMSeg ids, below);
  * BGR frame passed straight to mmseg's inference_model (it converts to RGB internally);
  * guidance stored at NATIVE resolution (WallFlower = 160x120).

Output, one file per sequence:  <out_dir>/<Sequence>.npz
    nums : int32   [N]        WallFlower frame numbers (b%05d.bmp)
    guid : uint8   [N,H,W]    0/255, argmax class is one of the 12 foreground classes
    prob : float16 [N,H,W]    max softmax probability over the 12 foreground classes, [0,1]
    meta : str                json: model, frames mode, fg ids, script.txt windows
plus  <out_dir>/check/<Sequence>_eval.png   (input | guidance | ground truth, for a visual check)
and   <out_dir>/guidance_summary.txt       (guidance-only F on the evaluated frame, per sequence)

--frames
  window (default) : from the first official Train frame to the Evaluate frame (train + run + eval)
                     -> 3,693 frames in total, enough for every experiment we plan on WallFlower
  run              : Run frames + Evaluate frame only (2,292 frames) -- the minimum
  all              : every frame of every sequence (16,158 frames) -- only if you want guidance
                     during the LightSwitch lights-on model (frames 0-811) as well

The script RESUMES: a sequence whose .npz already exists is skipped.
--dry_run replaces the network by an all-background predictor (tests I/O without a GPU).
"""
import os, re, json, time, argparse
import numpy as np, cv2

ADE20K_FG_CLASSES = {12: 'person', 20: 'car', 39: 'cushion', 41: 'box', 67: 'book',
                     76: 'boat', 80: 'bus', 83: 'truck', 98: 'bottle', 102: 'van',
                     115: 'bag', 127: 'bicycle'}
FG_IDS = np.array(sorted(ADE20K_FG_CLASSES.keys()), dtype=np.int64)
SEQS = ['Bootstrap', 'Camouflage', 'ForegroundAperture', 'LightSwitch', 'TimeOfDay', 'WavingTrees', 'MovedObject']


# ----------------------------------------------------------------------------- WallFlower layout
def find_sequences(root):
    """Walk `root` and return {sequence_name: folder} for every folder holding script.txt + b*.bmp."""
    found = {}
    for dp, dn, fn in os.walk(root, followlinks=True):
        if 'script.txt' in fn and any(f.lower().startswith('b') and f.lower().endswith('.bmp') for f in fn):
            found[os.path.basename(dp.rstrip('/'))] = dp
    return found


def frame_path(d, n):
    for pat in (f'b{n:05d}.bmp', f'b{n:05d}.BMP'):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            return p
    return None


def all_frames(d):
    return sorted(int(m.group(1)) for f in os.listdir(d) for m in [re.match(r'b(\d{5})\.bmp$', f, re.I)] if m)


def script_windows(d):
    """Official protocol from script.txt -> (train_nums, run_nums, eval_idx). Same parser as wallflower_eval.py."""
    txt = open(os.path.join(d, 'script.txt')).read()
    tr = re.search(r'(\d+)\s*-\s*(\d+)\s+Train', txt)
    rn = re.search(r'(\d+)\s*-\s*(\d+)\s+Run', txt)
    ev = re.search(r'(\d+)\s+Evaluate', txt)
    a, b = int(tr.group(1)), int(tr.group(2))
    r0, r1 = (int(rn.group(1)), int(rn.group(2))) if rn else (b + 1, int(ev.group(1)))
    have = set(all_frames(d))
    train = [n for n in range(a, b + 1) if n in have]
    run = [n for n in range(max(r0, b + 1), r1) if n in have]
    return train, run, int(ev.group(1))


def load_gt(d):
    for f in os.listdir(d):
        if f.lower().startswith('hand_segmented'):
            return cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
    return None


# ----------------------------------------------------------------------------- model
class MMSegGuidance:
    def __init__(self, config, ckpt):
        import torch
        from mmseg.apis import init_model
        from mmseg.utils import register_all_modules
        register_all_modules()
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = init_model(config, ckpt, device=self.dev)
        self.model.eval()
        self.fg = torch.as_tensor(FG_IDS, device=self.dev)

    def __call__(self, frames_bgr):
        from mmseg.apis import inference_model
        out = []
        with self.torch.no_grad():
            res = inference_model(self.model, frames_bgr)
        if not isinstance(res, (list, tuple)):
            res = [res]
        for im, r in zip(frames_bgr, res):
            H, W = im.shape[:2]
            pred = r.pred_sem_seg.data.squeeze().cpu().numpy()
            if pred.ndim > 2:
                pred = pred[0]
            mask = np.isin(pred, FG_IDS).astype(np.uint8) * 255
            try:
                probs = self.torch.softmax(r.seg_logits.data.float(), dim=0)
                prob = probs[self.fg].amax(0).cpu().numpy().astype(np.float32)
            except Exception:
                prob = (mask > 0).astype(np.float32)
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if prob.shape != (H, W):
                prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            out.append((mask, prob))
        return out


class DryRun:
    dev = 'none (dry run)'
    def __call__(self, frames_bgr):
        return [(np.zeros(f.shape[:2], np.uint8), np.zeros(f.shape[:2], np.float32)) for f in frames_bgr]


# ----------------------------------------------------------------------------- main
def f_measure(mask, gt):
    fg, g = mask > 127, gt > 127
    tp = int((fg & g).sum()); fp = int((fg & ~g).sum()); fn = int((~fg & g).sum()); tn = int((~fg & ~g).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return f, p, r, spec


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--wf_root', default='/kaggle/input', help='folder that CONTAINS the WallFlower sequences (searched recursively)')
    ap.add_argument('--config', default=None, help='mmseg config .py (from `mim download`)')
    ap.add_argument('--ckpt', default=None, help='mmseg checkpoint .pth')
    ap.add_argument('--name', default='deeplabv3plus', help='label stored in the cache metadata')
    ap.add_argument('--out_dir', default='wf_guidance_deeplabv3plus')
    ap.add_argument('--frames', choices=['window', 'run', 'all'], default='window')
    ap.add_argument('--seqs', default=','.join(SEQS))
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()

    seqmap = find_sequences(a.wf_root)
    want = [s.strip() for s in a.seqs.split(',') if s.strip()]
    missing = [s for s in want if s not in seqmap]
    print(f'WallFlower sequences found under {a.wf_root}: {sorted(seqmap)}')
    if missing:
        raise SystemExit(f'Missing sequences: {missing}. Point --wf_root at the folder that contains them.')

    os.makedirs(os.path.join(a.out_dir, 'check'), exist_ok=True)
    net = DryRun() if a.dry_run else MMSegGuidance(a.config, a.ckpt)
    print(f'device = {net.dev}   model = {a.name}   frames = {a.frames}   foreground ids = {FG_IDS.tolist()}')

    summary = []
    t_all = time.time(); n_all = 0
    for seq in want:
        d = seqmap[seq]
        out = os.path.join(a.out_dir, f'{seq}.npz')
        tr, run, ev = script_windows(d)
        if a.frames == 'window':
            nums = [n for n in all_frames(d) if tr[0] <= n <= ev]
        elif a.frames == 'run':
            nums = run + [ev]
        else:
            nums = all_frames(d)
        if os.path.exists(out):
            print(f'[skip] {seq}: {out} already exists'); continue
        t0 = time.time()
        G, P = [], []
        for i in range(0, len(nums), a.batch):
            chunk = nums[i:i + a.batch]
            ims = [cv2.imread(frame_path(d, n)) for n in chunk]
            for m, p in net(ims):
                G.append(m); P.append(p.astype(np.float16))
            if (i // a.batch) % 25 == 0:
                print(f'  {seq}: {min(i + a.batch, len(nums))}/{len(nums)}  ({(min(i + a.batch, len(nums))) / max(time.time() - t0, 1e-6):.1f} frames/s)', flush=True)
        meta = dict(sequence=seq, model=a.name, config=a.config, ckpt=a.ckpt, frames=a.frames,
                    fg_ids=FG_IDS.tolist(), fg_classes=ADE20K_FG_CLASSES, train=[tr[0], tr[-1]],
                    run=[run[0], run[-1]] if run else None, eval=ev, native_hw=list(G[0].shape))
        np.savez_compressed(out, nums=np.array(nums, np.int32), guid=np.stack(G), prob=np.stack(P),
                            meta=np.array(json.dumps(meta)))
        n_all += len(nums)

        # sanity check on the evaluated frame (guidance ALONE, no CPB)
        gt = load_gt(d)
        k = nums.index(ev)
        f, p, r, spec = f_measure(G[k], gt)
        line = (f'{seq:20s} N={len(nums):5d}  {time.time() - t0:6.1f}s  eval frame {ev}: guidance-only '
                + (f'specificity={spec:.3f}' if seq == 'MovedObject' else f'F={f:.3f} P={p:.2f} R={r:.2f}')
                + f'  guidance fg {100 * np.mean(G[k] > 127):.1f}%  GT fg {100 * np.mean(gt > 127):.1f}%')
        print('[saved] ' + line, flush=True)
        summary.append(line)
        im = cv2.imread(frame_path(d, ev))
        panel = np.hstack([im, cv2.cvtColor(G[k], cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(os.path.join(a.out_dir, 'check', f'{seq}_eval.png'), cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))

    with open(os.path.join(a.out_dir, 'guidance_summary.txt'), 'a') as fh:
        fh.write(f'# model={a.name} frames={a.frames} config={a.config}\n' + '\n'.join(summary) + '\n')
    el = time.time() - t_all
    print(f'\nDONE: {n_all} frames in {el / 60:.1f} min ({n_all / max(el, 1e-6):.1f} frames/s). Caches in {os.path.abspath(a.out_dir)}')


if __name__ == '__main__':
    main()
