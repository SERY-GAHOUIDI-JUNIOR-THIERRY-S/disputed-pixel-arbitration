#!/usr/bin/env python3
"""
make_guidance_limu.py -- ADE20K semantic guidance caches for LIMU (GPU, Kaggle/Colab). Inference only, no training.
Same conventions as the CDnet / WallFlower caches: the paper's 12 ADE20K foreground classes, BGR frame passed to
mmseg inference_model, guidance at native resolution (LIMU = 320x240).
Frames: every frame from --start (default 400 = first frame after the CPB training window) to the end.
Output per sequence <out_dir>/<Sequence>.npz:
  nums int32[N] | guid uint8[N,H,W] 0/255 | prob float16[N,H,W] = max softmax over the 12 fg classes | meta (json)
Sequences are found by searching --limu_root for folders named CameraParameter, Intersection, LightSwitch.
Finished sequences are skipped (resume).
"""
import os, re, json, argparse, glob
import numpy as np, cv2, torch
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

ADE20K_FG = {12: 'person', 20: 'car', 39: 'cushion', 41: 'box', 67: 'book', 76: 'boat', 80: 'bus', 83: 'truck',
             98: 'bottle', 102: 'van', 115: 'bag', 127: 'bicycle'}
FG_IDS = np.array(sorted(ADE20K_FG), np.int64)
SEQS = ['CameraParameter', 'Intersection', 'LightSwitch']


def find_seq(root, name):
    for d, dirs, files in os.walk(root):
        if os.path.basename(d) == name and any(f.lower().endswith('.jpg') for f in files):
            return d
    return None


def predict(model, bgr):
    H, W = bgr.shape[:2]
    res = inference_model(model, bgr)
    pred = res.pred_sem_seg.data.squeeze().cpu().numpy()
    mask = np.isin(pred, FG_IDS).astype(np.uint8) * 255
    probs = torch.softmax(res.seg_logits.data.float(), dim=0)
    fg = probs[torch.as_tensor(FG_IDS, device=probs.device)].amax(0).cpu().numpy().astype(np.float32)
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST); fg = cv2.resize(fg, (W, H))
    return mask, fg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limu_root', required=True); ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', default='pspnet_r50_ade20k'); ap.add_argument('--out_dir', default='limu_guidance_pspnet')
    ap.add_argument('--start', type=int, default=400); ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    found = {s: find_seq(a.limu_root, s) for s in SEQS}
    print('sequences found:', found)
    if a.dry_run:
        return
    register_all_modules(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(a.config, a.ckpt, device=dev); model.eval(); print('model loaded on', dev)
    for s, d in found.items():
        out = os.path.join(a.out_dir, f'{s}.npz')
        if d is None:
            print(f'[warn] {s} not found'); continue
        if os.path.exists(out):
            print(f'[skip] {s} exists'); continue
        files = {}
        for p in glob.glob(os.path.join(d, '*.jpg')):
            files[int(re.findall(r'(\d+)', os.path.basename(p))[-1])] = p
        nums = [n for n in sorted(files) if n >= a.start]
        G, Pr, K = [], [], []
        with torch.no_grad():
            for i, n in enumerate(nums):
                m, f = predict(model, cv2.imread(files[n]))
                G.append(m); Pr.append(f.astype(np.float16)); K.append(n)
                if i % 500 == 0:
                    print(f'  {s} {i}/{len(nums)}', flush=True)
        meta = json.dumps(dict(sequence=s, model=a.name, config=os.path.basename(a.config), ckpt=os.path.basename(a.ckpt),
                               frames=f'{a.start}-end', fg_ids=FG_IDS.tolist(), fg_classes=ADE20K_FG, prob='float16 max fg-class softmax'))
        np.savez_compressed(out, nums=np.array(K, np.int32), guid=np.stack(G), prob=np.stack(Pr), meta=np.array(meta))
        print(f'[saved] {out}  N={len(K)}  {os.path.getsize(out) / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
