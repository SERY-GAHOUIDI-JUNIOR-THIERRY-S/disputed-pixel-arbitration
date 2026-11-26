#!/usr/bin/env python3
"""
DENSE ADE20K guidance caches for CDnet2014 (GPU), using YOUR corrected setup:
  * the paper's 12 foreground classes (fixed ADE20K ids),
  * BGR frame passed straight to inference_model (mmseg converts internally),
  * DeepLabV3+ (your working model) or PSPNet,
  * guidance at the frame's NATIVE resolution.
Writes, per scene, <out_dir>/<category>__<scene>.npz with:
  nums uint32[N], guid uint8[N,H,W] (0/255), prob float16[N,H,W] in [0,1]
— the exact format the CPB framework reads. Runs over the FULL temporalROI range.
"""
import os, re, argparse
import numpy as np, cv2, torch
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

# The paper's 12 foreground classes, canonical 0-indexed MMSeg ADE20K ids
ADE20K_FG_CLASSES = {12:'person',20:'car',39:'cushion',41:'box',67:'book',76:'boat',
                     80:'bus',83:'truck',98:'bottle',102:'van',115:'bag',127:'bicycle'}
FG_IDS = np.array(sorted(ADE20K_FG_CLASSES.keys()), dtype=np.int64)

def list_nums(d):
    ns = []
    for fn in os.listdir(d):
        if fn.lower().endswith(('.jpg', '.png')):
            m = re.findall(r'\d+', fn)
            if m: ns.append(int(m[-1]))
    return sorted(ns)

def temporal_roi(sp):
    a, b = open(os.path.join(sp, 'temporalROI.txt')).read().split()[:2]; return int(a), int(b)

def load_frame(idir, n):
    for pat in (f'in{n:06d}.jpg', f'in{n:05d}.jpg', f'{n:06d}.jpg', f'in{n:06d}.png'):
        p = os.path.join(idir, pat)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None: return im
    return None

def predict(model, frame_bgr):
    H, W = frame_bgr.shape[:2]
    res = inference_model(model, frame_bgr)                 # BGR in (bgr_to_rgb handled internally)
    pred = res.pred_sem_seg.data.squeeze().cpu().numpy()
    if pred.ndim > 2: pred = pred[0]
    mask = np.isin(pred, FG_IDS).astype(np.uint8) * 255
    try:
        logits = res.seg_logits.data                        # [C,h,w]
        probs = torch.softmax(logits.float(), dim=0)
        fg = probs[torch.as_tensor(FG_IDS, device=probs.device)].amax(0).cpu().numpy().astype(np.float32)
    except Exception:
        fg = (mask > 0).astype(np.float32)                  # fallback: binary prob
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    if fg.shape[:2] != (H, W):
        fg = cv2.resize(fg, (W, H))
    return mask, fg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--config', required=True, help='path to the mmseg config .py')
    ap.add_argument('--ckpt', required=True, help='path to the .pth checkpoint')
    ap.add_argument('--out_dir', default='cdnet_guidance_pspnet')
    ap.add_argument('--categories', default=None)
    ap.add_argument('--scenes', default=None)
    ap.add_argument('--stride', type=int, default=1, help='1 = every temporalROI frame')
    ap.add_argument('--max_dim', type=int, default=0, help='0 = native (matches your caches); else cap longest side')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    register_all_modules()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={dev}  config={args.config}')
    model = init_model(args.config, args.ckpt, device=dev); model.eval()
    print('model loaded OK; FG ids =', FG_IDS.tolist())

    root = args.dataset_root
    if args.scenes:
        sids = [s.strip() for s in args.scenes.split(',')]
    else:
        cats = [c.strip() for c in args.categories.split(',')] if args.categories else \
               sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        sids = [f'{c}/{s}' for c in cats
                for s in sorted(os.listdir(os.path.join(root, c)))
                if os.path.isdir(os.path.join(root, c, s, 'input'))]

    for sid in sids:
        cat, scene = sid.split('/')
        sp = os.path.join(root, cat, scene); idir = os.path.join(sp, 'input')
        out = os.path.join(args.out_dir, f'{cat}__{scene}.npz')
        if os.path.exists(out):
            print(f'[skip] {sid} exists'); continue
        a, b = temporal_roi(sp)
        nums = [n for n in list_nums(idir) if a <= n <= b][::args.stride]
        G, P, K = [], [], []
        with torch.no_grad():
            for i, n in enumerate(nums):
                im = load_frame(idir, n)
                if im is None: continue
                mask, fg = predict(model, im)
                if args.max_dim and args.max_dim > 0:
                    h, w = mask.shape; sc = args.max_dim / max(h, w)
                    if sc < 1.0:
                        nw, nh = int(round(w*sc)), int(round(h*sc))
                        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                        fg = cv2.resize(fg, (nw, nh))
                G.append(mask.astype(np.uint8)); P.append(fg.astype(np.float16)); K.append(n)
                if i % 200 == 0: print(f'  {sid} {i}/{len(nums)}')
        if not K:
            print(f'[warn] {sid} no frames'); continue
        np.savez_compressed(out, nums=np.array(K, np.int32), guid=np.stack(G), prob=np.stack(P))
        print(f'[saved] {out}  N={len(K)}  {os.path.getsize(out)/1e6:.0f} MB')

if __name__ == '__main__':
    main()
