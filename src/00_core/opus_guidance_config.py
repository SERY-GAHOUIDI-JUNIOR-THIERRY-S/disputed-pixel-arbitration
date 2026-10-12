#!/usr/bin/env python3
"""
should run this code first after trained my models , for getting the right guidance because the previous guidance
was wrong missing 8 category over 12 then after that and obtaining my model I can run the entire liang & al framework
================================================================================
ADE20K GUIDANCE (DeepLabV3+ / PSPNet via MMSegmentation) — CORRECTED
================================================================================
Fixes three real bugs in your deeplab_guidance_mmseg.py / pspnet_guidance_mmseg.py:

  BUG 1 (critical): WRONG ADE20K class indices. 8 of your 12 foreground class
        ids pointed at the wrong classes (idx 21 is 'water', not 'cushion';
        33 is 'desk'; 84 is 'tower'; 91 is 'dirt track'; 99 is 'buffet'; ...).
        Only person/car/bus/truck were correct — which is why baseline looked
        fine but the guidance was silently flagging water/desk/etc. as
        foreground and MISSING real boat/bottle/van/bag/bicycle. Corrected to
        the paper's 12 classes with the canonical 0-indexed MMSeg ADE20K ids.

  BUG 2: COLOR CONVENTION. MMSeg's data_preprocessor for ADE20K uses
        bgr_to_rgb=True, i.e. inference_model expects a BGR ndarray (cv2 native)
        and converts internally. Your code did cv2.cvtColor(frame, BGR2RGB)
        BEFORE inference_model -> the network saw swapped channels. Fixed:
        pass the BGR frame straight to inference_model.

  BUG 3: SILENT EDGE-DETECTION FALLBACK. If the checkpoint fails to load, your
        code silently returns a Canny-edge "guidance", which is NOT a semantic
        model and must never appear in reported results. Here the fallback is
        OFF by default; predict() raises unless allow_fallback=True.
        (Note: your PSPNet was running on this fallback — its numbers are not
        real PSPNet guidance. DeepLabV3+ loads, so use it as the working model.)

Also: guidance is returned at the INPUT frame's native resolution (out_size=None)
so it lines up with the CPB pipeline, which then resizes to the model fit size.

The 12 paper foreground classes (§4.1): person, car, cushion, box, book, boat,
bus, truck, bottle, van, bag, bicycle.
================================================================================
"""
import os
import cv2
import numpy as np

try:
    import torch
    from mmseg.apis import init_model, inference_model
    from mmseg.utils import register_all_modules
    MMSEG_AVAILABLE = True
except Exception:
    MMSEG_AVAILABLE = False

# CORRECT 0-indexed MMSeg ADE20K ids for the paper's 12 foreground classes
ADE20K_FG_CLASSES = {
    12: 'person', 20: 'car', 39: 'cushion', 41: 'box', 67: 'book', 76: 'boat',
    80: 'bus', 83: 'truck', 98: 'bottle', 102: 'van', 115: 'bag', 127: 'bicycle',
}
FG_IDS = np.array(sorted(ADE20K_FG_CLASSES.keys()), dtype=np.int64)

_MODELS = {
    'deeplabv3plus': dict(
        config='deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py',
        ckpt='deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth',
        name='DeepLabV3+_R101'),
    'pspnet': dict(
        config='pspnet_r50-d8_4xb4-80k_ade20k-512x512.py',
        ckpt='pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth',
        name='PSPNet_R50'),
}


class ADE20KGuidance:
    def __init__(self, backbone='deeplabv3plus', device='cpu',
                 models_dir='./models', smooth=False, allow_fallback=False):
        assert backbone in _MODELS, f'unknown backbone {backbone}'
        self.backbone = backbone
        self.name = _MODELS[backbone]['name']
        self.device = device
        self.models_dir = models_dir
        self.smooth = smooth
        self.allow_fallback = allow_fallback
        self.model = None
        self._last = None
        self._load()

    def _resolve(self, fname):
        for base in (os.path.join(self.models_dir, 'models'), self.models_dir, './models'):
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        return None

    def _load(self):
        if not MMSEG_AVAILABLE:
            print('  [Guidance] MMSegmentation not importable.')
            return
        cfg = self._resolve(_MODELS[self.backbone]['config'])
        ckpt = self._resolve(_MODELS[self.backbone]['ckpt'])
        if cfg is None or ckpt is None:
            print(f'  [Guidance] {self.name}: config/checkpoint not found under {self.models_dir}')
            return
        try:
            register_all_modules()
            self.model = init_model(cfg, ckpt, device=self.device)
            self.model.eval()
            print(f'  [Guidance] {self.name} loaded ({os.path.getsize(ckpt)/1e6:.0f} MB)')
        except Exception as e:
            print(f'  [Guidance] {self.name} failed to load: {e}')
            self.model = None

    # ---- core API -----------------------------------------------------------
    def predict(self, frame_bgr, out_hw=None):
        """
        frame_bgr: HxWx3 BGR uint8 (cv2 native, as from cv2.imread).
        Returns a 0/255 uint8 foreground mask at the frame's native size
        (or out_hw=(H,W) if given).
        """
        H, W = frame_bgr.shape[:2]
        if self.model is None:
            if not self.allow_fallback:
                raise RuntimeError(
                    f'{self.name} is not loaded and allow_fallback=False. '
                    'Refusing to return an edge-detection mask for a reported run.')
            mask = self._edge_fallback(frame_bgr)
        else:
            with torch.no_grad():
                res = inference_model(self.model, frame_bgr)   # BGR in (bgr_to_rgb handled internally)
            pred = res.pred_sem_seg.data.squeeze().cpu().numpy()
            if pred.ndim > 2:
                pred = pred[0]
            mask = np.isin(pred, FG_IDS).astype(np.uint8) * 255

        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if out_hw is not None and mask.shape[:2] != tuple(out_hw):
            mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)

        if self.smooth and self._last is not None and self._last.shape == mask.shape:
            mask = np.where((self._last > 127) & (mask > 127), 255, mask).astype(np.uint8)
        self._last = mask.copy()
        return mask

    def _edge_fallback(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        e = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
        return (e > 0).astype(np.uint8) * 255


if __name__ == '__main__':
    print('Corrected ADE20K foreground ids:', ADE20K_FG_CLASSES)
    g = ADE20KGuidance(backbone='deeplabv3plus', allow_fallback=True)
    im = cv2.imread('data/cdnet2014/baseline/highway/input/in001050.jpg')
    if im is not None:
        m = g.predict(im)
        print('mask', m.shape, 'FG%%=%.2f' % (100 * (m > 0).mean()))
        cv2.imwrite('guidance_check.png', m)
        print('wrote guidance_check.png — overlay it: cars should light up, road should not')
