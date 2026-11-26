#!/usr/bin/env python3
"""LIMU loader (rebuilt). Frames image%05d.jpg / image_%04d.jpg, GT *_gt.bmp every 15 frames from 500, white = foreground.
Processing size WORK = (W, H) = (160, 120) (INTER_AREA); GT is scored at native 320x240."""
import os, re, glob
import numpy as np, cv2

WORK = (160, 120)


class Seq:
    def __init__(self, root, name):
        self.d = os.path.join(root, name); self.name = name
        self.frames, self.gts = {}, {}
        for p in glob.glob(self.d + '/*.jpg'):
            self.frames[int(re.findall(r'(\d+)', os.path.basename(p))[-1])] = p
        for p in glob.glob(self.d + '/*_gt.bmp'):
            self.gts[int(re.findall(r'(\d+)_gt', os.path.basename(p))[-1])] = p
        self.N = max(self.frames) + 1

    def rgb(self, n, work=True):
        im = cv2.cvtColor(cv2.imread(self.frames[n]), cv2.COLOR_BGR2RGB)
        if work:
            im = cv2.resize(im, WORK, interpolation=cv2.INTER_AREA)
        return im.astype(np.float32)

    def gt(self, n):
        return cv2.imread(self.gts[n], 0) > 127
