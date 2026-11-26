"""Equivalence test: authors' ZBS main.py (image mode) vs our zbs_post.zbs_sequence on identical synthetic detections."""
import os, sys, types, runpy, shutil, numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ZBS = os.environ.get('ZBS_SRC', 'ZBS')   # clone of https://github.com/CASIA-LMC-Lab/ZBS
sys.path[:0] = [HERE, ZBS, ZBS + '/third_party/CenterNet2']
from detectron2.structures import Instances, Boxes
from pycocotools import mask as mask_util
import zbs_post as ZP

H, W, T, NI = 120, 160, 70, 40
def synth(seed):
    rng = np.random.default_rng(seed); frames = []
    for t in range(T):
        objs = [((20, 30, 60, 70), 3, 0.9)]                                   # static object, class 3
        x = 5 + 2 * t; objs.append(((x, 50, x + 20, 100), 7, 0.8))          # moving object, class 7
        if t > 20: objs.append(((100, 10, 130, 40), 7, 0.7))                # appears later, static, class 7
        objs.append(((22, 31, 61, 69), 3, 0.45))                             # weaker duplicate near the static one
        for _ in range(rng.integers(0, 4)):                                  # clutter
            a, b = rng.integers(0, W - 30), rng.integers(0, H - 30); w_, h_ = rng.integers(8, 30, 2)
            objs.append(((a, b, a + w_, b + h_), int(rng.choice([3, 7, 9])), float(rng.uniform(0.3, 0.95))))
        bx = np.array([o[0] for o in objs], np.float32) + rng.normal(0, 0.7, (len(objs), 4)).astype(np.float32)
        bx = np.clip(bx, 0, [W - 1, H - 1, W - 1, H - 1])
        ms = np.zeros((len(objs), H, W), bool)
        for i, b in enumerate(bx.astype(int)): ms[i, b[1]:b[3], b[0]:b[2]] = True
        frames.append(dict(n=t, role='init' if t < NI else 'score', h=H, w=W, boxes=bx,
                           scores=np.array([o[2] for o in objs], np.float32), classes=np.array([o[1] for o in objs], np.int16),
                           rle=[mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms], _masks=ms))
    return frames

def run_authors(frames, flags, tmp):
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp + '/input'); os.makedirs(tmp + '/groundtruth')
    cv2.imwrite(tmp + '/ROI.bmp', np.full((H, W), 255, np.uint8))
    for fr in frames:
        cv2.imwrite(f"{tmp}/input/in{fr['n'] + 1:06d}.jpg", np.zeros((H, W, 3), np.uint8))
        cv2.imwrite(f"{tmp}/groundtruth/gt{fr['n'] + 1:06d}.png", np.full((H, W), 85 if fr['role'] == 'init' else 0, np.uint8))
    by_name = {f"in{fr['n'] + 1:06d}.jpg": fr for fr in frames}; state = {'cur': None}
    import detectron2.data.detection_utils as du
    real_read = du.read_image
    def read_image(path, format=None):
        if path.endswith('.jpg'): state['cur'] = by_name[os.path.basename(path)]
        return real_read(path, format=format)
    du.read_image = read_image
    class Demo:
        def __init__(self, cfg, args, parallel=False): self.th = args.confidence_threshold
        def detic_detect(self, img):
            fr = state['cur']; k = fr['scores'] >= self.th - 1e-9
            inst = Instances((H, W)); inst.pred_boxes = Boxes(torch.from_numpy(fr['boxes'][k]))
            inst.scores = torch.from_numpy(fr['scores'][k]); inst.pred_classes = torch.from_numpy(fr['classes'][k].astype(np.int64))
            inst.pred_masks = torch.from_numpy(fr['_masks'][k]); return inst
    fake = types.ModuleType('detic.predictor'); fake.VisualizationDemo = Demo; sys.modules['detic.predictor'] = fake
    import detectron2.utils.visualizer as V
    captured = {}
    class Cap:
        def __init__(self, img): self.img = img
        def save(self, p): captured[os.path.basename(p)] = np.asarray(self.img).copy()
    V.VisImage = Cap
    cwd = os.getcwd(); os.chdir(ZBS)
    sys.argv = ['main.py', '--input-dir', tmp + '/', '--output-dir', '/zbs_out/', '--cpu', '--vocabulary', 'lvis',
                '--config-file', 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'] + flags
    try:
        runpy.run_path(ZBS + '/main.py', run_name='__main__')
    finally:
        os.chdir(cwd); du.read_image = real_read
    return {int(k[2:8]) - 1: v != 0 for k, v in captured.items()}

from sort_zbs import Sort
FLAGS = {'paper': [], 'p1': ['--confidence-threshold', '0.3', '--noiof', '--fore_threshold', '0.3', '--iou_threshold', '0.3', '--max_age', '100'],
         'p3': ['--confidence-threshold', '0.4', '--fore_threshold', '1', '--move_threshold', '0']}
allok = True
for seed in range(3):
    frames = synth(seed)
    for g, fl in FLAGS.items():
        ref = run_authors(frames, fl, os.path.join(HERE, 'tmp_seq'))
        ours = {n: m for n, m, _ in ZP.zbs_sequence(frames, ZP.GROUPS[g], Sort)}
        diff = sum(int((ref[n] != ours[n]).sum()) for n in ours)
        fgpix = sum(int(ours[n].sum()) for n in ours)
        print(f'seed {seed} group {g:5s}: scored frames {len(ours)} (authors {len(ref)}), fg pixels {fgpix}, differing pixels {diff}')
        allok &= (diff == 0 and set(ref) == set(ours))
print('EQUIVALENT' if allok else 'MISMATCH')
