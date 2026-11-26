#!/usr/bin/env python3
"""
CDnet 2014 -- locked robust CPB, frozen and/or online, with saved models, progress bars and every metric in the log.

Protocol (fixed before any result):
  * every video folder that contains temporalROI.txt and input/ (inXXXXXX.jpg), groundtruth/ (gtXXXXXX.png)
  * training frames = frames BEFORE the temporal ROI, up to --max_train evenly sampled; no ground truth used
    (fewer than --min_pre pre-ROI frames -> the first --min_pre frames are used and the video is flagged)
  * locked CPB (= wf_frame_switch.py --preset final): per-channel RGB models, consistent + robust training, frame guard 0.5,
    MAD sigma, floor 3, eta 2.5, lambda 0.5, K 20, 3x3 median, OR over channels, frame-level switching (trigger 0.7)
  * frozen = no update ; online = causal update after every frame (alpha 0.01, sigma frozen), as on WallFlower
  * working size: s = min(1, sqrt(--budget / (W*H))), sides rounded to multiples of 8; masks upsampled (nearest) for scoring
  * scoring labels as in the official CDnet tool: 255 = fg; 0 and 50 (shadow) = bg; 85 and 170 ignored; ROI frames only

Folders (all under --out, default cdnet_runs):
  trained_models/<cat>__<video>.npz (+ .json)  trained CPB, shared by every mode and by test/full runs -> trained ONCE
  frozen[_testN]/  counts/*.csv  info/*.json  log.txt  summary.txt  summary.csv
  online_a0.01[_testN]/  same  + final_models/*.npz (the model after the last online update, --save_final 1)
--mode both runs frozen and online in ONE pass over the frames (each frame read once, one training).
Finished videos are skipped, so a stopped run resumes where it stopped (Ctrl+C is safe).

Examples
  python cdnet_cpb_eval.py --root "<CDnet folder>" --list
  python -u cdnet_cpb_eval.py --root "<CDnet folder>" --mode frozen --max_frames 60      (quick test -> cdnet_runs/frozen_test60)
  python -u cdnet_cpb_eval.py --root "<CDnet folder>" --mode frozen                      (full run   -> cdnet_runs/frozen)
  python -u cdnet_cpb_eval.py --root "<CDnet folder>" --mode online                      (reuses the trained models)
  python -u cdnet_cpb_eval.py --root "<CDnet folder>" --mode both
"""
import sys, os, re, csv, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import wf_frame_switch as S
import cdnet_metrics as CM

LOGS = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    for fh in LOGS:
        fh.write(s + '\n'); fh.flush()


class Bar:
    """Plain-text progress bar (no extra package); redraws one line with percent, rate and ETA."""
    def __init__(self, total, label, width=20):
        self.total, self.label, self.width, self.n, self.t0, self.last = max(int(total), 1), label, width, 0, time.time(), 0.0

    def update(self, k=1, extra=''):
        self.n += k
        now = time.time()
        if now - self.last < 0.5 and self.n < self.total:
            return
        self.last = now
        f = min(self.n / self.total, 1.0); el = now - self.t0
        eta = el / f - el if f > 0 else 0
        bar = '#' * int(self.width * f) + '-' * (self.width - int(self.width * f))
        sys.stdout.write(f'\r  {self.label[:16]:16s} [{bar}] {100 * f:3.0f}% {self.n}/{self.total} '
                         f'{self.n / max(el, 1e-9):4.1f}fr/s ETA {eta / 60:4.1f}m ')
        sys.stdout.flush()

    def close(self):
        self.n = self.total - 1; self.last = 0; self.update(1); sys.stdout.write('\n'); sys.stdout.flush()


def find_videos(root):
    vids = []
    for d, dirs, files in os.walk(root, followlinks=True):
        if 'temporalROI.txt' in files and os.path.isdir(os.path.join(d, 'input')):
            vids.append(d)
    return sorted(vids)


def work_size(W, H, budget):
    s = min(1.0, np.sqrt(budget / float(W * H)))
    return max(8, int(round(W * s / 8.0)) * 8), max(8, int(round(H * s / 8.0)) * 8)


def locked_args(alpha=0.0):
    a = argparse.Namespace()
    a.trainer, a.robust, a.robust_B, a.frame_guard, a.sig_all = 'consistent', 1, -1, 0.5, 0
    a.robust_sigma, a.block, a.selection, a.gamma = 'mad', 'median', 'minstd', 'matlab'
    a.af, a.floor, a.bth, a.ref, a.median = 2.5, 3.0, 7.0, 20, 1
    a.alpha, a.alpha_sig = alpha, 0.0
    a.switch, a.trigger, a.state_diff, a.state_frac, a.min_cluster, a.max_train = True, 0.7, 25.0, 0.5, 50, 200
    return a


# ----------------------------------------------------------------------------- safe file writing
def safe_replace(tmp, path, tries=20):
    """os.replace with retries: on Windows a just-written file can be locked for a moment by antivirus/indexing."""
    for i in range(tries):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            time.sleep(0.5 + 0.25 * i)
    raise PermissionError(f'could not write {path}: the file is locked by another process. Make sure only ONE '
                          f'cdnet_cpb_eval.py is running (close other terminals / the PyCharm Run window).')


def tmp_name(path):
    return f'{path}.{os.getpid()}.tmp'


class VideoLock:
    """Prevents two running copies of the script from processing the same video in the same mode folder."""
    def __init__(self, path):
        self.path, self.ok = path, False

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            self.ok = True
        except FileExistsError:
            self.ok = False
        return self

    def __exit__(self, *exc):
        if self.ok:
            try:
                os.remove(self.path)
            except OSError:
                pass
        return False


# ----------------------------------------------------------------------------- model save / load
def save_models(models, path, meta):
    arr = {}
    for i, M in enumerate(models):
        for c, m in enumerate(M.models):
            arr[f'm{i}_c{c}'] = np.ascontiguousarray(m['DBLK_r'], np.float32)
    meta = dict(meta, n_models=len(models), names=[M.name for M in models], img_size=list(models[0].models[0]['img_size']))
    tmp = tmp_name(path)
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arr)
    safe_replace(tmp, path)
    tj = tmp_name(path[:-4] + '.json')
    with open(tj, 'w') as fh:
        json.dump(meta, fh, indent=1)
    safe_replace(tj, path[:-4] + '.json')


def load_models(path):
    meta = json.load(open(path[:-4] + '.json'))
    z = np.load(path)
    H, W = meta['img_size']; models = []
    for i in range(meta['n_models']):
        M = S.RGBModel.__new__(S.RGBModel); M.name = meta['names'][i]; M.models = []
        for c in range(3):
            D = np.array(z[f'm{i}_c{c}'], np.float32, copy=True)
            M.models.append({'DBLK_r': D, 'DBLK_g': D, 'DBLK_b': D, 'img_size': (H, W), 'params': {'poolnum': D.shape[2]}})
        models.append(M)
    return models, meta


def copy_models(models):
    out = []
    for M in models:
        N = S.RGBModel.__new__(S.RGBModel); N.name = M.name; N.models = []
        for m in M.models:
            D = np.array(m['DBLK_r'], np.float32, copy=True)
            N.models.append({'DBLK_r': D, 'DBLK_g': D, 'DBLK_b': D, 'img_size': m['img_size'], 'params': dict(m['params'])})
        out.append(N)
    return out


def protocol(a):
    return dict(max_train=a.max_train, min_pre=a.min_pre, budget=a.budget, cpb='locked robust CPB (wf_frame_switch --preset final)')


# ----------------------------------------------------------------------------- one video
def train_or_load(vd, tag, a, rd, nums, r0, ww, wh):
    mp = os.path.join(a.out, 'trained_models', tag + '.npz')
    if os.path.exists(mp) and os.path.exists(mp[:-4] + '.json'):
        models, meta = load_models(mp)
        if meta.get('protocol') == protocol(a) and meta.get('work') == [ww, wh]:
            return models, meta, True
        P(f'  saved model {tag} was made with another protocol -> retraining')
    pre = [n for n in nums if n < r0]; flag = ''
    if len(pre) < a.min_pre:
        flag = f'only {len(pre)} pre-ROI frames -> trained on the first {a.min_pre} frames (overlaps the ROI)'
        pre = nums[:a.min_pre]
    tr = sorted(set(pre[i] for i in np.linspace(0, len(pre) - 1, min(a.max_train, len(pre))).round().astype(int)))
    bar = Bar(len(tr), 'reading train frames'); frames = []
    for n in tr:
        frames.append(rd(n)); bar.update()
    bar.close()
    L = locked_args()
    t0 = time.time(); P(f'  training CPB on {len(tr)} frames ...')
    A = S.RGBModel(frames, L, 'A (pre-ROI)'); models = [A]
    feats = np.stack([cv2.resize(f, (40, 30), interpolation=cv2.INTER_AREA).ravel() for f in frames])
    lab, C = S.kmeans2(feats); cur = lab[-1]
    other = [i for i in range(len(tr)) if lab[i] != cur]
    cimg = C.reshape(2, 30, 40, 3).mean(3); frac = float(np.mean(np.abs(cimg[0] - cimg[1]) > L.state_diff))
    second = bool(frac > L.state_frac and len(other) >= L.min_cluster)
    if second:
        sel = [other[i] for i in np.linspace(0, len(other) - 1, min(L.max_train, len(other))).round().astype(int)]
        models.append(S.RGBModel([frames[i] for i in sel], L, 'B (other state)'))
    meta = dict(protocol=protocol(a), work=[ww, wh], n_train=len(tr), train_frames=[int(x) for x in tr],
                n_pre=len([n for n in nums if n < r0]), second_state=second, flag=flag, t_train=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    save_models(models, mp, meta)
    P(f'  trained in {meta["t_train"]:.0f}s, saved -> {os.path.relpath(mp)}')
    return models, meta, False


class Runner:
    """One mode (frozen or online) on one video: own model copy and own switching state."""
    def __init__(self, name, models, alpha):
        self.name, self.models, self.L, self.active, self.switches, self.rows = name, models, locked_args(alpha), 0, 0, []

    def step(self, rgb, shape):
        m = self.models[self.active].detect(rgb, shape, self.L)
        if len(self.models) > 1 and np.mean(m > 127) > self.L.trigger:
            fr = [float(np.mean(mm.detect(rgb, shape, self.L) > 127)) for mm in self.models]; b = int(np.argmin(fr))
            if b != self.active:
                self.active, self.switches = b, self.switches + 1
                m = self.models[self.active].detect(rgb, shape, self.L)
        if self.L.alpha > 0:
            self.models[self.active].update(rgb, self.L)     # causal: after the decision on this frame
        return m


def score(m, g):
    pm = cv2.resize((m > 127).astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = (g != 85) & (g != 170); pos = (g == 255) & valid; neg = ((g == 0) | (g == 50)) & valid
    return (int((pm & pos).sum()), int((pm & neg).sum()), int((~pm & pos).sum()), int((~pm & neg).sum()), int(pos.sum()), int(valid.sum()))


def run_video(vd, a, modes, idx, nvid):
    cat, vid = os.path.basename(os.path.dirname(vd)), os.path.basename(vd); tag = f'{cat}__{vid}'
    todo = [(name, d, alpha) for name, d, alpha in modes if not os.path.exists(os.path.join(d, 'counts', tag + '.csv'))]
    if not todo:
        P(f'[{idx}/{nvid}] {tag}: already done in {", ".join(os.path.basename(d) for _, d, _ in modes)} -> skip'); return
    os.makedirs(os.path.join(a.out, 'locks'), exist_ok=True)
    with VideoLock(os.path.join(a.out, 'locks', tag + '.lock')) as lk:
        if not lk.ok:
            P(f'[{idx}/{nvid}] {tag}: LOCKED -- another running copy of this script is processing it -> skip '
              f'(if no other copy is running, delete the folder {os.path.join(a.out, "locks")} and run again)')
            return
        _run_video(vd, a, todo, idx, nvid, cat, vid, tag)


def _run_video(vd, a, todo, idx, nvid, cat, vid, tag):
    t0 = time.time()
    r0, r1 = [int(x) for x in open(os.path.join(vd, 'temporalROI.txt')).read().split()[:2]]
    inp, gtd = os.path.join(vd, 'input'), os.path.join(vd, 'groundtruth')
    nums = sorted(int(re.findall(r'(\d+)', os.path.basename(p))[-1]) for p in glob.glob(os.path.join(inp, 'in*.jpg')))
    path = lambda n: os.path.join(inp, f'in{n:06d}.jpg')
    H, W = cv2.imread(path(nums[0])).shape[:2]; ww, wh = work_size(W, H, a.budget)
    rd = lambda n: cv2.cvtColor(cv2.resize(cv2.imread(path(n)), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    P(f'[{idx}/{nvid}] {tag}  {W}x{H} -> {ww}x{wh}  ROI {r0}-{r1}  modes: {", ".join(n for n, _, _ in todo)}')
    models, meta, loaded = train_or_load(vd, tag, a, rd, nums, r0, ww, wh)
    if loaded:
        P(f'  loaded trained model ({meta["n_train"]} frames, second state {int(meta["second_state"])})')
    runners = [Runner(name, copy_models(models) if alpha > 0 else models, alpha) for name, d, alpha in todo]
    test = [n for n in nums if r0 <= n <= r1]
    if a.max_frames:
        test = test[:a.max_frames]
    if a.stride > 1:                                   # diagnostic only: every n-th ROI frame (valid for frozen runs without switching)
        test = test[::a.stride]
    bar = Bar(len(test), 'detecting ' + '+'.join(r.name for r in runners))
    for n in test:
        rgb = rd(n)
        masks = [r.step(rgb, (wh, ww)) for r in runners]
        gp = os.path.join(gtd, f'gt{n:06d}.png')
        g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE) if os.path.exists(gp) else None
        if g is not None:
            for r, m in zip(runners, masks):
                r.rows.append((n,) + score(m, g))
        bar.update()
    bar.close()
    for r, (name, d, alpha) in zip(runners, todo):
        os.makedirs(os.path.join(d, 'counts'), exist_ok=True); os.makedirs(os.path.join(d, 'info'), exist_ok=True)
        if alpha > 0 and a.save_final:
            os.makedirs(os.path.join(d, 'final_models'), exist_ok=True)
            save_models(r.models, os.path.join(d, 'final_models', tag + '.npz'), dict(meta, alpha=alpha, last_frame=int(test[-1]) if test else None))
        csvp = os.path.join(d, 'counts', tag + '.csv'); tmp = tmp_name(csvp)
        with open(tmp, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['frame', 'tp', 'fp', 'fn', 'tn', 'gt_fg', 'valid']); w.writerows(r.rows)
        safe_replace(tmp, csvp)
        rows = [x for x in r.rows if x[6] > 0]
        vm = CM.video_metrics(rows) if rows else None
        info = dict(meta, mode=name, alpha=alpha, category=cat, video=vid, native=[W, H], roi=[r0, r1], switches=r.switches,
                    n_scored=len(rows), max_frames=a.max_frames, t_video=round(time.time() - t0, 1), model_loaded=loaded)
        if vm:
            info['metrics'] = {k: (float(vm[k]) if np.isfinite(vm[k]) else None) for k in CM.OFFICIAL + CM.EXTRA}
        json.dump(info, open(os.path.join(d, 'info', tag + '.json'), 'w'), indent=1)
        line = CM.video_line(f'  {name:12s} {vid}', vm) if vm else f'  {name:12s} {vid}: no scored frames'
        P(line + (f'  switches {r.switches}' if r.switches else '') + (f'  [{meta["flag"]}]' if meta.get('flag') else ''))
    P(f'  video done in {(time.time() - t0) / 60:.1f} min')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='CDnet 2014 folder (searched recursively for temporalROI.txt)')
    ap.add_argument('--out', default='cdnet_runs', help='base output folder')
    ap.add_argument('--mode', choices=['frozen', 'online', 'both'], default='frozen')
    ap.add_argument('--alpha', type=float, default=0.01, help='online update rate (used by --mode online/both)')
    ap.add_argument('--max_frames', type=int, default=0, help='quick test: only the first N ROI frames of each video (0 = all)')
    ap.add_argument('--stride', type=int, default=1, help='diagnostic: score every n-th ROI frame only (frozen runs)')
    ap.add_argument('--only', default='', help='comma list of categories or category/video (default: all)')
    ap.add_argument('--list', action='store_true', help='only list the videos found')
    ap.add_argument('--save_final', type=int, default=1, help='online: save the model after the last update (1/0)')
    ap.add_argument('--max_train', type=int, default=400); ap.add_argument('--min_pre', type=int, default=50)
    ap.add_argument('--budget', type=float, default=160 * 120, help='working-size pixel budget')
    ap.add_argument('--clear_locks', action='store_true', help='delete stale lock files left by a killed run, then start')
    a = ap.parse_args()
    if a.clear_locks:
        for f in glob.glob(os.path.join(a.out, 'locks', '*.lock')):
            os.remove(f)
        for f in glob.glob(os.path.join(a.out, '**', '*.tmp'), recursive=True):
            os.remove(f)
    vids = find_videos(a.root)
    if a.only:
        keep = [k.replace('\\', '/') for k in a.only.split(',')]
        vids = [v for v in vids if os.path.basename(os.path.dirname(v)) in keep
                or f'{os.path.basename(os.path.dirname(v))}/{os.path.basename(v)}' in keep]
    print(f'{len(vids)} videos found under {os.path.abspath(a.root)}', flush=True)
    if not vids:
        print('  -> no folder with temporalROI.txt + input/ found. --root must contain the category folders '
              '(baseline, badWeather, ...). Check the path and that it has exactly one quote at each end.')
        return
    if a.list:
        for v in vids:
            print('  ' + os.path.relpath(v, a.root))
        return
    sfx = (f'_test{a.max_frames}' if a.max_frames else '') + (f'_stride{a.stride}' if a.stride > 1 else '')
    modes = []
    if a.mode in ('frozen', 'both'):
        modes.append(('frozen', os.path.join(a.out, 'frozen' + sfx), 0.0))
    if a.mode in ('online', 'both'):
        modes.append((f'online_a{a.alpha:g}', os.path.join(a.out, f'online_a{a.alpha:g}' + sfx), a.alpha))
    for _, d, _ in modes:
        os.makedirs(d, exist_ok=True); LOGS.append(open(os.path.join(d, 'log.txt'), 'a', encoding='utf-8'))
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")}  mode={a.mode}  videos={len(vids)}  max_frames={a.max_frames or "all"}  '
      f'out={", ".join(os.path.abspath(d) for _, d, _ in modes)}')
    P(f'### trained models: {os.path.abspath(os.path.join(a.out, "trained_models"))} (shared, reused when present)')
    t0 = time.time()
    try:
        for i, v in enumerate(vids):
            run_video(v, a, modes, i + 1, len(vids))
            el = time.time() - t0
            P(f'  == overall {i + 1}/{len(vids)} videos, elapsed {el / 60:.1f} min ==\n')
    except KeyboardInterrupt:
        P('\nSTOPPED by user. Finished videos are saved; run the same command again to continue.')
    for name, d, _ in modes:
        V, _ = CM.load_run(d)
        if V:
            txt, _, _ = CM.summarize(V, f'SUMMARY  {name}{sfx}  ({len(V)} videos)  {os.path.abspath(d)}')
            P('\n' + txt)
            open(os.path.join(d, 'summary.txt'), 'w', encoding='utf-8').write(txt + '\n')
            CM.write_csv(V, os.path.join(d, 'summary.csv'))
            P(f'  -> {os.path.join(d, "summary.txt")} and summary.csv written')
    P(f'ALL DONE in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
