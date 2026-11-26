#!/usr/bin/env python3
"""
CDnet 2014 -- FULL FRAMEWORK: locked CPB + semantic guidance + 3-stage communication (Liang et al. 2021),
frozen and/or with the guidance-gated online update, every metric, resumable, progress bars.

Reuses the trained CPB models saved by cdnet_cpb_eval.py (<out>/trained_models), so nothing is retrained.
Guidance cache: <cache>/<category>__<video>.npz  with  nums int[N], guid uint8[N,H,W] (0/255), prob float16 (unused)
  (the format written by make_guidance_cdnet_mmseg.py; read frame by frame, so long videos fit in memory).

Per ROI frame, from the SAME model state, it scores 7 masks:
  cpb            locked CPB alone (per RGB channel, 3x3 median, OR)
  guidance       guidance mask alone (at its own resolution, resized to the GT)
  stage1         CPB with guided support veto (blocks with >= 7 guidance-fg pixels skipped)
  stage12        + stage 2 as in the paper (Eq. 6-8: supports with gamma >= mean gamma replaced, Eq. 5 re-run)
  stage123       + stage 3 (foreground similarity, eps = 0.8)            <- the framework
  stage12_blk    + stage 2, block-replacement variant
  stage123_blk   + stage 3 after the block-replacement variant
Modes: frozen = no update ; online = guidance-gated update after every frame (alpha 0.01: no update where the
guidance says foreground; vetoed supports neither used nor updated), as in the WallFlower framework.

Output: <out>/framework_<name>/<mode>[_testN]/<key>/counts/<cat>__<video>.csv (+ info/, log.txt, SUMMARY.txt)
Examples
  python cdnet_framework_eval.py --root "<CDnet>" --cache "<cdnet_guidance_cache>" --inspect
  python -u cdnet_framework_eval.py --root "<CDnet>" --cache "<cdnet_guidance_cache>" --mode frozen --max_frames 60
  python -u cdnet_framework_eval.py --root "<CDnet>" --cache "<cdnet_guidance_cache>" --mode frozen
  python -u cdnet_framework_eval.py --root "<CDnet>" --cache "<cdnet_guidance_cache>" --mode online
"""
import sys, os, re, csv, json, time, argparse, glob, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import cpb_kernels as F
import wf_framework as WF
import cdnet_cpb_eval as E
import cdnet_metrics as CM

KEYS = ['cpb', 'guidance', 'stage1', 'stage12', 'stage123', 'stage12_blk', 'stage123_blk']
NAMES = {'cpb': 'CPB alone', 'guidance': 'guidance alone', 'stage1': 'stage 1', 'stage12': 'stage 1+2 (paper)',
         'stage123': 'stage 1+2+3 (FRAMEWORK)', 'stage12_blk': 'stage 1+2 (block-repl.)', 'stage123_blk': 'stage 1+2+3 (block-repl.)'}
P = E.P


class GuidStream:
    """Reads guid[N,H,W] from a .npz one frame at a time (no full decompression in memory)."""
    def __init__(self, path):
        z = np.load(path)
        self.nums = [int(n) for n in z['nums']]
        self.pos = {n: i for i, n in enumerate(self.nums)}
        z.close()
        self.zf = zipfile.ZipFile(path)
        self.f = self.zf.open('guid.npy')
        ver = np.lib.format.read_magic(self.f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(self.f) if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0(self.f))
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.fsize = int(np.prod(shape[1:])) * self.dtype.itemsize
        self.next = 0

    def get(self, n):
        i = self.pos.get(n)
        if i is None or i < self.next:
            return None
        while self.next < i:                                     # skip frames we do not need
            self.f.read(self.fsize); self.next += 1
        buf = self.f.read(self.fsize); self.next += 1
        return np.frombuffer(buf, self.dtype).reshape(self.shape[1:])

    def close(self):
        self.f.close(); self.zf.close()


def stages(model_rgb, rgb, gfg, a):
    """Stage masks for one frame, per RGB channel then OR (same code paths as the WallFlower/LIMU frameworks)."""
    acc = {}
    for c, m in enumerate(model_rgb.models):
        D = np.ascontiguousarray(m['DBLK_r'], np.float32)
        H, Wd = m['img_size']; ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8)); pool = D.shape[2]
        g = np.ascontiguousarray(rgb[:, :, c], np.float32)
        gcnt = F.block_fg_count(gfg, ws, hs); DBM = F.block_median(g, ws, hs)
        s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
        F.k_stage1(g, DBM, gcnt, gfg, D, s1, sc, su, a.af, a.floor, F.LAMBDA_DEC, a.bth, a.ref, pool, True)
        s1 = cv2.medianBlur(s1, 3)
        s2p = np.zeros_like(s1); WF.k_stage2_paper(g, DBM, gcnt, gfg, D, s1, s2p, a.af, a.floor, F.LAMBDA_DEC, a.bth, a.ref, pool)
        s3p = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2p, s3p, a.sigema, a.bth, a.ref, pool)
        s2m = np.zeros_like(s1); F.k_stage2(g, DBM, gcnt, gfg, gfg.astype(np.float32), D, s1, s2m, a.af, a.floor, a.bth, a.ref, pool, 1.0, 0.0)
        s3m = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2m, s3m, a.sigema, a.bth, a.ref, pool)
        for k, v in (('stage1', s1), ('stage12', s2p), ('stage123', s3p), ('stage12_blk', s2m), ('stage123_blk', s3m)):
            acc[k] = v if k not in acc else np.maximum(acc[k], v)
    return acc


class Runner:
    def __init__(self, mode, models, alpha):
        self.mode, self.models = mode, models
        self.L = E.locked_args(alpha); self.L.sigema = 0.8
        self.active, self.switches, self.rows = 0, 0, {k: [] for k in KEYS}

    def step(self, rgb, shape, gfg, need_stages=True):
        L = self.L
        cpb = self.models[self.active].detect(rgb, shape, L)
        if len(self.models) > 1 and np.mean(cpb > 127) > L.trigger:          # same switching rule as the CPB runs
            fr = [float(np.mean(mm.detect(rgb, shape, L) > 127)) for mm in self.models]; b = int(np.argmin(fr))
            if b != self.active:
                self.active, self.switches = b, self.switches + 1
                cpb = self.models[self.active].detect(rgb, shape, L)
        out = {'cpb': cpb}
        if need_stages:
            out.update(stages(self.models[self.active], rgb, gfg, L))
        if L.alpha > 0:
            WF.guided_update(self.models[self.active], rgb, gfg, L)          # causal, after the decision on this frame
        return out


def mode_dirs(a):
    sfx = f'_test{a.max_frames}' if a.max_frames else ''
    base = os.path.join(a.out, f'framework_{a.name}')
    modes = []
    if a.mode in ('frozen', 'both'):
        modes.append(('frozen', os.path.join(base, 'frozen' + sfx), 0.0))
    if a.mode in ('online', 'both'):
        modes.append((f'online_a{a.alpha:g}', os.path.join(base, f'online_a{a.alpha:g}' + sfx), a.alpha))
    return modes


def run_video(vd, a, modes, idx, nvid):
    cat, vid = os.path.basename(os.path.dirname(vd)), os.path.basename(vd); tag = f'{cat}__{vid}'
    todo = [(m, d, al) for m, d, al in modes if not all(os.path.exists(os.path.join(d, k, 'counts', tag + '.csv')) for k in KEYS)]
    if not todo:
        P(f'[{idx}/{nvid}] {tag}: already done -> skip'); return
    gp = os.path.join(a.cache, tag + '.npz')
    if not os.path.exists(gp):
        P(f'[{idx}/{nvid}] {tag}: NO guidance cache ({gp}) -> skip'); return
    os.makedirs(os.path.join(a.out, 'locks'), exist_ok=True)
    with E.VideoLock(os.path.join(a.out, 'locks', f'fw_{a.name}__{tag}.lock')) as lk:
        if not lk.ok:
            P(f'[{idx}/{nvid}] {tag}: LOCKED by another running copy -> skip'); return
        _run(vd, a, todo, idx, nvid, cat, vid, tag, gp)


def _run(vd, a, todo, idx, nvid, cat, vid, tag, gp):
    t0 = time.time()
    r0, r1 = [int(x) for x in open(os.path.join(vd, 'temporalROI.txt')).read().split()[:2]]
    inp, gtd = os.path.join(vd, 'input'), os.path.join(vd, 'groundtruth')
    nums = sorted(int(re.findall(r'(\d+)', os.path.basename(p))[-1]) for p in glob.glob(os.path.join(inp, 'in*.jpg')))
    path = lambda n: os.path.join(inp, f'in{n:06d}.jpg')
    H, W = cv2.imread(path(nums[0])).shape[:2]; ww, wh = E.work_size(W, H, a.budget)
    rd = lambda n: cv2.cvtColor(cv2.resize(cv2.imread(path(n)), (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    P(f'[{idx}/{nvid}] {tag}  {W}x{H} -> {ww}x{wh}  ROI {r0}-{r1}  modes: {", ".join(m for m, _, _ in todo)}')
    models, meta, loaded = E.train_or_load(vd, tag, a, rd, nums, r0, ww, wh)
    P(f'  {"loaded saved" if loaded else "trained new"} CPB model ({meta["n_train"]} frames, second state {int(meta["second_state"])})')
    runners = [Runner(m, E.copy_models(models) if al > 0 else models, al) for m, d, al in todo]
    G = GuidStream(gp)
    P(f'  guidance cache: {len(G.nums)} frames {G.shape[2]}x{G.shape[1]} ({os.path.basename(gp)})')
    test = [n for n in nums if r0 <= n <= r1]
    if a.max_frames:
        test = test[:a.max_frames]
    missing = 0
    bar = Bar(len(test), 'framework ' + '+'.join(r.mode for r in runners))
    for n in test:
        rgb = rd(n)
        gn = G.get(n)
        if gn is None:
            # no guidance for this frame: the CPB still runs (switching state, plain update in online mode),
            # but NO row is scored, so every key is evaluated on exactly the frames that have guidance
            missing += 1
            for r in runners:
                r.step(rgb, (wh, ww), np.zeros((wh, ww), np.uint8), need_stages=False)
            bar.update(); continue
        gnat = gn > 127
        gfg = cv2.resize(gnat.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
        outs = [r.step(rgb, (wh, ww), gfg) for r in runners]
        gpth = os.path.join(gtd, f'gt{n:06d}.png')
        g = cv2.imread(gpth, cv2.IMREAD_GRAYSCALE) if os.path.exists(gpth) else None
        if g is not None:
            gmask = (cv2.resize(gnat.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
                     if gnat.shape != g.shape else gnat.astype(np.uint8) * 255)
            for r, o in zip(runners, outs):
                for k in KEYS:
                    m = gmask if k == 'guidance' else o[k]
                    r.rows[k].append((n,) + E.score(m, g))
        bar.update()
    bar.close(); G.close()
    for r, (m, d, al) in zip(runners, todo):
        lines = []
        for k in KEYS:
            kd = os.path.join(d, k, 'counts'); os.makedirs(kd, exist_ok=True)
            csvp = os.path.join(kd, tag + '.csv'); tmp = E.tmp_name(csvp)
            with open(tmp, 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(['frame', 'tp', 'fp', 'fn', 'tn', 'gt_fg', 'valid']); w.writerows(r.rows[k])
            E.safe_replace(tmp, csvp)
            rows = [x for x in r.rows[k] if x[6] > 0]
            vm = CM.video_metrics(rows) if rows else None
            if vm:
                lines.append(f'    {NAMES[k]:27s} F {CM.fmt(vm["F"], 6)}  P {CM.fmt(vm["Pr"], 5, 2)}  R {CM.fmt(vm["Re"], 5, 2)}  '
                             f'thesis {CM.fmt(vm["thesis_fg"], 6)}')
        os.makedirs(os.path.join(d, 'info'), exist_ok=True)
        json.dump(dict(meta, mode=m, alpha=al, category=cat, video=vid, native=[W, H], roi=[r0, r1], switches=r.switches,
                       guidance_cache=gp, guidance_missing_frames=missing, max_frames=a.max_frames,
                       t_video=round(time.time() - t0, 1)), open(os.path.join(d, 'info', tag + '.json'), 'w'), indent=1)
        P(f'  [{m}] {vid}' + (f'  (switches {r.switches})' if r.switches else '') + (f'  ({missing} ROI frames without guidance: not scored)' if missing else ''))
        for ln in lines:
            P(ln)
    P(f'  video done in {(time.time() - t0) / 60:.1f} min')


class Bar(E.Bar):
    pass


def summarize_mode(name, d):
    rows, full = [], []
    for k in KEYS:
        V, _ = CM.load_run(os.path.join(d, k))
        if not V:
            continue
        txt, C, O = CM.summarize(V, f'{name} -- {NAMES[k]}  ({len(V)} videos)')
        full.append(txt)
        rows.append((k, len(V), O['official']['F'], O['official']['thesis_fg'], O['pooled'], C))
    if not rows:
        return ''
    cats = [c for c in CM.CATS if c in rows[0][5]]
    L = ['=' * 118, f'FRAMEWORK SUMMARY  {name}   (overall = mean of categories)', '=' * 118,
         f'{"":28s}{"videos":>7s}{"official F":>12s}{"thesis F":>10s}{"pooled F":>10s}',]
    for k, n, f, t, p, _ in rows:
        L.append(f'{NAMES[k]:28s}{n:7d}{CM.fmt(f, 12, 4)}{CM.fmt(t, 10, 4)}{CM.fmt(p, 10, 4)}')
    L.append('')
    L.append('Per category, official F:')
    L.append(f'{"":28s}' + ''.join(f'{c[:9]:>10s}' for c in cats))
    for k, n, f, t, p, C in rows:
        L.append(f'{NAMES[k]:28s}' + ''.join(CM.fmt(C[c]['F'], 10) for c in cats))
    L.append('Per category, thesis F:')
    for k, n, f, t, p, C in rows:
        L.append(f'{NAMES[k]:28s}' + ''.join(CM.fmt(C[c]['thesis_fg'], 10) for c in cats))
    L.append('Liang et al. 2021 (CDnet, published): CPB 0.633 | CPB guided by DeepLabV3+ 0.753 | by PSPNet 0.772 | DeepLabV3+ alone 0.676')
    short = '\n'.join(L)
    open(os.path.join(d, 'SUMMARY.txt'), 'w', encoding='utf-8').write(short + '\n\n' + '\n\n'.join(full) + '\n')
    return short


def inspect(a):
    files = sorted(glob.glob(os.path.join(a.cache, '*.npz')))
    print(f'{len(files)} cache files in {os.path.abspath(a.cache)}')
    vids = {f'{os.path.basename(os.path.dirname(v))}__{os.path.basename(v)}': v for v in E.find_videos(a.root)}
    names = {os.path.basename(f)[:-4] for f in files}
    for f in files[:3]:
        z = np.load(f); print(f'  {os.path.basename(f)}: keys {z.files}  ' + '  '.join(f'{k}{z[k].shape}{z[k].dtype}' for k in z.files if k != 'meta'))
    miss = sorted(set(vids) - names)
    print(f'videos in dataset: {len(vids)}   with a cache: {len(set(vids) & names)}   without: {len(miss)}')
    for m in miss:
        print(f'  no cache for {m}')
    extra = sorted(names - set(vids))
    if extra:
        print(f'cache files whose name matches no video (expected <category>__<video>.npz): {extra[:10]}')
    for t in sorted(set(vids) & names)[:53]:
        G = GuidStream(os.path.join(a.cache, t + '.npz'))
        r0, r1 = [int(x) for x in open(os.path.join(vids[t], 'temporalROI.txt')).read().split()[:2]]
        cov = sum(1 for n in range(r0, r1 + 1) if n in G.pos)
        st = sorted(n for n in G.nums if r0 <= n <= r1); steps = sorted(set(np.diff(st).tolist())) if len(st) > 1 else []
        print(f'  {t:45s} frames {len(G.nums):5d}  {G.shape[2]}x{G.shape[1]}  ROI coverage {cov}/{r1 - r0 + 1}  '
              f'first {st[0] if st else "-"} last {st[-1] if st else "-"} step {steps[:3]}')
        G.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='CDnet 2014 folder')
    ap.add_argument('--cache', required=True, help='guidance cache folder (<category>__<video>.npz)')
    ap.add_argument('--name', default='deeplab', help='guidance name, used in the output folder (deeplab / pspnet)')
    ap.add_argument('--out', default='cdnet_runs', help='same base folder as cdnet_cpb_eval.py (reuses trained_models)')
    ap.add_argument('--mode', choices=['frozen', 'online', 'both'], default='frozen')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--max_frames', type=int, default=0, help='quick test: first N ROI frames of each video')
    ap.add_argument('--only', default='', help='comma list of categories or category/video')
    ap.add_argument('--inspect', action='store_true', help='check the cache against the dataset, then stop')
    ap.add_argument('--summary', action='store_true', help='only print the summary of finished videos')
    ap.add_argument('--clear_locks', action='store_true')
    ap.add_argument('--max_train', type=int, default=400); ap.add_argument('--min_pre', type=int, default=50)
    ap.add_argument('--budget', type=float, default=160 * 120)
    a = ap.parse_args()
    if a.inspect:
        inspect(a); return
    modes = mode_dirs(a)
    if a.summary:
        for m, d, _ in modes:
            print(summarize_mode(m, d))
        return
    if a.clear_locks:
        for f in glob.glob(os.path.join(a.out, 'locks', f'fw_{a.name}__*.lock')):
            os.remove(f)
    vids = E.find_videos(a.root)
    if a.only:
        keep = [k.replace('\\', '/') for k in a.only.split(',')]
        vids = [v for v in vids if os.path.basename(os.path.dirname(v)) in keep
                or f'{os.path.basename(os.path.dirname(v))}/{os.path.basename(v)}' in keep]
    print(f'{len(vids)} videos found under {os.path.abspath(a.root)}', flush=True)
    if not vids:
        return
    for _, d, _ in modes:
        os.makedirs(d, exist_ok=True); E.LOGS.append(open(os.path.join(d, 'log.txt'), 'a', encoding='utf-8'))
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")}  FRAMEWORK guidance={a.name}  mode={a.mode}  videos={len(vids)}  '
      f'max_frames={a.max_frames or "all"}  cache={os.path.abspath(a.cache)}')
    t0 = time.time()
    try:
        for i, v in enumerate(vids):
            run_video(v, a, modes, i + 1, len(vids))
            P(f'  == overall {i + 1}/{len(vids)} videos, elapsed {(time.time() - t0) / 60:.1f} min ==\n')
    except KeyboardInterrupt:
        P('\nSTOPPED by user. Finished videos are saved; run the same command again to continue.')
    for m, d, _ in modes:
        s = summarize_mode(m, d)
        if s:
            P('\n' + s); P(f'  -> full tables: {os.path.join(d, "SUMMARY.txt")}')
    P(f'ALL DONE in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
