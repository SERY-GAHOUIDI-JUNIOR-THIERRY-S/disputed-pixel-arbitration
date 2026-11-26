#!/usr/bin/env python3
"""
================================================================================
DISPUTE ARBITER (MLP) for frozen CPB vs DeepLab guidance  --  CDnet2014 source domain
================================================================================
What it does
  The locked CPB (frozen, the models saved by cdnet_cpb_eval.py) runs stage 1 with the guidance veto.
  A pixel is DISPUTED when the stage-1 CPB mask and the guidance mask disagree. Agreement pixels keep
  the common label; the arbiter decides every disputed pixel:
        M(p) = CPB(p)                     if CPB(p) == guidance(p)
             = [ MLP(x_p) >= thr ]        otherwise          (target = ground truth foreground)

Protocol (fixed before any result; see the methodology doc, section 13)
  * CDnet2014 is the SOURCE domain only.  PTZ is excluded (moving camera breaks the CPB assumption).
  * ONE split by video: in each of the other 10 categories, 1 video -> validation, the rest -> training.
    No held-out CDnet test, no k-fold.  Split = split.json (seeded, written once, then reused).
  * Rows = every valid disputed pixel of --n_frames evenly spaced ROI frames that have guidance and GT
    (GT 85/170 ignored, 50 = background).  Training rows are capped per video and weighted so every video
    and both classes count equally.  Validation uses ALL its disputed rows.
  * Model selection on validation only: early stopping on the balanced validation loss; the full-frame
    F-measure is recomposed exactly = agreement counts + the arbiter's decisions on the disputed pixels.
    Scored at the CPB working size (160x120 budget), for selection only.
  * External datasets (WallFlower, LIMU, ...) are never touched here.

Candidate features (28), grouped for the ablation
  core : cpb_says_fg                       dispute type (1 = CPB fg / guidance bg, 0 = the reverse)
  C    : cpb_score, supp_norm, chan_votes, resid_z
         gamma-weighted changed fraction, valid supports / 20 (same channel), channels voting fg / 3,
         mean over supports of min_block |p - Q - b| / (eta sigma)   (continuous margin of the pair test)
  S    : rpb, spb, spq                     stage-3 cues: |p - Q~ - b|, |Q~ - p|, |I_q - I_p| to nearest CPB fg
  G    : guid_prob, guid_prob_loc, guid_bdist, guid_blob
         guidance probability, its 5x5 mean, signed distance to the guidance boundary, guidance blob size
  L    : cpb_occ, guid_occ, disagree_occ, cpb_blob      local 5x5 / 9x9 occupancies, CPB blob size
  R    : frame_cpb_frac, frame_guid_frac, frame_dis_frac, frame_iou, frame_luma, frame_dluma
         frame-level reliability: how much each side marks, disagreement, IoU, brightness and its jump
  T    : motion, cpb_prev_occ, guid_prev  frame difference, CPB at t-1, guidance at the previous cached frame
  A    : luma, texture, edge              appearance

Sub-commands
  extract  per-video feature files       (slow part, CPU, resumable: finished videos are skipped)
  split    write / show split.json
  ablate   feature ablation: group ladder, leave-one-group-out, single groups, linear model,
           permutation importance per feature and per group  -> ablation/ (log, csv, json)
  train    final MLP on a chosen feature set, several seeds, threshold chosen on validation,
           optional refit on train+val with the selected epoch count  -> models/

Example (Windows PowerShell, one line each; run from the folder that holds cdnet_cpb_eval.py)
  python -u arbiter_mlp.py extract --root "<CDnet2014>" --cache "<cdnet_guidance_cache>" --cpb_out cdnet_runs --work arbiter_runs
  python -u arbiter_mlp.py split   --work arbiter_runs
  python -u arbiter_mlp.py ablate  --work arbiter_runs
  python -u arbiter_mlp.py train   --work arbiter_runs --features ALL
Requires only numpy, opencv, numba (no torch): the MLP is written in numpy.
================================================================================
"""
import sys, os, re, json, time, glob, argparse, zipfile, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from numba import njit, prange
import cpb_kernels as F
import wf_framework as WF
import cdnet_cpb_eval as E
import cdnet_metrics as CM

VERSION = 'arb1'
P = E.P
Bar = E.Bar
SPLIT_SEED = 2026
EXCLUDE_CATS = ('PTZ',)

FEATURES = ['cpb_says_fg',
            'cpb_score', 'supp_norm', 'chan_votes', 'resid_z',
            'rpb', 'spb', 'spq',
            'guid_prob', 'guid_prob_loc', 'guid_bdist', 'guid_blob',
            'cpb_occ', 'guid_occ', 'disagree_occ', 'cpb_blob',
            'frame_cpb_frac', 'frame_guid_frac', 'frame_dis_frac', 'frame_iou', 'frame_luma', 'frame_dluma',
            'motion', 'cpb_prev_occ', 'guid_prev',
            'luma', 'texture', 'edge']
GROUPS = {'core': ['cpb_says_fg'],
          'C': ['cpb_score', 'supp_norm', 'chan_votes', 'resid_z'],
          'S': ['rpb', 'spb', 'spq'],
          'G': ['guid_prob', 'guid_prob_loc', 'guid_bdist', 'guid_blob'],
          'L': ['cpb_occ', 'guid_occ', 'disagree_occ', 'cpb_blob'],
          'R': ['frame_cpb_frac', 'frame_guid_frac', 'frame_dis_frac', 'frame_iou', 'frame_luma', 'frame_dluma'],
          'T': ['motion', 'cpb_prev_occ', 'guid_prev'],
          'A': ['luma', 'texture', 'edge']}
GROUP_NAMES = {'core': 'dispute type', 'C': 'CPB evidence', 'S': 'stage-3 cues', 'G': 'guidance',
               'L': 'local context', 'R': 'frame reliability', 'T': 'temporal', 'A': 'appearance'}
ALL_GROUPS = ['core', 'C', 'S', 'G', 'L', 'R', 'T', 'A']
assert sum(len(v) for v in GROUPS.values()) == len(FEATURES)


# =============================================================================== utilities
def tee(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    E.LOGS.append(open(path, 'a', encoding='utf-8'))


def save_npz(path, **arr):
    tmp = E.tmp_name(path)
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arr)
    E.safe_replace(tmp, path)


def save_json(path, obj):
    tmp = E.tmp_name(path)
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o)))
    E.safe_replace(tmp, path)


def parse_features(spec):
    """'ALL' | 'core,C,G' (groups) | 'cpb_score,guid_prob' (names) | mixes; core is always added."""
    toks = [s.strip() for s in spec.split(',') if s.strip()]
    out = []
    for t in toks:
        if t.startswith('-'):
            continue
        out += list(FEATURES) if t.upper() == 'ALL' else (GROUPS[t] if t in GROUPS else [t])
    if not out:
        out = list(FEATURES)
    for t in toks:
        if t.startswith('-'):
            drop = GROUPS[t[1:]] if t[1:] in GROUPS else [t[1:]]
            out = [f for f in out if f not in drop]
    if 'cpb_says_fg' not in out:
        out = ['cpb_says_fg'] + out
    bad = [f for f in out if f not in FEATURES]
    if bad:
        raise SystemExit(f'unknown feature(s) {bad}; groups: {list(GROUPS)}')
    return [f for f in FEATURES if f in out]                     # canonical order


# =============================================================================== guidance stream
class NpyStream:
    """Reads one array of a .npz frame by frame (forward only) and remembers the previous frame."""
    def __init__(self, path, member):
        self.zf = zipfile.ZipFile(path)
        self.f = self.zf.open(member + '.npy')
        ver = np.lib.format.read_magic(self.f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(self.f) if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0(self.f))
        if fortran:
            raise RuntimeError(f'{path}:{member} is Fortran-ordered; re-save the cache in C order')
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.fsize = int(np.prod(shape[1:])) * self.dtype.itemsize
        self.next, self.prev = 0, None

    def get(self, i):
        if i < self.next:
            raise RuntimeError('NpyStream is forward-only')
        prev = self.prev
        while self.next < i:
            prev = self.f.read(self.fsize); self.next += 1
        cur = self.f.read(self.fsize); self.next += 1
        self.prev = cur
        conv = lambda b: None if b is None else np.frombuffer(b, self.dtype).reshape(self.shape[1:])
        return conv(cur), conv(prev)

    def close(self):
        self.f.close(); self.zf.close()


# =============================================================================== per-pixel CPB kernel
@njit(cache=True, parallel=True, fastmath=False)
def k_disp(gray, DBM, gcnt, D, s1, rows, cols, af, floor, BTH, REF, pool, out):
    """For each disputed pixel (one channel): out[:,0] resid_z, [:,1] rpb, [:,2] spb, [:,3] spq."""
    H, W = gray.shape
    ws, hs = DBM.shape
    N = rows.shape[0]
    for t in prange(N):
        x = rows[t]; y = cols[t]; p = gray[x, y]
        nb = 0; rz = 0.0; rp = 0.0; sb = 0.0
        for k in range(pool):
            if nb >= REF:
                break
            gam = D[x, y, k, 4]
            if gam <= 0.0:
                continue
            bi = int(D[x, y, k, 0]); bj = int(D[x, y, k, 1])
            if bi < 0: bi = 0
            elif bi >= ws: bi = ws - 1
            if bj < 0: bj = 0
            elif bj >= hs: bj = hs - 1
            if gcnt[bi, bj] >= BTH:
                continue
            nb += 1
            sig = D[x, y, k, 3]
            if sig < floor: sig = floor
            b = D[x, y, k, 2]
            mn = 1e9
            for u in range(8):
                rr = bi * 8 + u
                if rr >= H: continue
                for v in range(8):
                    cc = bj * 8 + v
                    if cc >= W: continue
                    d = abs((p - gray[rr, cc]) - b)
                    if d < mn: mn = d
            z = mn / (af * sig) if af * sig > 0 else 0.0
            if z > 5.0: z = 5.0
            rz += z
            rp += abs(p - DBM[bi, bj] - b)
            sb += abs(DBM[bi, bj] - p)
        if nb > 0:
            out[t, 0] = rz / nb; out[t, 1] = rp / nb; out[t, 2] = sb / nb
        else:
            out[t, 0] = 0.0; out[t, 1] = 0.0; out[t, 2] = 0.0
        S = 0.0; fg = 0
        for r in range(1, 16):
            if fg >= 20:
                break
            for i in range(x - r, x + r + 1):
                if i < 0 or i >= H:
                    continue
                if i == x - r or i == x + r:
                    for j in range(y - r, y + r + 1):
                        if j >= 0 and j < W and s1[i, j] > 0:
                            S += abs(gray[i, j] - p); fg += 1
                else:
                    j = y - r
                    if j >= 0 and s1[i, j] > 0:
                        S += abs(gray[i, j] - p); fg += 1
                    j = y + r
                    if j < W and s1[i, j] > 0:
                        S += abs(gray[i, j] - p); fg += 1
        out[t, 3] = S / fg if fg > 0 else 255.0


def blob_map(m):
    m8 = (m > 0).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m8, 8)
    area = st[:, cv2.CC_STAT_AREA].astype(np.float32); area[0] = 0.0
    return (np.log1p(area) / np.log1p(m8.size))[lbl].astype(np.float32)


def signed_dist(g):
    g8 = (g > 0).astype(np.uint8)
    if not g8.any():
        return np.full(g8.shape, -1.0, np.float32)
    if g8.all():
        return np.full(g8.shape, 1.0, np.float32)
    din = cv2.distanceTransform(g8, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - g8, cv2.DIST_L2, 3)
    return (np.clip(np.where(g8 > 0, din, -dout), -20, 20) / 20.0).astype(np.float32)


def box(a, k):
    return cv2.blur(a.astype(np.float32), (k, k))


def luma(rgb):
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)


# =============================================================================== extraction
def video_list(root, cache):
    out = []
    for vd in E.find_videos(root):
        cat, vid = os.path.basename(os.path.dirname(vd)), os.path.basename(vd)
        if cat in EXCLUDE_CATS:
            continue
        out.append((vd, cat, vid, f'{cat}__{vid}', os.path.join(cache, f'{cat}__{vid}.npz')))
    return out


def frame_features(M, rgb, rgbp, gfg, gp, gpp, gw, L):
    """All per-frame maps; returns (agree counts, disputed rows (y, x), X rows, labels, fw labels, check)."""
    H, Wd = gfg.shape
    ws, hs = int(np.ceil(H / 8)), int(np.ceil(Wd / 8))
    gcnt = F.block_fg_count(gfg, ws, hs)
    s1c, scs, sus, chans, DBMs, Ds = [], [], [], [], [], []
    s3p_acc = None; s3b_acc = None
    for c, m in enumerate(M.models):
        D = np.ascontiguousarray(m['DBLK_r'], np.float32); pool = D.shape[2]
        g = np.ascontiguousarray(rgb[:, :, c], np.float32); DBM = F.block_median(g, ws, hs)
        s1 = np.zeros((H, Wd), np.uint8); sc = np.zeros((H, Wd), np.float32); su = np.zeros((H, Wd), np.float32)
        F.k_stage1(g, DBM, gcnt, gfg, D, s1, sc, su, L.af, L.floor, F.LAMBDA_DEC, L.bth, L.ref, pool, True)
        s1 = cv2.medianBlur(s1, 3)
        s2p = np.zeros_like(s1); WF.k_stage2_paper(g, DBM, gcnt, gfg, D, s1, s2p, L.af, L.floor, F.LAMBDA_DEC, L.bth, L.ref, pool)
        s3p = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2p, s3p, L.sigema, L.bth, L.ref, pool)
        s2b = np.zeros_like(s1); F.k_stage2(g, DBM, gcnt, gfg, gfg.astype(np.float32), D, s1, s2b, L.af, L.floor, L.bth, L.ref, pool, 1.0, 0.0)
        s3b = np.zeros_like(s1); F.k_stage3(g, DBM, gcnt, gfg, D, s2b, s3b, L.sigema, L.bth, L.ref, pool)
        s3p_acc = s3p if s3p_acc is None else np.maximum(s3p_acc, s3p)
        s3b_acc = s3b if s3b_acc is None else np.maximum(s3b_acc, s3b)
        s1c.append(s1); scs.append(sc); sus.append(su); chans.append(g); DBMs.append(DBM); Ds.append(D)
    s1 = np.maximum.reduce(s1c)
    cpb = s1 > 0; gf = gfg > 0
    valid = (gw != 85) & (gw != 170); pos = (gw == 255) & valid
    agree = valid & (cpb == gf); disp = valid & (cpb != gf)
    ac = np.array([(cpb & pos & agree).sum(), (cpb & ~pos & agree).sum(), (~cpb & pos & agree).sum(),
                   (~cpb & ~pos & agree).sum()], np.int64)
    check = int((agree & ((s3p_acc > 0) != cpb)).sum() + (agree & ((s3b_acc > 0) != cpb)).sum())
    ys, xs = np.nonzero(disp)
    info = dict(agree=ac, gt_fg=int(pos.sum()), valid=int(valid.sum()), check=check, s1=s1, s3p=s3p_acc, s3b=s3b_acc)
    if len(ys) == 0:
        return info, None
    N = len(ys)
    ys32 = ys.astype(np.int64); xs32 = xs.astype(np.int64)
    # ---- C + S (per channel, then combined) ----
    SC = np.stack(scs); SU = np.stack(sus)
    amax = np.argmax(SC[:, ys, xs], axis=0)
    cpb_score = SC[amax, ys, xs]; supp = SU[amax, ys, xs] / float(L.ref)
    votes = np.stack([s[ys, xs] > 0 for s in s1c]).sum(0) / 3.0
    kd = np.zeros((len(M.models), N, 4), np.float32)
    for c in range(len(M.models)):
        out = np.zeros((N, 4), np.float32)
        k_disp(chans[c], DBMs[c], gcnt, Ds[c], s1, ys32, xs32, float(L.af), float(L.floor), float(L.bth),
               int(L.ref), int(Ds[c].shape[2]), out)
        kd[c] = out
    resid_z = kd[:, :, 0].max(0); rpb = kd[:, :, 1].mean(0) / 255.0; spb = kd[:, :, 2].mean(0) / 255.0
    spq = kd[:, :, 3].mean(0) / 255.0
    # ---- G, L ----
    gpl = box(gp, 5); gbd = signed_dist(gfg); gbl = blob_map(gfg)
    cocc = box(cpb, 5); gocc = box(gf, 5); docc = box(cpb != gf, 9); cbl = blob_map(cpb)
    # ---- R ----
    Lc = luma(rgb); Lp = luma(rgbp)
    union = float((cpb | gf).sum())
    iou = float((cpb & gf).sum()) / union if union > 0 else 1.0
    fr = [float(cpb.mean()), float(gf.mean()), float((cpb != gf).mean()), iou, float(Lc.mean() / 255.0),
          float(abs(Lc.mean() - Lp.mean()) / 255.0)]
    # ---- T ----
    mot = box(np.abs(Lc - Lp), 5) / 255.0
    cprev = box(M.detect(rgbp, (H, Wd), L) > 127, 5)
    # ---- A ----
    mu = box(Lc, 5); tex = np.clip(np.sqrt(np.clip(box(Lc * Lc, 5) - mu * mu, 0, None)) / 64.0, 0, 2)
    sx = cv2.Sobel(Lc, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(Lc, cv2.CV_32F, 0, 1, ksize=3)
    edg = np.clip(np.sqrt(sx * sx + sy * sy) / 255.0, 0, 2)
    full = lambda v: np.full(N, v, np.float32)
    cols = {
        'cpb_says_fg': cpb[ys, xs].astype(np.float32),
        'cpb_score': cpb_score, 'supp_norm': supp, 'chan_votes': votes, 'resid_z': resid_z,
        'rpb': rpb, 'spb': spb, 'spq': spq,
        'guid_prob': gp[ys, xs], 'guid_prob_loc': gpl[ys, xs], 'guid_bdist': gbd[ys, xs], 'guid_blob': gbl[ys, xs],
        'cpb_occ': cocc[ys, xs], 'guid_occ': gocc[ys, xs], 'disagree_occ': docc[ys, xs], 'cpb_blob': cbl[ys, xs],
        'frame_cpb_frac': full(fr[0]), 'frame_guid_frac': full(fr[1]), 'frame_dis_frac': full(fr[2]),
        'frame_iou': full(fr[3]), 'frame_luma': full(fr[4]), 'frame_dluma': full(fr[5]),
        'motion': mot[ys, xs], 'cpb_prev_occ': cprev[ys, xs], 'guid_prev': gpp[ys, xs],
        'luma': Lc[ys, xs] / 255.0, 'texture': tex[ys, xs], 'edge': edg[ys, xs]}
    X = np.stack([np.asarray(cols[f], np.float32) for f in FEATURES], 1)
    rows = dict(X=X.astype(np.float16), y=pos[ys, xs].astype(np.uint8), cpb=cpb[ys, xs].astype(np.uint8),
                guid=gf[ys, xs].astype(np.uint8), fwp=(s3p_acc[ys, xs] > 0).astype(np.uint8),
                fwb=(s3b_acc[ys, xs] > 0).astype(np.uint8), ys=ys, xs=xs)
    return info, rows


def extract_video(vd, cat, vid, tag, gp_path, a, k, nv):
    out_path = os.path.join(a.work, 'feats', tag + '.npz')
    if os.path.exists(out_path):
        try:
            z = np.load(out_path, allow_pickle=False)
            if str(z['version']) == VERSION and int(z['n_frames_req']) == a.n_frames:
                P(f'[{k}/{nv}] {tag}: done -> skip'); return 'done'
        except Exception:
            pass
    if not os.path.exists(gp_path):
        P(f'[{k}/{nv}] {tag}: NO guidance cache -> skip'); return 'nocache'
    mp = os.path.join(a.cpb_out, 'trained_models', tag + '.npz')
    if not (os.path.exists(mp) and os.path.exists(mp[:-4] + '.json')):
        P(f'[{k}/{nv}] {tag}: NO trained CPB model in {os.path.dirname(mp)} -> skip (run cdnet_cpb_eval.py first)'); return 'nomodel'
    t0 = time.time()
    z = np.load(gp_path)
    if 'prob' not in z.files:
        raise RuntimeError(f"{gp_path} has no 'prob' array: regenerate the guidance cache with soft probabilities")
    cnums = [int(x) for x in z['nums']]; z.close()
    r0, r1 = [int(x) for x in open(os.path.join(vd, 'temporalROI.txt')).read().split()[:2]]
    inp, gtd = os.path.join(vd, 'input'), os.path.join(vd, 'groundtruth')
    ipath = lambda n: os.path.join(inp, f'in{n:06d}.jpg')
    H, W = cv2.imread(ipath(cnums[0])).shape[:2]; ww, wh = E.work_size(W, H, a.budget)
    models, meta = E.load_models(mp)
    if meta.get('work') != [ww, wh]:
        raise RuntimeError(f'{tag}: saved CPB model is {meta.get("work")} but --budget gives {[ww, wh]}; use the same --budget as cdnet_cpb_eval.py')
    rd = lambda n: cv2.cvtColor(cv2.resize(cv2.imread(ipath(n)), (ww, wh), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2RGB).astype(np.float32)
    L = E.locked_args(0.0); L.sigema = 0.8
    cand = [i for i, n in enumerate(cnums) if r0 <= n <= r1 and os.path.exists(os.path.join(gtd, f'gt{n:06d}.png'))]
    if not cand:
        P(f'[{k}/{nv}] {tag}: no guided ROI frame with GT -> skip'); return 'noframes'
    sel = sorted(set(cand[j] for j in np.linspace(0, len(cand) - 1, min(a.n_frames, len(cand))).round().astype(int)))
    P(f'[{k}/{nv}] {tag}  {W}x{H} -> {ww}x{wh}  ROI {r0}-{r1}  guided ROI frames {len(cand)}  sampled {len(sel)}  '
      f'CPB models {len(models)}')
    # ---- frame switching state (only matters with 2 models): same rule as the CPB / framework runs ----
    active = {}
    if len(models) > 1:
        last = cnums[sel[-1]]; seq = [n for n in range(r0, last + 1) if os.path.exists(ipath(n))]
        want = set(cnums[i] for i in sel); act = 0; sw = 0
        bar = Bar(len(seq), 'switch state')
        for n in seq:
            rgb = rd(n); m = models[act].detect(rgb, (wh, ww), L)
            if np.mean(m > 127) > L.trigger:
                fr = [float(np.mean(mm.detect(rgb, (wh, ww), L) > 127)) for mm in models]; b = int(np.argmin(fr))
                if b != act:
                    act, sw = b, sw + 1
            if n in want:
                active[n] = act
            bar.update()
        bar.close(); P(f'  model switches over the ROI: {sw}')
    SG, SP = NpyStream(gp_path, 'guid'), NpyStream(gp_path, 'prob')
    acc = {k_: [] for k_ in ('X', 'y', 'cpb', 'guid', 'fwp', 'fwb', 'fidx')}
    frames, agree, gtfg, valid = [], [], [], []
    check = 0; nfr = 0
    bar = Bar(len(sel), 'features')
    for i in sel:
        gcur, gprev = SG.get(i); pcur, pprev = SP.get(i)
        n = cnums[i]
        gt = cv2.imread(os.path.join(gtd, f'gt{n:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            bar.update(); continue
        gw = cv2.resize(gt, (ww, wh), interpolation=cv2.INTER_NEAREST)
        if not ((gw != 85) & (gw != 170)).any():
            bar.update(); continue                      # unreleased GT (all 85/170): not a test frame
        rgb = rd(n); rgbp = rd(n - 1) if os.path.exists(ipath(n - 1)) else rgb
        gfg = cv2.resize((gcur > 127).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
        gp = cv2.resize(pcur.astype(np.float32), (ww, wh), interpolation=cv2.INTER_LINEAR)
        gpp = cv2.resize(pprev.astype(np.float32), (ww, wh), interpolation=cv2.INTER_LINEAR) if pprev is not None else gp
        M = models[active.get(n, 0)]
        info, rows = frame_features(M, rgb, rgbp, gfg, gp, gpp, gw, L)
        fi = len(frames)
        frames.append(n); agree.append(info['agree']); gtfg.append(info['gt_fg']); valid.append(info['valid'])
        check += info['check']; nfr += 1
        if rows is not None:
            for k_ in ('X', 'y', 'cpb', 'guid', 'fwp', 'fwb'):
                acc[k_].append(rows[k_])
            acc['fidx'].append(np.full(len(rows['y']), fi, np.int32))
        bar.update()
    bar.close(); SG.close(); SP.close()
    if not frames:
        P(f'  {tag}: no scorable frame -> skip'); return 'noframes'
    cat_ = lambda k_, dt, w=None: (np.concatenate(acc[k_]) if acc[k_] else np.zeros((0, w) if w else 0, dt))
    D = dict(X=cat_('X', np.float16, len(FEATURES)), y=cat_('y', np.uint8), cpb=cat_('cpb', np.uint8),
             guid=cat_('guid', np.uint8), fwp=cat_('fwp', np.uint8), fwb=cat_('fwb', np.uint8),
             fidx=cat_('fidx', np.int32), frames=np.array(frames, np.int32), agree=np.stack(agree).astype(np.int64),
             gt_fg=np.array(gtfg, np.int64), valid=np.array(valid, np.int64), features=np.array(FEATURES),
             cat=np.array(cat), video=np.array(vid), version=np.array(VERSION), n_frames_req=np.array(a.n_frames),
             work=np.array([ww, wh]), n_models=np.array(len(models)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_npz(out_path, **D)
    v = load_video(out_path)
    m = {s: video_metrics(v, dec) for s, dec in base_decisions(v).items()}
    nd = len(v['y'])
    P(f'  disputed px {nd} ({nd / max(nfr, 1):.0f}/frame, {100 * v["y"].mean() if nd else 0:.0f}% GT-fg)  '
      f'agreement check {"OK" if check == 0 else f"MISMATCH {check} px"}  {(time.time() - t0) / 60:.1f} min')
    P('  F (official, working size): ' + '  '.join(f'{s} {CM.fmt(m[s]["F"], 5)}' for s in m))
    return 'ok'


def cmd_extract(a):
    os.makedirs(a.work, exist_ok=True); tee(os.path.join(a.work, 'extract_log.txt'))
    for nm, pth in (('--root', a.root), ('--cache', a.cache), ('--cpb_out trained_models', os.path.join(a.cpb_out, 'trained_models'))):
        if not os.path.isdir(pth):
            raise SystemExit(f'ERROR: {nm} folder not found: {os.path.abspath(pth)}\n'
                             f'  Use the real paths (the same ones you gave cdnet_framework_eval.py), in quotes.')
    vids = video_list(a.root, a.cache)
    if not vids:
        raise SystemExit(f'ERROR: no CDnet video (folder with temporalROI.txt + input/) under {os.path.abspath(a.root)}')
    if a.only:
        keep = [x.replace('\\', '/') for x in a.only.split(',')]
        vids = [v for v in vids if v[1] in keep or f'{v[1]}/{v[2]}' in keep]
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} EXTRACT {len(vids)} videos (PTZ excluded)  n_frames={a.n_frames}  '
      f'cache={os.path.abspath(a.cache)}  cpb models={os.path.abspath(os.path.join(a.cpb_out, "trained_models"))}')
    t0 = time.time(); st = {}
    try:
        for k, (vd, cat, vid, tag, gp) in enumerate(vids, 1):
            r = extract_video(vd, cat, vid, tag, gp, a, k, len(vids)); st[r] = st.get(r, 0) + 1
            P(f'  == {k}/{len(vids)} videos, elapsed {(time.time() - t0) / 60:.1f} min ==')
    except KeyboardInterrupt:
        P('\nSTOPPED. Finished videos are saved; run the same command to continue.')
    P(f'EXTRACT finished: {st}  ({(time.time() - t0) / 60:.1f} min)')


# =============================================================================== data + metrics
def load_video(path):
    z = np.load(path, allow_pickle=False)
    v = {k: z[k] for k in ('X', 'y', 'cpb', 'guid', 'fwp', 'fwb', 'fidx', 'frames', 'agree', 'gt_fg', 'valid')}
    v['cat'] = str(z['cat']); v['video'] = str(z['video']); v['tag'] = f'{v["cat"]}__{v["video"]}'
    feats = [str(f) for f in z['features']]
    if feats != FEATURES:
        raise RuntimeError(f'{path} was extracted with another feature list; re-extract')
    v['y'] = v['y'].astype(bool)
    return v


def base_decisions(v):
    return {'cpb': v['cpb'] > 0, 'guidance': v['guid'] > 0, 'framework': v['fwp'] > 0,
            'framework_blk': v['fwb'] > 0, 'oracle': v['y']}


def video_metrics(v, dec):
    """Exact full-frame metrics = agreement counts + the decisions on the disputed rows."""
    n = len(v['frames']); y = v['y']; d = np.asarray(dec, bool); fi = v['fidx']
    c = v['agree'].astype(np.float64).copy()
    if len(y):
        c[:, 0] += np.bincount(fi, weights=(d & y), minlength=n)
        c[:, 1] += np.bincount(fi, weights=(d & ~y), minlength=n)
        c[:, 2] += np.bincount(fi, weights=(~d & y), minlength=n)
        c[:, 3] += np.bincount(fi, weights=(~d & ~y), minlength=n)
    rows = [(v['frames'][i], c[i, 0], c[i, 1], c[i, 2], c[i, 3], v['gt_fg'][i], v['valid'][i]) for i in range(n) if v['valid'][i] > 0]
    return CM.video_metrics(rows)


def summarize(vids, decs):
    """vids: list of video dicts; decs: list of decision arrays. -> official overall (category mean), thesis, per video."""
    per = {}
    for v, d in zip(vids, decs):
        per[v['tag']] = video_metrics(v, d)
    cats = {}
    for v in vids:
        cats.setdefault(v['cat'], []).append(per[v['tag']])
    def overall(key):
        cv = [np.nanmean([m[key] for m in ms]) for ms in cats.values() if np.isfinite([m[key] for m in ms]).any()]
        return float(np.mean(cv)) if cv else float('nan')
    return dict(F=overall('F'), thesis=overall('thesis_fg'), meanF=overall('meanF_fg')), per


def dispute_metrics(p, y, w=None):
    """p = predicted P(fg) on disputed rows; y = GT fg. Balanced (video/class) loss, balanced accuracy, AUC."""
    eps = 1e-7; p = np.clip(p, eps, 1 - eps); yb = y.astype(bool)
    w = np.ones(len(y)) if w is None else w
    ll = -np.sum(w * (yb * np.log(p) + (~yb) * np.log(1 - p))) / np.sum(w)
    d = p >= 0.5
    tpr = (d & yb).sum() / max(yb.sum(), 1); tnr = (~d & ~yb).sum() / max((~yb).sum(), 1)
    acc = float((d == yb).mean()) if len(y) else float('nan')
    if yb.any() and (~yb).any():
        r = np.empty(len(p)); r[np.argsort(p, kind='mergesort')] = np.arange(1, len(p) + 1)
        n1 = float(yb.sum()); n0 = float((~yb).sum())          # float: avoids int32 overflow on Windows
        auc = (float(r[yb].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
    else:
        auc = float('nan')
    return dict(logloss=float(ll), bal_acc=float(0.5 * (tpr + tnr)), acc=acc, auc=float(auc))


def balanced_weights(y, vid_ids):
    """Every video the same total weight, then both classes the same total weight; mean 1."""
    w = np.zeros(len(y), np.float64)
    for u in np.unique(vid_ids):
        m = vid_ids == u; w[m] = 1.0 / m.sum()
    for c in (False, True):
        m = (y == c)
        if m.any():
            w[m] *= 0.5 * w.sum() / w[m].sum()
    return (w / w.mean()).astype(np.float32)


# =============================================================================== numpy MLP
class MLP:
    def __init__(self, d, hidden=(64, 64), seed=0):
        rng = np.random.RandomState(seed); sizes = [d] + list(hidden) + [1]
        self.W = [(rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.hidden = tuple(hidden)

    def params(self):
        return self.W + self.b

    def logits(self, X):
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
        return (h @ self.W[-1] + self.b[-1])[:, 0]

    def predict(self, X, bs=262144):
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), bs):
            z = self.logits(X[s:s + bs]); out[s:s + bs] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return out

    def grads(self, X, y, w, wd):
        hs = [X]; h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0); hs.append(h)
        z = (h @ self.W[-1] + self.b[-1])[:, 0]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        sw = w.sum()
        loss = -np.sum(w * (y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))) / sw
        g = ((p - y) * w / sw)[:, None].astype(np.float32)
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = hs[i].T @ g + wd * self.W[i]; gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (hs[i] > 0)
        return loss, gW + gb

    def state(self):
        return [p.copy() for p in self.params()]

    def load(self, st):
        n = len(self.W)
        self.W = [s.copy() for s in st[:n]]; self.b = [s.copy() for s in st[n:]]


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            p -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)


def load_mlp(npz_path, json_path=None):
    """Load a model saved by `train`: returns (net, cols, mu, sd, threshold, meta)."""
    z = np.load(npz_path)
    meta = json.load(open(json_path or npz_path[:-4] + '.json'))
    n = int(z['n_layers']); net = MLP(len(meta['features']), tuple(meta['hidden']), 0)
    net.W = [z[f'p{i}'].astype(np.float32) for i in range(n)]; net.b = [z[f'p{n + i}'].astype(np.float32) for i in range(n)]
    cols = [FEATURES.index(f) for f in meta['features']]
    return net, cols, z['mu'].astype(np.float32), z['sd'].astype(np.float32), float(meta['threshold']), meta


# =============================================================================== split + datasets
def feat_files(work):
    return sorted(glob.glob(os.path.join(work, 'feats', '*.npz')))


def get_split(work, create=True, verbose=True):
    sp = os.path.join(work, 'split.json')
    if os.path.exists(sp):
        s = json.load(open(sp))
    else:
        if not create:
            raise SystemExit('no split.json')
        tags = [os.path.basename(f)[:-4] for f in feat_files(work)]
        if not tags:
            raise SystemExit('no feature files yet: run extract first')
        bycat = {}
        for t in tags:
            c = t.split('__')[0]
            if c not in EXCLUDE_CATS:
                bycat.setdefault(c, []).append(t)
        rng = np.random.RandomState(SPLIT_SEED); val, train = [], []
        for c in sorted(bycat):
            vs = sorted(bycat[c])
            if len(vs) >= 2:
                j = rng.randint(len(vs)); val.append(vs[j]); train += [x for i, x in enumerate(vs) if i != j]
            else:
                train += vs
        s = dict(seed=SPLIT_SEED, rule='1 validation video per category (PTZ excluded), rest training', val=val, train=train,
                 created=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(sp, s)
        if verbose:
            P(f'split.json written ({len(train)} train / {len(val)} val videos)')
    if verbose:
        P(f'SPLIT  train {len(s["train"])} videos | val {len(s["val"])} videos: ' + ', '.join(s['val']))
    return s


def load_sets(a, split):
    vids = {}
    files = {os.path.basename(f)[:-4]: f for f in feat_files(a.work)}
    miss = [t for t in split['train'] + split['val'] if t not in files]
    if miss:
        P(f'  [warn] {len(miss)} split videos have no feature file (not extracted yet): {", ".join(miss[:6])}' + (' ...' if len(miss) > 6 else ''))
    bar = Bar(len(files), 'loading features')
    for t, f in files.items():
        if t in split['train'] or t in split['val']:
            vids[t] = load_video(f)
        bar.update()
    bar.close()
    tr = [vids[t] for t in split['train'] if t in vids]; va = [vids[t] for t in split['val'] if t in vids]
    if not tr or not va:
        raise SystemExit('need at least one training and one validation video with features')
    return tr, va


def training_matrix(tr, cap, seed):
    rng = np.random.RandomState(seed); X, y, vid = [], [], []
    for k, v in enumerate(tr):
        n = len(v['y'])
        if n == 0:
            continue
        idx = rng.choice(n, cap, replace=False) if (cap and n > cap) else np.arange(n)
        X.append(v['X'][idx].astype(np.float32)); y.append(v['y'][idx]); vid.append(np.full(len(idx), k, np.int32))
    X, y, vid = np.concatenate(X), np.concatenate(y), np.concatenate(vid)
    return X, y, vid


class ValSet:
    def __init__(self, va):
        self.va = va
        self.X = np.concatenate([v['X'].astype(np.float32) for v in va]) if va else None
        self.y = np.concatenate([v['y'] for v in va])
        self.vid = np.concatenate([np.full(len(v['y']), k, np.int32) for k, v in enumerate(va)])
        self.off = np.cumsum([0] + [len(v['y']) for v in va])
        self.w = balanced_weights(self.y, self.vid)

    def decisions(self, p, thr):
        return [p[self.off[k]:self.off[k + 1]] >= thr for k in range(len(self.va))]


def fit_mlp(Xtr, ytr, wtr, val, cols, mu, sd, cfg, seed, label=''):
    """Adam on the weighted BCE; early stopping on the balanced validation loss. Returns model, history, best epoch."""
    Xs = (Xtr[:, cols] - mu) / sd; Xv = (val.X[:, cols] - mu) / sd if val is not None else None
    net = MLP(len(cols), cfg.hidden, seed); opt = Adam(net.params(), cfg.lr)
    rng = np.random.RandomState(seed + 7); yt = ytr.astype(np.float32)
    best, best_ep, best_state, hist, bad = np.inf, 0, net.state(), [], 0
    epochs = cfg.epochs
    for ep in range(1, epochs + 1):
        t0 = time.time(); perm = rng.permutation(len(Xs)); tl = 0.0; nb = 0
        bar = Bar(int(np.ceil(len(perm) / cfg.batch)), f'{label} ep{ep}'[:16]) if cfg.verbose else None
        for s in range(0, len(perm), cfg.batch):
            j = perm[s:s + cfg.batch]
            loss, g = net.grads(Xs[j], yt[j], wtr[j], cfg.wd)
            opt.step(net.params(), g); tl += loss; nb += 1
            if bar: bar.update()
        if bar: bar.close()
        rec = dict(epoch=ep, train_loss=tl / max(nb, 1), sec=round(time.time() - t0, 1))
        if val is not None:
            p = net.predict(Xv); dm = dispute_metrics(p, val.y, val.w)
            ov, _ = summarize(val.va, val.decisions(p, 0.5))
            rec.update(val_loss=dm['logloss'], val_bal_acc=dm['bal_acc'], val_auc=dm['auc'], val_F=ov['F'], val_thesis=ov['thesis'])
            if cfg.verbose:
                P(f'    ep {ep:2d}  train loss {rec["train_loss"]:.4f} | val loss {dm["logloss"]:.4f}  bal-acc {dm["bal_acc"]:.3f}  '
                  f'AUC {dm["auc"]:.3f}  F(official) {ov["F"]:.4f}  F(per-frame) {ov["thesis"]:.4f}  ({rec["sec"]}s)')
            if dm['logloss'] < best - 1e-4:
                best, best_ep, best_state, bad = dm['logloss'], ep, net.state(), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    hist.append(rec); break
        else:
            best_ep, best_state = ep, net.state()
        hist.append(rec)
    net.load(best_state)
    return net, hist, best_ep


def evaluate(net, val, cols, mu, sd, thr=0.5):
    p = net.predict((val.X[:, cols] - mu) / sd)
    dm = dispute_metrics(p, val.y, val.w)
    ov, per = summarize(val.va, val.decisions(p, thr))
    return p, dm, ov, per


def baselines(val):
    out = {}
    for s in ('cpb', 'guidance', 'framework', 'framework_blk', 'oracle'):
        decs = [base_decisions(v)[s] for v in val.va]
        ov, per = summarize(val.va, decs)
        d = np.concatenate(decs).astype(np.float32) if decs else np.zeros(0)
        dm = dispute_metrics(np.clip(d, 0.001, 0.999), val.y, val.w)
        out[s] = dict(overall=ov, per=per, bal_acc=dm['bal_acc'])
    return out


def cfg_from(a):
    c = argparse.Namespace(hidden=tuple(int(x) for x in a.hidden.split(',') if x), lr=a.lr, batch=a.batch, wd=a.wd,
                           epochs=a.epochs, patience=a.patience, verbose=a.verbose)
    return c


def standardizer(X, cols):
    mu = X[:, cols].mean(0); sd = X[:, cols].std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def print_baselines(B):
    P(f'  {"system (validation videos)":30s} {"F official":>10s} {"F per-frame":>11s} {"bal-acc on disputes":>20s}')
    for s, nm in (('cpb', 'CPB stage 1 (always CPB)'), ('guidance', 'guidance (always guidance)'),
                  ('framework', 'framework (paper stage 2)'), ('framework_blk', 'framework (block-repl.)'),
                  ('oracle', 'oracle (perfect arbiter)')):
        P(f'  {nm:30s} {B[s]["overall"]["F"]:10.4f} {B[s]["overall"]["thesis"]:11.4f} {B[s]["bal_acc"]:20.3f}')


# =============================================================================== ablation
def cmd_ablate(a):
    out = os.path.join(a.work, 'ablation'); os.makedirs(out, exist_ok=True); tee(os.path.join(out, 'log.txt'))
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} FEATURE ABLATION  seeds={a.seeds}  cap/video={a.cap}  hidden={a.hidden}')
    split = get_split(a.work); tr, va = load_sets(a, split)
    Xtr, ytr, vtr = training_matrix(tr, a.cap, 0); wtr = balanced_weights(ytr, vtr)
    val = ValSet(va)
    P(f'  training rows {len(ytr):,} from {len(tr)} videos ({100 * ytr.mean():.1f}% GT-fg) | validation rows {len(val.y):,} '
      f'from {len(va)} videos ({100 * val.y.mean():.1f}% GT-fg)')
    B = baselines(val); print_baselines(B)
    best_base = max(B[s]['overall']['F'] for s in ('cpb', 'guidance', 'framework', 'framework_blk'))
    orc = B['oracle']['overall']['F']
    base = ['core', 'C', 'G']
    configs = [('core only', ['core'], None), ('core+C (CPB only)', ['core', 'C'], None), ('core+G (guidance only)', ['core', 'G'], None),
               ('C+G', base, None), ('C+G+S', base + ['S'], None), ('C+G+S+L', base + ['S', 'L'], None),
               ('C+G+S+L+R', base + ['S', 'L', 'R'], None), ('C+G+S+L+R+T', base + ['S', 'L', 'R', 'T'], None),
               ('ALL', ALL_GROUPS, None)]
    configs += [(f'ALL - {g}', [x for x in ALL_GROUPS if x != g], None) for g in ALL_GROUPS if g != 'core']
    configs += [('ALL linear (logistic)', ALL_GROUPS, ())]
    if a.quick:
        configs = [c for c in configs if c[0] in ('core+C (CPB only)', 'core+G (guidance only)', 'C+G', 'ALL', 'ALL - R', 'ALL linear (logistic)')]
    results = []; nets = {}
    cfg = cfg_from(a); cfg.verbose = a.verbose
    bar = Bar(len(configs) * a.seeds, 'ablation')
    for name, groups, hidden in configs:
        feats = [f for g in groups for f in GROUPS[g]]; feats = [f for f in FEATURES if f in feats]
        cols = [FEATURES.index(f) for f in feats]; mu, sd = standardizer(Xtr, cols)
        c = argparse.Namespace(**vars(cfg)); c.hidden = cfg.hidden if hidden is None else hidden
        runs = []
        for s in range(a.seeds):
            t0 = time.time()
            net, hist, be = fit_mlp(Xtr, ytr, wtr, val, cols, mu, sd, c, seed=100 + s, label=name)
            p, dm, ov, per = evaluate(net, val, cols, mu, sd)
            runs.append(dict(F=ov['F'], thesis=ov['thesis'], loss=dm['logloss'], bal_acc=dm['bal_acc'], auc=dm['auc'],
                             best_epoch=be, sec=round(time.time() - t0, 1)))
            if name == 'ALL' and s == 0:
                nets['ALL'] = (net, cols, mu, sd)
            bar.update(extra=name)
        r = {k: float(np.mean([x[k] for x in runs])) for k in ('F', 'thesis', 'loss', 'bal_acc', 'auc', 'best_epoch', 'sec')}
        r.update(F_sd=float(np.std([x['F'] for x in runs])), name=name, groups=groups, n_feat=len(cols),
                 linear=(hidden == ()), runs=runs)
        results.append(r)
        sys.stdout.write('\n')
        P(f'  {name:26s} feats {len(cols):2d} | F {r["F"]:.4f} (+-{r["F_sd"]:.4f})  per-frame {r["thesis"]:.4f}  '
          f'bal-acc {r["bal_acc"]:.3f}  AUC {r["auc"]:.3f}  loss {r["loss"]:.4f}  ({r["sec"]:.0f}s/seed)')
    bar.close()
    allF = next(r['F'] for r in results if r['name'] == 'ALL')
    # ---------------- permutation importance (ALL model, first seed) ----------------
    net, cols, mu, sd = nets['ALL']
    Xv = (val.X[:, cols] - mu) / sd
    p0 = net.predict(Xv); ov0, _ = summarize(val.va, val.decisions(p0, 0.5)); l0 = dispute_metrics(p0, val.y, val.w)['logloss']
    rng = np.random.RandomState(5); imp = []
    units = [(f, [FEATURES.index(f)]) for f in FEATURES] + [(f'[group {g}]', [FEATURES.index(f) for f in GROUPS[g]]) for g in ALL_GROUPS]
    bar = Bar(len(units) * a.perm_repeats, 'permutation')
    for nm, idx in units:
        dF, dL = [], []
        for _ in range(a.perm_repeats):
            Xp = Xv.copy(); perm = rng.permutation(len(Xp))
            ci = [cols.index(i) for i in idx]
            Xp[:, ci] = Xv[perm][:, ci]                  # same permutation for every column of a group
            pp = net.predict(Xp); ov, _ = summarize(val.va, val.decisions(pp, 0.5))
            dF.append(ov0['F'] - ov['F']); dL.append(dispute_metrics(pp, val.y, val.w)['logloss'] - l0)
            bar.update()
        imp.append(dict(unit=nm, dF=float(np.mean(dF)), dLoss=float(np.mean(dL))))
    bar.close()
    # ---------------- report ----------------
    P('\n' + '=' * 112)
    P(f'  FEATURE ABLATION on the {len(va)} validation videos  (F = official CDnet, category mean; per-frame = Zhou thesis; '
      f'working size)')
    P('=' * 112)
    print_baselines(B)
    P('  ' + '-' * 108)
    P(f'  {"configuration":26s} {"#f":>3s} {"F official":>10s} {"+-sd":>7s} {"per-frame":>9s} {"bal-acc":>8s} {"AUC":>6s} '
      f'{"val loss":>8s} {"vs ALL":>7s} {"vs best fixed":>13s} {"headroom":>9s}')
    for r in results:
        hr = (r['F'] - best_base) / (orc - best_base) if orc > best_base else float('nan')
        P(f'  {r["name"]:26s} {r["n_feat"]:3d} {r["F"]:10.4f} {r["F_sd"]:7.4f} {r["thesis"]:9.4f} {r["bal_acc"]:8.3f} {r["auc"]:6.3f} '
          f'{r["loss"]:8.4f} {r["F"] - allF:+7.4f} {r["F"] - best_base:+13.4f} {100 * hr:8.0f}%')
    P('  headroom = share of the gap between the best fixed system (CPB / guidance / framework) and the oracle that is recovered')
    P('\n  PERMUTATION IMPORTANCE (ALL model; drop when the feature is shuffled on validation; larger = more useful)')
    P(f'  {"feature / group":28s} {"drop in F":>10s} {"rise in loss":>13s}')
    for r in sorted(imp, key=lambda r: -r['dLoss']):
        P(f'  {r["unit"]:28s} {r["dF"]:+10.4f} {r["dLoss"]:+13.4f}')
    P('\n  READING: keep a group if removing it (ALL - g) lowers F or raises the loss beyond the seed spread (+-sd);\n'
      '  features with ~0 permutation importance and no loss in "ALL - g" can be dropped.')
    with open(os.path.join(out, 'ablation.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['configuration', 'n_feat', 'F_official', 'F_sd', 'F_per_frame', 'bal_acc', 'auc', 'val_loss', 'best_epoch'])
        for r in results:
            w.writerow([r['name'], r['n_feat'], f'{r["F"]:.4f}', f'{r["F_sd"]:.4f}', f'{r["thesis"]:.4f}', f'{r["bal_acc"]:.4f}',
                        f'{r["auc"]:.4f}', f'{r["loss"]:.4f}', f'{r["best_epoch"]:.1f}'])
    with open(os.path.join(out, 'permutation_importance.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['feature_or_group', 'drop_in_F', 'rise_in_loss'])
        for r in sorted(imp, key=lambda r: -r['dLoss']):
            w.writerow([r['unit'], f'{r["dF"]:.5f}', f'{r["dLoss"]:.5f}'])
    save_json(os.path.join(out, 'ablation.json'), dict(
        split=split, baselines={s: B[s]['overall'] for s in B}, results=results, importance=imp,
        settings=dict(seeds=a.seeds, cap=a.cap, hidden=a.hidden, lr=a.lr, batch=a.batch, wd=a.wd, epochs=a.epochs, patience=a.patience)))
    P(f'\n  -> {out}/ablation.csv, permutation_importance.csv, ablation.json, log.txt')


# =============================================================================== final training
def cmd_train(a):
    out = os.path.join(a.work, 'models'); os.makedirs(out, exist_ok=True); tee(os.path.join(out, 'train_log.txt'))
    feats = parse_features(a.features); cols = [FEATURES.index(f) for f in feats]
    P(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")} TRAIN  features={a.features} ({len(feats)}): {", ".join(feats)}')
    P(f'    hidden={a.hidden}  lr={a.lr}  batch={a.batch}  wd={a.wd}  epochs<={a.epochs}  patience={a.patience}  seeds={a.seeds}  cap/video={a.cap}')
    split = get_split(a.work); tr, va = load_sets(a, split)
    Xtr, ytr, vtr = training_matrix(tr, a.cap, 0); wtr = balanced_weights(ytr, vtr); val = ValSet(va)
    P(f'  training rows {len(ytr):,} ({100 * ytr.mean():.1f}% GT-fg) | validation rows {len(val.y):,} ({100 * val.y.mean():.1f}% GT-fg)')
    B = baselines(val); print_baselines(B)
    mu, sd = standardizer(Xtr, cols); cfg = cfg_from(a); cfg.verbose = True
    runs = []; thr_grid = np.round(np.arange(0.10, 0.901, 0.025), 3)
    for s in range(a.seeds):
        P(f'\n  --- seed {s} ---')
        net, hist, be = fit_mlp(Xtr, ytr, wtr, val, cols, mu, sd, cfg, seed=100 + s, label=f'seed{s}')
        p, dm, ov, per = evaluate(net, val, cols, mu, sd, 0.5)
        Fthr = [summarize(val.va, val.decisions(p, t))[0]['F'] for t in thr_grid]
        runs.append(dict(net=net, hist=hist, best_epoch=be, p=p, dm=dm, ov=ov, per=per, Fthr=Fthr))
        P(f'  seed {s}: best epoch {be}  F(official) {ov["F"]:.4f}  per-frame {ov["thesis"]:.4f}  bal-acc {dm["bal_acc"]:.3f}  AUC {dm["auc"]:.3f}')
    Fm = np.mean([r['Fthr'] for r in runs], 0); thr = float(thr_grid[int(np.argmax(Fm))])
    ep_final = int(np.median([r['best_epoch'] for r in runs]))
    res = []
    for r in runs:
        ov, per = summarize(val.va, val.decisions(r['p'], thr)); res.append((ov, per))
    P('\n' + '=' * 104)
    P(f'  VALIDATION ({len(va)} videos, one per category)   threshold chosen on validation: {thr:.3f} (0.5 also shown)')
    P('=' * 104)
    print_baselines(B)
    F05 = [r['ov']['F'] for r in runs]; T05 = [r['ov']['thesis'] for r in runs]
    Ft = [x[0]['F'] for x in res]; Tt = [x[0]['thesis'] for x in res]
    P(f'  {"MLP arbiter, thr 0.5":30s} {np.mean(F05):10.4f} {np.mean(T05):11.4f}   (+-{np.std(F05):.4f} over {a.seeds} seeds)')
    P(f'  {"MLP arbiter, thr " + f"{thr:.3f}":30s} {np.mean(Ft):10.4f} {np.mean(Tt):11.4f}   (+-{np.std(Ft):.4f})')
    fw = B['framework']['overall']['F']; gd = B['guidance']['overall']['F']; cp = B['cpb']['overall']['F']; orc = B['oracle']['overall']['F']
    P(f'  gain vs framework {np.mean(Ft) - fw:+.4f} | vs guidance {np.mean(Ft) - gd:+.4f} | vs CPB {np.mean(Ft) - cp:+.4f} | '
      f'oracle gap left {orc - np.mean(Ft):.4f}')
    P(f'\n  per validation video (official F; MLP = seed mean at thr {thr:.3f}):')
    P(f'  {"video":42s} {"CPB":>6s} {"guid":>6s} {"frmwk":>6s} {"blk":>6s} {"MLP":>6s} {"oracle":>6s} {"disputes":>9s}')
    for v in va:
        t = v['tag']; mF = np.nanmean([x[1][t]['F'] for x in res])
        P(f'  {t:42s} {B["cpb"]["per"][t]["F"]:6.3f} {B["guidance"]["per"][t]["F"]:6.3f} {B["framework"]["per"][t]["F"]:6.3f} '
          f'{B["framework_blk"]["per"][t]["F"]:6.3f} {mF:6.3f} {B["oracle"]["per"][t]["F"]:6.3f} {len(v["y"]):9d}')
    best = int(np.argmax(Ft)); net = runs[best]['net']
    tag = a.name or a.features.replace(',', '+').replace('-', 'no')
    meta = dict(features=feats, hidden=list(cfg.hidden), threshold=thr, best_epoch=runs[best]['best_epoch'], seed=100 + best,
                val_F=Ft[best], val_F_mean=float(np.mean(Ft)), val_F_sd=float(np.std(Ft)), val_thesis=Tt[best], split=split,
                settings=dict(lr=a.lr, batch=a.batch, wd=a.wd, epochs=a.epochs, patience=a.patience, cap=a.cap),
                baselines={s: B[s]['overall'] for s in B}, version=VERSION, trained_on='train videos only')
    path = os.path.join(out, f'mlp_{tag}.npz')
    save_npz(path, mu=mu, sd=sd, **{f'p{i}': p for i, p in enumerate(net.params())}, n_layers=np.array(len(net.W)))
    save_json(path[:-4] + '.json', meta)
    P(f'\n  model (train videos only, best seed) -> {path}')
    if a.refit_all:
        P(f'\n  REFIT on train + validation videos, {ep_final} epochs (median best epoch), threshold {thr:.3f} kept')
        Xa, ya, va_ = training_matrix(tr + va, a.cap, 0); wa = balanced_weights(ya, va_)
        mu2, sd2 = standardizer(Xa, cols); c2 = argparse.Namespace(**vars(cfg)); c2.epochs = ep_final
        net2, _, _ = fit_mlp(Xa, ya, wa, None, cols, mu2, sd2, c2, seed=100 + best, label='refit')
        path2 = os.path.join(out, f'mlp_{tag}_final.npz')
        save_npz(path2, mu=mu2, sd=sd2, **{f'p{i}': p for i, p in enumerate(net2.params())}, n_layers=np.array(len(net2.W)))
        save_json(path2[:-4] + '.json', dict(meta, trained_on='train + validation videos', epochs=ep_final))
        P(f'  final model for the external datasets -> {path2}')
    save_json(os.path.join(out, f'history_{tag}.json'), [r['hist'] for r in runs])


def cmd_split(a):
    tee(os.path.join(a.work, 'extract_log.txt')); s = get_split(a.work)
    for k in ('train', 'val'):
        P(f'  {k}: ' + ', '.join(s[k]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--root', required=True, help='CDnet2014 folder'); e.add_argument('--cache', required=True, help='DeepLab guidance cache folder')
    e.add_argument('--cpb_out', default='cdnet_runs', help='--out folder of cdnet_cpb_eval.py (holds trained_models/)')
    e.add_argument('--work', default='arbiter_runs'); e.add_argument('--n_frames', type=int, default=150, help='frames sampled per video')
    e.add_argument('--budget', type=float, default=160 * 120); e.add_argument('--only', default='')
    s = sub.add_parser('split'); s.add_argument('--work', default='arbiter_runs')
    for nm in ('ablate', 'train'):
        t = sub.add_parser(nm)
        t.add_argument('--work', default='arbiter_runs')
        t.add_argument('--hidden', default='64,64'); t.add_argument('--lr', type=float, default=2e-3)
        t.add_argument('--batch', type=int, default=4096); t.add_argument('--wd', type=float, default=1e-5)
        t.add_argument('--epochs', type=int, default=30); t.add_argument('--patience', type=int, default=4)
        t.add_argument('--cap', type=int, default=40000, help='max training rows per video')
        if nm == 'ablate':
            t.add_argument('--seeds', type=int, default=2); t.add_argument('--perm_repeats', type=int, default=2)
            t.add_argument('--quick', action='store_true', help='6 configurations only (smoke test)')
            t.add_argument('--verbose', action='store_true', help='print every epoch')
        else:
            t.add_argument('--features', default='ALL', help="ALL | groups 'core,C,G,R' | 'ALL,-A' | feature names")
            t.add_argument('--seeds', type=int, default=3); t.add_argument('--name', default='')
            t.add_argument('--refit_all', type=int, default=1, help='1 = also refit on train+val for the external tests')
    a = ap.parse_args()
    if a.cmd in ('ablate', 'train'):
        a.verbose = getattr(a, 'verbose', True)
    {'extract': cmd_extract, 'split': cmd_split, 'ablate': cmd_ablate, 'train': cmd_train}[a.cmd](a)


if __name__ == '__main__':
    main()
