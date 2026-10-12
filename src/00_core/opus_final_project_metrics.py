"opus_final_project_metrics"


#!/usr/bin/env python3
"""
================================================================================
DRL GATE AGENT (junior v4)  —  patch-CNN policy + persistence + per-frame baseline
================================================================================
v3 (3-way gate + soft-F1 reward + 22-D state + CV) PLUS:

 (1) PATCH-CNN POLICY BRANCH. For each disputed pixel the policy now sees, besides
     the scalar state, a small (2k+1)x(2k+1) patch from 3 maps (CPB score, guidance
     probability, gray) run through a tiny CNN, whose embedding is concatenated
     with the scalars. This lets the agent read the SHAPE of the local CPB firing
     (compact blob = object; scattered speckle = turbulence noise) — the biggest
     untapped cue for the low-recall categories (turbulence/thermal/IOM).
     Toggle off with --no_patch (falls back to the pure-scalar MLP).

 (2) TEMPORAL PERSISTENCE feature (score_win_persist, in the env, 23-D state):
     fraction of the temporal window the pixel's CPB score exceeded 0.5. Objects
     persist; turbulence flickers. Computed in the trainer's in-order pass.

 (3) REWARD/EXPLORATION fixes (from the reward analysis):
     * PER-FRAME advantage baseline (default --baseline frame): centre the
       advantage by THIS frame's own mean reward instead of a global EMA. The old
       global EMA systematically pushed the agent AWAY from its correct decisions
       on low-reward (hard) scenes like turbulence/night — a real drag on those
       categories. Per-frame centring removes that bias and lowers variance.
       (--baseline ema restores the old behaviour for ablation.)
     * w_conf lowered 0.10 -> 0.05: the confidence-shaping term is a mild
       anti-exploration pressure; softening it lets the agent keep exploring the
       hard disputes longer.

Everything else (3-way gate, per-scene soft-F1 reward, harm-aware selection,
k-fold CV, self-contained 3-way mask) is unchanged.

NOTE: state grew to 23 scalars AND the policy architecture changed, so a v4
checkpoint is NOT compatible with v3. Re-train (no re-precompute; cache reused).
The patch-CNN makes each step heavier -> training is ~2x slower. Run --stage train
on the single split first to sanity-check speed/behaviour before the full CV.

USAGE
  python opus_final_drl_train_junior.py --stage train --models_dir models/cpb_numba \
      --cache_dir drl_cache_junior --steps 40000 --win_size 10 --debug --save_dir models/drl_gate_v4
  python opus_final_drl_train_junior.py --stage test  --models_dir models/cpb_numba \
      --cache_dir drl_cache_junior --save_dir models/drl_gate_v4
  python opus_final_drl_train_junior.py --stage cv --models_dir models/cpb_numba \
      --cache_dir drl_cache_junior --cv_folds 5 --steps 20000 --win_size 10 --save_dir models/drl_gate_cv4
================================================================================
"""

import argparse, json, os, pickle, re, random
from collections import deque
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import opus_drl_env_junior_enhanced_with_patch_cnn as ENV
try:
    import opus_final_hod_framework as F
except Exception:
    import opus_liang_hod_framework as F

SEED = 1234
N_ACTIONS = 3            # 0: trust CPB | 1: trust guidance | 2: keep baseline


# ============================================================================
# SELF-CONTAINED 3-WAY GATE (works even with an env lacking segment_gate3)
# ============================================================================
def _gate3_labels(e, actions, rows, cols):
    a = np.asarray(actions)
    cpb_lab = e.s1_default[rows, cols] > 0
    guid_lab = e.guid_fg[rows, cols] > 0
    base_lab = e.base_pred_fit[rows, cols]
    return np.where(a == 0, cpb_lab, np.where(a == 1, guid_lab, base_lab))


def _segment_gate3(e, actions, rows, cols):
    lab = _gate3_labels(e, actions, rows, cols)
    final = e.s1_default.copy()
    final[rows, cols] = (lab.astype(np.uint8) * 255)
    return e._to_native(final)


# ============================================================================
# SPLITS
# ============================================================================
def make_split(train=0.6, val=0.2, seed=SEED):
    rng = random.Random(seed)
    split = {'train': [], 'val': [], 'test': []}
    for cat, scenes in F.CDNET2014.items():
        sc = list(scenes); rng.shuffle(sc)
        n = len(sc); ntr = max(1, int(round(n * train))); nva = max(1, int(round(n * val)))
        ntr = min(ntr, n - 2) if n >= 3 else 1
        nva = min(nva, n - ntr - 1) if n - ntr >= 2 else max(0, n - ntr - 1)
        for i, s in enumerate(sc):
            part = 'train' if i < ntr else ('val' if i < ntr + nva else 'test')
            split[part].append(f'{cat}/{s}')
    return split


def make_kfold(k, seed=SEED):
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for cat, scenes in F.CDNET2014.items():
        sc = list(scenes); rng.shuffle(sc)
        for i, s in enumerate(sc):
            folds[i % k].append(f'{cat}/{s}')
    return folds


def carve_val(rest_ids, seed):
    rng = random.Random(seed)
    by_cat = {}
    for sid in rest_ids:
        by_cat.setdefault(sid.split('/')[0], []).append(sid)
    val = []
    for cat, lst in by_cat.items():
        if len(lst) >= 2:
            lst = list(lst); rng.shuffle(lst)
            val.append(lst[0])
    return set(val)


# ============================================================================
# DATA HELPERS
# ============================================================================
def roi(scene_path):
    f = os.path.join(scene_path, 'temporalROI.txt')
    if not os.path.exists(f):
        return 1, 99999
    t = open(f).read().strip().replace('-', ' ').split()
    return (int(t[0]), int(t[1])) if len(t) >= 2 else (1, int(t[0]) if t else 99999)


def frame_nums(scene_path, n_frames):
    idir = os.path.join(scene_path, 'input'); gdir = os.path.join(scene_path, 'groundtruth')
    rs, re_ = roi(scene_path)
    ns = []
    for fn in os.listdir(idir):
        if fn.lower().endswith(('.jpg', '.png')):
            m = re.findall(r'\d+', fn)
            if m and rs <= int(m[-1]) <= re_ and os.path.exists(os.path.join(gdir, f'gt{int(m[-1]):06d}.png')):
                ns.append(int(m[-1]))
    ns = sorted(ns)
    if n_frames and len(ns) > n_frames:
        ns = ns[::max(1, len(ns) // n_frames)][:n_frames]
    return ns


def load_bgr(idir, n):
    for pat in (f'in{n:06d}.jpg', f'in{n:05d}.jpg', f'{n:06d}.jpg', f'in{n:06d}.png'):
        p = os.path.join(idir, pat)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return im
    return None


def load_gt(gdir, n):
    for pat in (f'gt{n:06d}.png', f'gt{n:05d}.png'):
        p = os.path.join(gdir, pat)
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None


def find_root():
    return F.find_root()


# ============================================================================
# PRECOMPUTE : guidance binary mask + soft PROBABILITY (graceful fallback)
# ============================================================================
_ADE20K_FG = [12, 20, 21, 33, 39, 41, 80, 83, 84, 86, 91, 99]


def _guidance_prob(guidance, bgr):
    model = None
    for attr in ('model', 'segmentor', 'net', '_model'):
        m = getattr(guidance, attr, None)
        if m is not None:
            model = m; break
    if model is not None:
        try:
            import torch
            from mmseg.apis import inference_model
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                res = inference_model(model, rgb)
            logits = res.seg_logits.data
            prob = torch.softmax(logits.float(), dim=0)
            fg = prob[_ADE20K_FG].sum(0).cpu().numpy().astype(np.float32)
            return np.clip(fg, 0.0, 1.0)
        except Exception:
            pass
    for meth in ('predict_proba', 'predict_prob', 'proba'):
        fn = getattr(guidance, meth, None)
        if callable(fn):
            try:
                p = np.asarray(fn(bgr), np.float32)
                if p.ndim == 2 and np.isfinite(p).all():
                    if p.max() > 1.5:
                        p = 1.0 / (1.0 + np.exp(-p))
                    return np.clip(p, 0, 1)
            except Exception:
                pass
    return None


def precompute(args):
    root = find_root(); os.makedirs(args.cache_dir, exist_ok=True)
    guidance = F.make_guidance(args.guidance, False, models_dir=args.guidance_models_dir,
                               allow_fallback=False, device=args.device)
    got_prob = False
    for sid in tqdm([s for v in make_split().values() for s in v], desc='precompute guidance', ncols=100):
        cat, scene = sid.split('/')
        out = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
        if os.path.exists(out) and not args.force:
            continue
        mp = os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')
        sp = os.path.join(root, cat, scene)
        if not (os.path.exists(mp) and os.path.isdir(sp)):
            tqdm.write(f'  [skip] missing {sid}'); continue
        nums = frame_nums(sp, args.n_frames); idir = os.path.join(sp, 'input')
        guids, probs = [], []
        for n in nums:
            bgr = load_bgr(idir, n)
            guids.append(guidance.predict(bgr) if bgr is not None else None)
            p = _guidance_prob(guidance, bgr) if bgr is not None else None
            probs.append(p)
        if not guids:
            continue
        H, W = guids[0].shape
        arr = np.stack([g if g is not None else np.zeros((H, W), np.uint8) for g in guids]).astype(np.uint8)
        payload = {'nums': np.array(nums), 'guid': arr}
        if all(p is not None for p in probs):
            payload['prob'] = np.stack([cv2.resize(p, (W, H)) if p.shape != (H, W) else p
                                        for p in probs]).astype(np.float16)
            got_prob = True
        np.savez_compressed(out, **payload)
        tqdm.write(f'  cached {sid} ({len(nums)} frames{" +prob" if "prob" in payload else " (binary only)"})')
    if not got_prob:
        tqdm.write('  [note] guidance object exposed no probability method -> cache is BINARY only; '
                   'guid_prob will fall back to the binary mask (geometry features still help).')


# ============================================================================
# SCENE LOADER : temporal CPB window (mean / std / slope / persist)
# ============================================================================
def _win_stats(hist_stack):
    """hist_stack: [T,H,W] of CPB scores -> (mean, std, slope, persist) per pixel."""
    T = hist_stack.shape[0]
    mean = hist_stack.mean(0).astype(np.float32)
    std = hist_stack.std(0).astype(np.float32)
    persist = (hist_stack > 0.5).mean(0).astype(np.float32)     # fraction of window as foreground
    if T >= 2:
        t = np.arange(T, dtype=np.float32)
        tbar = t.mean(); denom = float(((t - tbar) ** 2).sum()) + 1e-6
        num = ((t - tbar)[:, None, None] * (hist_stack - mean[None])).sum(0)
        slope = (num / denom).astype(np.float32)
    else:
        slope = np.zeros_like(mean)
    return mean, std, slope, persist


class Scene:
    def __init__(self, sid, models_dir, root, envargs, cache_dir, win_size):
        self.sid = sid; cat, scene = sid.split('/')
        self.model = pickle.load(open(os.path.join(models_dir, f'cpb_paper_{cat}_{scene}.pkl'), 'rb'))
        sp = os.path.join(root, cat, scene)
        self.idir = os.path.join(sp, 'input'); self.gdir = os.path.join(sp, 'groundtruth')
        cache = np.load(os.path.join(cache_dir, f'{cat}__{scene}.npz'))
        self.nums = list(cache['nums']); self.guid = cache['guid']
        self.prob = cache['prob'].astype(np.float32) if 'prob' in cache.files else None
        self.envargs = envargs
        s1p = dict(af=envargs['af'], floor=envargs['floor'], bth=envargs['bth'], ref=envargs['ref'])
        self._win = []; hist = deque(maxlen=win_size)
        for i, n in enumerate(self.nums):
            bgr = load_bgr(self.idir, n)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            score = ENV.compute_stage1(self.model, rgb, self.guid[i], **s1p)
            hist.append(score)
            self._win.append(_win_stats(np.stack(hist)))
        self._cache = {}

    def env(self, i):
        if i in self._cache:
            return self._cache[i]
        n = self.nums[i]
        bgr = load_bgr(self.idir, n); gt = load_gt(self.gdir, n)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        wm, ws, wsl, wp = self._win[i]
        e = ENV.FrameEnv(self.model, **self.envargs)
        e.set_frame(rgb, self.guid[i], gt,
                    guid_prob_native=(self.prob[i] if self.prob is not None else None),
                    win_mean=wm, win_std=ws, win_slope=wsl, win_persist=wp)
        if len(self._cache) < 400:
            self._cache[i] = e
        return e


def build_scenes(ids, args, root, envargs):
    out = []
    for sid in ids:
        cat, scene = sid.split('/')
        if os.path.exists(os.path.join(args.cache_dir, f'{cat}__{scene}.npz')) and \
           os.path.exists(os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')):
            out.append(Scene(sid, args.models_dir, root, envargs, args.cache_dir, args.win_size))
    return out


# ============================================================================
# PER-SCENE SOFT-F1 REWARD
# ============================================================================
def f1_marginal_reward(pred, gt, base_pred, chosen_conf, valid, w, agent_counts):
    TP, FP, FN = float(agent_counts[0]), float(agent_counts[1]), float(agent_counts[2])
    D = 2.0 * TP + FP + FN + 1e-6
    w_tp = 2.0 * (FP + FN) / (D * D)
    w_pen = 2.0 * TP / (D * D)
    norm = max(w_tp, w_pen, 1e-12)
    w_tp /= norm; w_pen /= norm

    fg = gt; bg = ~gt
    r = np.zeros(pred.shape, np.float32)
    r[fg & pred] = w_tp
    r[fg & ~pred] = -w_pen
    r[bg & pred] = -w_pen

    correct = (pred == gt)
    base_correct = (base_pred == gt)
    r = r + np.where(correct & ~base_correct, w['fix'], 0.0)
    r = r + np.where((~correct) & base_correct, -w['break'], 0.0)
    r = r + np.where(correct, w['conf'] * chosen_conf, -0.5 * w['conf'] * chosen_conf)

    r = (r * valid.astype(np.float32)).astype(np.float32)
    v = valid if valid.any() else np.ones_like(valid)
    fgv = gt & v; nfg = int(fgv.sum())
    comp = {
        'reward': float(r[v].mean()),
        'frac_correct': float(correct[v].mean()),
        'frac_fixed': float((correct & ~base_correct)[v].mean()),
        'frac_broken': float(((~correct) & base_correct)[v].mean()),
        'frac_fg': float(gt[v].mean()),
        'fg_recall': float(pred[fgv].mean()) if nfg else 0.0,
        'bg_correct': float((~pred[(~gt) & v]).mean()) if ((~gt) & v).any() else 0.0,
    }
    return r, comp


def gate_reward(pred, gt, base_pred, chosen_conf, valid, w):
    correct = (pred == gt)
    base_correct = (base_pred == gt)
    cw = np.where(gt, w['fg_weight'], 1.0)
    core = np.where(correct, w['correct'], -w['wrong'])
    core = core + np.where(correct & ~base_correct, w['fix'], 0.0)
    core = core + np.where((~correct) & base_correct, -w['break'], 0.0)
    conf = np.clip(chosen_conf, 0.0, 1.0)
    core = core + np.where(correct, w['conf'] * conf, -0.5 * w['conf'] * conf)
    r = (core * cw).astype(np.float32) * valid.astype(np.float32)
    v = valid if valid.any() else np.ones_like(valid)
    fgv = gt & v; nfg = int(fgv.sum())
    comp = {
        'reward': float(r[v].mean()),
        'frac_correct': float(correct[v].mean()),
        'frac_fixed': float((correct & ~base_correct)[v].mean()),
        'frac_broken': float(((~correct) & base_correct)[v].mean()),
        'frac_fg': float(gt[v].mean()),
        'fg_recall': float(pred[fgv].mean()) if nfg else 0.0,
        'bg_correct': float((~pred[(~gt) & v]).mean()) if ((~gt) & v).any() else 0.0,
    }
    return r, comp


# ============================================================================
# POLICY  (scalar MLP + optional PATCH-CNN branch)
# ============================================================================
def build_gate(scalar_dim, n_actions=N_ACTIONS, abstain_init=1.5,
               use_patch=True, patch_k=4, patch_c=ENV.PATCH_CHANNELS):
    import torch, torch.nn as nn

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_patch = bool(use_patch)
            feat = scalar_dim
            if self.use_patch:
                self.cnn = nn.Sequential(
                    nn.Conv2d(patch_c, 8, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1))
                feat = feat + 16
            self.head = nn.Sequential(nn.Linear(feat, 64), nn.ReLU(),
                                      nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                if n_actions == 3:
                    self.head[-1].bias[2] = float(abstain_init)

        def forward(self, s, patch=None):
            if self.use_patch and patch is not None:
                z = self.cnn(patch).flatten(1)
                s = torch.cat([s, z], dim=1)
            return self.head(s)

    return Gate()


class RunningNorm:
    def __init__(self, d):
        self.n = 0; self.mean = np.zeros(d); self.M2 = np.ones(d); self.frozen = False

    def update(self, X):
        if self.frozen or len(X) == 0:
            return
        X = np.asarray(X, np.float64); m = len(X)
        bmean = X.mean(0); bM2 = ((X - bmean) ** 2).sum(0)
        delta = bmean - self.mean; tot = self.n + m
        self.mean += delta * (m / tot); self.M2 += bM2 + delta ** 2 * (self.n * m / tot); self.n = tot

    def __call__(self, X):
        std = np.sqrt(self.M2 / max(self.n - 1, 1) + 1e-8)
        return np.clip((X - self.mean) / std, -5, 5).astype(np.float32)


# ============================================================================
# FULL-PIPELINE CDnet EVALUATION (per scene/category)
# ============================================================================
def evaluate(gate, norm, scenes, args):
    import torch
    per_cat = {}; A = [0, 0, 0, 0]; B = [0, 0, 0, 0]
    act_tot = np.zeros(N_ACTIONS, np.int64)
    scene_f1 = {}
    for sc in scenes:
        cat = sc.sid.split('/')[0]
        a = [0, 0, 0, 0]; b = [0, 0, 0, 0]
        for i in range(len(sc.nums)):
            e = sc.env(i); r, c = e.dispute_ij()
            base = e.baseline_mask()
            if len(r) == 0:
                mask = base
            else:
                Sn = norm(e.state(r, c))
                Pt = torch.from_numpy(e.patches(r, c, args.patch_k)) if args.use_patch else None
                with torch.no_grad():
                    logits = gate(torch.from_numpy(Sn), Pt)
                    actions = logits.argmax(-1).numpy()
                for k in range(N_ACTIONS):
                    act_tot[k] += int((actions == k).sum())
                mask = _segment_gate3(e, actions, r, c)
            for k, v in enumerate(e.counts(mask)):
                a[k] += v
            for k, v in enumerate(e.counts(base)):
                b[k] += v
        scene_f1[sc.sid] = {'cat': cat, 'agent': ENV.f1_from_counts(*a), 'base': ENV.f1_from_counts(*b),
                            'a': [int(x) for x in a], 'b': [int(x) for x in b]}
        per_cat.setdefault(cat, {'a': [0, 0, 0, 0], 'b': [0, 0, 0, 0]})
        for k in range(4):
            per_cat[cat]['a'][k] += a[k]; per_cat[cat]['b'][k] += b[k]; A[k] += a[k]; B[k] += b[k]
    cats = {c: {'agent': ENV.f1_from_counts(*d['a']), 'base': ENV.f1_from_counts(*d['b'])}
            for c, d in per_cat.items()}
    gains = [d['agent'] - d['base'] for d in cats.values()]
    am = act_tot / max(int(act_tot.sum()), 1)
    # THREE aggregations, so the comparison to Liang is unambiguous:
    #   pooled   = micro-average over ALL scenes' counts  == Liang Table 2 "Overall"
    #   scene    = macro-average over scenes  (equal weight per scene)
    #   category = macro-average over the 11 categories   == official CDnet metric
    sm_agent = float(np.mean([v['agent'] for v in scene_f1.values()])) if scene_f1 else 0.0
    sm_base = float(np.mean([v['base'] for v in scene_f1.values()])) if scene_f1 else 0.0
    cm_agent = float(np.mean([d['agent'] for d in cats.values()])) if cats else 0.0
    cm_base = float(np.mean([d['base'] for d in cats.values()])) if cats else 0.0
    return {'f1': ENV.f1_from_counts(*A), 'base': ENV.f1_from_counts(*B),
            'dF1': ENV.f1_from_counts(*A) - ENV.f1_from_counts(*B),
            'mean_cat_gain': float(np.mean(gains)), 'worst_cat_gain': float(np.min(gains)),
            'agent_scene_mean': sm_agent, 'base_scene_mean': sm_base,
            'agent_cat_mean': cm_agent, 'base_cat_mean': cm_base,
            'A': [int(x) for x in A], 'B': [int(x) for x in B],
            'act_mix': {'cpb': float(am[0]), 'guid': float(am[1]), 'base': float(am[2])},
            'per_cat': cats, 'scene_f1': scene_f1}


# ============================================================================
# TRAIN ONE AGENT (shared by --stage train and each CV fold)
# ============================================================================
def train_one(args, train_sc, val_sc, save_dir, seed=SEED, tag=''):
    import torch
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 31)); random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    w = {'correct': args.w_correct, 'wrong': args.w_wrong, 'fix': args.w_fix,
         'break': args.w_break, 'conf': args.w_conf, 'fg_weight': args.fg_weight}
    reward_fn = f1_marginal_reward if args.reward_mode == 'f1marg' else gate_reward

    gate = build_gate(ENV.FrameEnv.STATE_DIM, N_ACTIONS, args.abstain_init, args.use_patch, args.patch_k)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    norm = RunningNorm(ENV.FrameEnv.STATE_DIM)

    ema = 0.0; best_sel = -1e9
    best_state = None
    wk = ['reward', 'frac_correct', 'frac_fixed', 'frac_broken', 'frac_fg', 'fg_recall',
          'bg_correct', 'dF1', 'act_cpb', 'act_guid', 'act_base', 'bc']
    win = {k: deque(maxlen=300) for k in wk}
    hist = {'step': [], 'val_dF1': [], 'val_f1': [], 'mean_cat_gain': [], 'worst_cat_gain': [],
            'reward': [], 'fg_recall': [], 'frac_broken': [], 'act_base': []}

    pbar = tqdm(range(1, args.steps + 1), desc=f'train{tag}', ncols=150)
    opt.zero_grad(); acc = 0.0; n_acc = 0
    entropy_coef = args.entropy
    for step in pbar:
        if args.steps > 1:
            entropy_coef = args.entropy + (args.entropy_final - args.entropy) * (step / args.steps)
        sc = random.choice(train_sc); i = random.randrange(len(sc.nums))
        e = sc.env(i); r, c = e.dispute_ij()
        if len(r) == 0:
            continue
        S = e.state(r, c); norm.update(S); Sn = norm(S)
        Pt = torch.from_numpy(e.patches(r, c, args.patch_k)) if args.use_patch else None
        logits = gate(torch.from_numpy(Sn), Pt)
        dist = torch.distributions.Categorical(logits=logits)
        a_t = dist.sample()
        logp = dist.log_prob(a_t); ent = dist.entropy().mean()
        actions = a_t.detach().numpy()
        probs = torch.softmax(logits, dim=-1).detach().numpy()
        chosen_conf = probs[np.arange(len(actions)), actions].astype(np.float32)

        mask = _segment_gate3(e, actions, r, c)
        pred = _gate3_labels(e, actions, r, c)
        gt = e.gtfg_fit[r, c]; base_pred = e.base_pred_fit[r, c]; valid = e.valid_fit[r, c]
        tp, fp, fn, tn = e.counts(mask)

        # ---- BEHAVIOR-CLONING AUXILIARY -------------------------------------
        # On a dispute exactly one primitive side matches GT; supervise the policy
        # toward it. Weight is annealed from --bc_coef (step 0) to --bc_final (end):
        # strong early to INJECT the (0.92-separable) arbitration signal that REINFORCE
        # extracts poorly, then decayed so the F1 reward + the defer-to-baseline action
        # can override it on scenes where the rule inverts (tramstop/streetLight). The
        # BC target is the GT-matching side {0=CPB,1=guid}; action 2 stays RL-only.
        if args.bc_coef != 0.0 or args.bc_final != 0.0:
            guid_lab = e.guid_fg[r, c] > 0
            oracle_side = torch.from_numpy(np.where(guid_lab == gt, 1, 0).astype(np.int64))
            ce = -dist.log_prob(oracle_side)                    # per-pixel cross-entropy
            vmb = torch.from_numpy(valid.astype(np.float32))
            bc_loss = (ce * vmb).sum() / (vmb.sum() + 1.0)
            bc_c = args.bc_coef + (args.bc_final - args.bc_coef) * (step / max(args.steps, 1))
            bc_term = bc_c * bc_loss
        else:
            bc_term = 0.0; bc_loss = None

        if args.reward_mode == 'f1marg':
            rew_px, comp = reward_fn(pred, gt, base_pred, chosen_conf, valid, w, (tp, fp, fn))
        else:
            rew_px, comp = reward_fn(pred, gt, base_pred, chosen_conf, valid, w)
        comp['bc'] = float(bc_loss.item()) if bc_loss is not None else 0.0

        # ---- advantage baseline (FIX: per-frame centring by default) ----
        ema = 0.99 * ema + 0.01 * comp['reward']
        if args.baseline == 'frame':
            adv_np = rew_px - comp['reward']       # centre by THIS frame's own mean reward
        else:
            adv_np = rew_px - ema                  # global EMA (old behaviour, for ablation)
        adv = torch.from_numpy(adv_np.astype(np.float32))
        vmask = torch.from_numpy(valid.astype(np.float32))
        acc = acc + (-(adv * logp) * vmask).sum() / (vmask.sum() + 1.0) - entropy_coef * ent + bc_term
        n_acc += 1

        av = valid
        comp['act_cpb'] = float((actions == 0)[av].mean()) if av.any() else 0.0
        comp['act_guid'] = float((actions == 1)[av].mean()) if av.any() else 0.0
        comp['act_base'] = float((actions == 2)[av].mean()) if av.any() else 0.0
        for k in wk:
            win[k].append((e.f1(mask) - e.baseline_f1()) if k == 'dF1' else comp[k])

        if n_acc >= args.batch_frames:
            acc.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step(); opt.zero_grad(); acc = 0.0; n_acc = 0

        pbar.set_postfix({'rew': f'{np.mean(win["reward"]):+.2f}', 'dF1': f'{np.mean(win["dF1"]):+.4f}',
                          'bc': f'{np.mean(win["bc"]):.3f}',
                          'fgRec': f'{np.mean(win["fg_recall"]):.2f}', 'brk': f'{np.mean(win["frac_broken"]):.3f}',
                          'act(C/G/B)': f'{np.mean(win["act_cpb"]):.2f}/{np.mean(win["act_guid"]):.2f}/{np.mean(win["act_base"]):.2f}',
                          'ent': f'{entropy_coef:.4f}', 'best': f'{best_sel:+.4f}' if best_sel > -1e8 else 'n/a'})

        if args.debug and step % args.debug_every == 0:
            tqdm.write(f'  [dbg{tag} {step}] reward={np.mean(win["reward"]):+.2f} | %corr={np.mean(win["frac_correct"]):.2f} '
                       f'fgRecall={np.mean(win["fg_recall"]):.2f} bgCorrect={np.mean(win["bg_correct"]):.2f} '
                       f'| %fix={np.mean(win["frac_fixed"]):.3f} %broke={np.mean(win["frac_broken"]):.3f} '
                       f'| act C/G/B={np.mean(win["act_cpb"]):.2f}/{np.mean(win["act_guid"]):.2f}/{np.mean(win["act_base"]):.2f}')

        if step % args.eval_every == 0:
            norm.frozen = True; vg = evaluate(gate, norm, val_sc, args); norm.frozen = False
            sel = vg['mean_cat_gain'] - args.harm_penalty * max(0.0, -vg['worst_cat_gain'])
            am = vg['act_mix']
            tqdm.write(f'\n[{tag} step {step}] VAL agent={vg["f1"]:.4f} base={vg["base"]:.4f} dF1={vg["dF1"]:+.4f} | '
                       f'mean_cat_gain={vg["mean_cat_gain"]:+.4f} worst_cat={vg["worst_cat_gain"]:+.4f} sel={sel:+.4f}'
                       f' | act C/G/B={am["cpb"]:.2f}/{am["guid"]:.2f}/{am["base"]:.2f}')
            gains = sorted(vg['per_cat'].items(), key=lambda kv: kv[1]['agent'] - kv[1]['base'])
            tqdm.write('   worst 4: ' + '  '.join(f'{c.split("/")[0][:4]}{(d["agent"]-d["base"]):+.3f}' for c, d in gains[:4])
                       + '   best 4: ' + '  '.join(f'{c.split("/")[0][:4]}{(d["agent"]-d["base"]):+.3f}' for c, d in gains[-4:]))
            for k, v in [('step', step), ('val_dF1', vg['dF1']), ('val_f1', vg['f1']),
                         ('mean_cat_gain', vg['mean_cat_gain']), ('worst_cat_gain', vg['worst_cat_gain']),
                         ('reward', float(np.mean(win['reward']))), ('fg_recall', float(np.mean(win['fg_recall']))),
                         ('frac_broken', float(np.mean(win['frac_broken']))), ('act_base', am['base'])]:
                hist[k].append(v)
            _plot(hist, os.path.join(save_dir, 'training_curves.png'))
            if sel > best_sel:
                best_sel = sel
                best_state = {'gate': {kk: vv.clone() for kk, vv in gate.state_dict().items()},
                              'norm_mean': norm.mean.copy(), 'norm_M2': norm.M2.copy(), 'norm_n': norm.n}
                torch.save({'gate': gate.state_dict(), 'norm_mean': norm.mean, 'norm_M2': norm.M2,
                            'norm_n': norm.n, 'step': step, 'val': vg, 'sel': sel, 'weights': w,
                            'reward_mode': args.reward_mode, 'n_actions': N_ACTIONS,
                            'state_dim': ENV.FrameEnv.STATE_DIM, 'use_patch': args.use_patch,
                            'patch_k': args.patch_k, 'bc_coef': args.bc_coef, 'bc_final': args.bc_final},
                           os.path.join(save_dir, 'best.pth'))
                tqdm.write(f'   * new best sel={sel:+.4f} [mean_gain={vg["mean_cat_gain"]:+.4f}, worst={vg["worst_cat_gain"]:+.4f}] -> best.pth')
    pbar.close()
    torch.save({'gate': gate.state_dict(), 'norm_mean': norm.mean, 'norm_M2': norm.M2, 'norm_n': norm.n,
                'step': args.steps, 'best_sel': best_sel, 'weights': w, 'reward_mode': args.reward_mode,
                'n_actions': N_ACTIONS, 'state_dim': ENV.FrameEnv.STATE_DIM, 'use_patch': args.use_patch,
                'patch_k': args.patch_k, 'history': hist},
               os.path.join(save_dir, 'final.pth'))
    json.dump(hist, open(os.path.join(save_dir, 'history.json'), 'w'), indent=2)
    if best_state is not None:
        gate.load_state_dict(best_state['gate'])
        norm.mean = best_state['norm_mean']; norm.M2 = best_state['norm_M2']; norm.n = best_state['norm_n']
    norm.frozen = True
    return gate, norm, best_sel


def train(args):
    root = find_root(); split = make_split()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    print('loading scenes (in-order temporal pass) ...')
    train_sc = build_scenes(split['train'], args, root, envargs)
    val_sc = build_scenes(split['val'], args, root, envargs)
    print(f'SPLIT train={len(train_sc)} val={len(val_sc)}  state_dim={ENV.FrameEnv.STATE_DIM}  '
          f'patch={args.use_patch}(k={args.patch_k})  baseline={args.baseline}  reward={args.reward_mode}  '
          f'win_size={args.win_size}  abstain_init={args.abstain_init}')
    if not train_sc:
        raise SystemExit('No training scenes cached — run --stage precompute first.')
    _, _, best_sel = train_one(args, train_sc, val_sc, args.save_dir, seed=SEED)
    print(f'\nDONE. best harm-aware sel = {best_sel:+.4f}  (checkpoints -> {args.save_dir})')


# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================
def _catmean(rows):
    d = {}
    for cat, b, a in rows:
        d.setdefault(cat, []).append((b, a))
    per_cat = {c: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for c, v in d.items()}
    base_cm = float(np.mean([v[0] for v in per_cat.values()]))
    agent_cm = float(np.mean([v[1] for v in per_cat.values()]))
    return base_cm, agent_cm, per_cat


def cross_validate(args):
    root = find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    folds = make_kfold(args.cv_folds, SEED)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'CROSS-VALIDATION  folds={args.cv_folds}  steps/fold={args.steps}  state_dim={ENV.FrameEnv.STATE_DIM}  '
          f'patch={args.use_patch}(k={args.patch_k})  baseline={args.baseline}  win_size={args.win_size}')
    for kf in range(args.cv_folds):
        print(f'  fold {kf}: test scenes = {folds[kf]}')

    scene_rows = {}
    fold_base_cm, fold_agent_cm = [], []
    fold_act = []
    poolA = [0, 0, 0, 0]; poolB = [0, 0, 0, 0]   # micro-average over ALL held-out scenes
    scene_agent_f1, scene_base_f1 = [], []       # macro-average over scenes

    for kf in range(args.cv_folds):
        test_ids = folds[kf]
        rest = [s for j in range(args.cv_folds) if j != kf for s in folds[j]]
        val_ids = carve_val(rest, seed=SEED + kf)
        train_ids = [s for s in rest if s not in val_ids]
        fold_dir = os.path.join(args.save_dir, f'fold{kf}')
        print(f'\n================  FOLD {kf}/{args.cv_folds - 1}  '
              f'(train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})  ================')
        train_sc = build_scenes(train_ids, args, root, envargs)
        val_sc = build_scenes(list(val_ids), args, root, envargs)
        if not train_sc or not val_sc:
            print('  [skip fold] missing cached scenes'); continue
        gate, norm, _ = train_one(args, train_sc, val_sc, fold_dir, seed=SEED + kf, tag=f'-f{kf}')
        del train_sc, val_sc

        test_sc = build_scenes(test_ids, args, root, envargs)
        fold_rows = []
        for sc in test_sc:
            rr = evaluate(gate, norm, [sc], args)
            row = (sc.sid.split('/')[0], rr['base'], rr['f1'])
            scene_rows[sc.sid] = row
            fold_rows.append(row)
            fold_act.append(rr['act_mix'])
            for k in range(4):
                poolA[k] += rr['A'][k]; poolB[k] += rr['B'][k]
            scene_agent_f1.append(rr['f1']); scene_base_f1.append(rr['base'])
        bcm, acm, _ = _catmean(fold_rows)
        fold_base_cm.append(bcm); fold_agent_cm.append(acm)
        print(f'  [fold {kf}] category-mean  base={bcm:.4f}  agent={acm:.4f}  gain={acm - bcm:+.4f}')
        del test_sc, gate, norm

    all_rows = [(c, b, a) for (c, b, a) in scene_rows.values()]
    base_cm, agent_cm, per_cat = _catmean(all_rows)
    print(f'\n{"=" * 74}\n  CROSS-VALIDATION RESULT  ({args.cv_folds}-fold, every scene tested once)\n{"=" * 74}')
    print(f"  {'category':<26}{'base':>9}{'agent':>9}{'gain':>9}")
    for c in sorted(per_cat):
        b, a = per_cat[c]
        print(f"  {c:<26}{b:>9.4f}{a:>9.4f}{a - b:>+9.4f}")
    print('  ' + '-' * 52)
    gains = [a - b for b, a in per_cat.values()]
    print(f"  {'CATEGORY-MEAN (CDnet)':<26}{base_cm:>9.4f}{agent_cm:>9.4f}{agent_cm - base_cm:>+9.4f}")
    print(f"  worst category gain = {min(gains):+.4f}   |   categories improved = {sum(g > 0 for g in gains)}/{len(gains)}")
    fb = np.array(fold_base_cm); fa = np.array(fold_agent_cm)
    print(f"\n  Per-fold category-mean (mean +/- std over {len(fa)} folds):")
    print(f"    base  = {fb.mean():.4f} +/- {fb.std():.4f}")
    print(f"    agent = {fa.mean():.4f} +/- {fa.std():.4f}")
    print(f"    gain  = {(fa - fb).mean():+.4f} +/- {(fa - fb).std():.4f}")
    am = {k: float(np.mean([d[k] for d in fold_act])) for k in ('cpb', 'guid', 'base')} if fold_act else {}
    if am:
        print(f"  action mix  trust-CPB={am['cpb']:.2f}  trust-guid={am['guid']:.2f}  keep-baseline={am['base']:.2f}")

    # ---- THREE aggregations over ALL held-out scenes (every scene tested once) ----
    pooled_base = ENV.f1_from_counts(*poolB); pooled_agent = ENV.f1_from_counts(*poolA)
    sm_base = float(np.mean(scene_base_f1)) if scene_base_f1 else 0.0
    sm_agent = float(np.mean(scene_agent_f1)) if scene_agent_f1 else 0.0
    print(f"\n{'='*74}\n  ALL-SCENES SUMMARY  (every scene held out once — comparable to Liang)")
    print(f"  {'aggregation':<28}{'base':>10}{'agent':>10}{'gain':>10}")
    print(f"  {'pooled  (= Liang Overall)':<28}{pooled_base:>10.4f}{pooled_agent:>10.4f}{pooled_agent-pooled_base:>+10.4f}")
    print(f"  {'scene-mean':<28}{sm_base:>10.4f}{sm_agent:>10.4f}{sm_agent-sm_base:>+10.4f}")
    print(f"  {'category-mean (CDnet macro)':<28}{base_cm:>10.4f}{agent_cm:>10.4f}{agent_cm-base_cm:>+10.4f}")
    print(f"{'='*74}")
    print(f"  Liang CPB+DeepLabV3+ 'Overall' (POOLED) = 0.753   (PSPNet-guided = 0.772)")
    print(f"  ^ compare against the POOLED row above. Category-mean is the official CDnet metric;")
    print(f"    Liang does not report it, so it is NOT directly comparable to 0.753.")

    out = {'cv_folds': args.cv_folds, 'steps_per_fold': args.steps, 'win_size': args.win_size,
           'use_patch': args.use_patch, 'patch_k': args.patch_k, 'baseline': args.baseline,
           'pooled': {'base': pooled_base, 'agent': pooled_agent, 'gain': pooled_agent - pooled_base},
           'scene_mean': {'base': sm_base, 'agent': sm_agent, 'gain': sm_agent - sm_base},
           'category_mean': {'base': base_cm, 'agent': agent_cm, 'gain': agent_cm - base_cm},
           'liang_pooled_overall': {'deeplabv3+': 0.753, 'pspnet': 0.772},
           'per_category': {c: {'base': b, 'agent': a, 'gain': a - b} for c, (b, a) in per_cat.items()},
           'per_fold': {'base_cm': fold_base_cm, 'agent_cm': fold_agent_cm},
           'worst_category_gain': float(min(gains)),
           'scene_f1': {sid: {'cat': c, 'base': b, 'agent': a} for sid, (c, b, a) in scene_rows.items()}}
    json.dump(out, open(os.path.join(args.save_dir, 'cv_results.json'), 'w'), indent=2)
    print(f"\n  saved -> {os.path.join(args.save_dir, 'cv_results.json')}")


def _plot(hist, path):
    if len(hist['step']) < 2:
        return
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    ax[0, 0].plot(hist['step'], hist['mean_cat_gain'], 'r-o', ms=3, label='mean cat gain')
    ax[0, 0].plot(hist['step'], hist['worst_cat_gain'], 'b-o', ms=3, label='worst cat gain')
    ax[0, 0].axhline(0, color='k', ls='--', alpha=.4); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 0].set_title('category-mean & worst-category gain')
    ax[0, 1].plot(hist['step'], hist['val_f1'], 'g-o', ms=3); ax[0, 1].grid(alpha=.3)
    ax[0, 1].set_title('VAL pooled F1'); ax[0, 1].set_ylim(0, 1)
    ax[0, 2].plot(hist['step'], hist['val_dF1'], 'purple'); ax[0, 2].axhline(0, color='k', ls='--', alpha=.4)
    ax[0, 2].grid(alpha=.3); ax[0, 2].set_title('VAL pooled dF1')
    ax[1, 0].plot(hist['step'], hist['reward'], 'orange'); ax[1, 0].grid(alpha=.3); ax[1, 0].set_title('mean reward (soft-F1)')
    ax[1, 1].plot(hist['step'], hist['fg_recall'], 'c-'); ax[1, 1].grid(alpha=.3); ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_title('FG recall on disputes (precious!)')
    ax[1, 2].plot(hist['step'], hist['frac_broken'], 'r-', label='broke (train win)')
    ax[1, 2].plot(hist['step'], hist['act_base'], 'g-', label='keep-baseline frac (VAL)')
    ax[1, 2].legend(); ax[1, 2].grid(alpha=.3); ax[1, 2].set_ylim(0, 1)
    ax[1, 2].set_title('do-no-harm: broken vs keep-baseline')
    for a in ax.ravel():
        a.set_xlabel('step')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def test(args):
    import torch
    root = find_root(); split = make_split()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    scenes = build_scenes(split['test'], args, root, envargs)
    ck = torch.load(os.path.join(args.save_dir, 'best.pth'), map_location='cpu')
    if ck.get('state_dim', ENV.FrameEnv.STATE_DIM) != ENV.FrameEnv.STATE_DIM:
        raise SystemExit(f"checkpoint state_dim={ck.get('state_dim')} != env STATE_DIM={ENV.FrameEnv.STATE_DIM}. "
                         f"Re-train with this env version.")
    # rebuild architecture exactly as trained
    args.use_patch = ck.get('use_patch', False); args.patch_k = ck.get('patch_k', args.patch_k)
    gate = build_gate(ck['state_dim'], ck.get('n_actions', N_ACTIONS), 0.0, args.use_patch, args.patch_k)
    gate.load_state_dict(ck['gate'])
    norm = RunningNorm(ENV.FrameEnv.STATE_DIM)
    norm.mean = ck['norm_mean']; norm.M2 = ck['norm_M2']; norm.n = ck['norm_n']; norm.frozen = True
    res = evaluate(gate, norm, scenes, args)
    am = res['act_mix']
    print(f"\n{'='*74}\n  TEST (held-out scenes)")
    print(f"  {'aggregation':<28}{'base':>10}{'agent':>10}{'gain':>10}")
    print(f"  {'pooled  (= Liang Overall)':<28}{res['base']:>10.4f}{res['f1']:>10.4f}{res['dF1']:>+10.4f}")
    print(f"  {'scene-mean':<28}{res['base_scene_mean']:>10.4f}{res['agent_scene_mean']:>10.4f}"
          f"{res['agent_scene_mean']-res['base_scene_mean']:>+10.4f}")
    print(f"  {'category-mean (CDnet macro)':<28}{res['base_cat_mean']:>10.4f}{res['agent_cat_mean']:>10.4f}"
          f"{res['agent_cat_mean']-res['base_cat_mean']:>+10.4f}")
    print(f"  worst-category gain={res['worst_cat_gain']:+.4f}    action mix C/G/B="
          f"{am['cpb']:.2f}/{am['guid']:.2f}/{am['base']:.2f}")
    print(f"  Liang CPB+DeepLabV3+ 'Overall' (POOLED) = 0.753  -> compare to the POOLED row.")
    print(f"  NB: this is the held-out TEST SUBSET; use --stage cv for the all-scenes pooled number.\n{'='*74}")
    print(f"  {'category':<26}{'base':>9}{'agent':>9}{'gain':>9}")
    for c, d in sorted(res['per_cat'].items()):
        print(f"  {c:<26}{d['base']:>9.4f}{d['agent']:>9.4f}{d['agent']-d['base']:>+9.4f}")
    out = {k: v for k, v in res.items() if k != 'scene_f1'}
    out['scene_f1'] = res['scene_f1']
    json.dump(out, open(os.path.join(args.save_dir, 'test_results.json'), 'w'), indent=2, default=float)
    print(f"\n  saved -> {os.path.join(args.save_dir, 'test_results.json')}")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['precompute', 'train', 'test', 'cv', 'split'])
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--guidance', default='deeplab')
    ap.add_argument('--guidance_models_dir', default='./models')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache_dir', default='drl_cache_junior')
    ap.add_argument('--n_frames', type=int, default=60)
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--batch_frames', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--entropy_final', type=float, default=0.003)
    ap.add_argument('--bc_coef', type=float, default=0.0,
                    help='behavior-cloning aux weight at step 0 (0.0 = OFF = exact prior agent; try 1.0)')
    ap.add_argument('--bc_final', type=float, default=0.2,
                    help='behavior-cloning aux weight at the final step (linear anneal from --bc_coef)')
    # ---- patch-CNN ----
    ap.add_argument('--no_patch', action='store_true', help='disable the patch-CNN branch (pure-scalar MLP)')
    ap.add_argument('--patch_k', type=int, default=4, help='patch radius -> (2k+1)x(2k+1); 4 = 9x9')
    # ---- reward / exploration ----
    ap.add_argument('--reward_mode', choices=['f1marg', 'classbal'], default='f1marg')
    ap.add_argument('--baseline', choices=['frame', 'ema'], default='frame',
                    help='advantage baseline: frame = per-frame centring (fix); ema = old global (ablation)')
    ap.add_argument('--w_fix', type=float, default=0.5)
    ap.add_argument('--w_break', type=float, default=0.5)
    ap.add_argument('--w_conf', type=float, default=0.05, help='confidence shaping (lowered from 0.1)')
    ap.add_argument('--fg_weight', type=float, default=10.0, help='(classbal only)')
    ap.add_argument('--w_correct', type=float, default=1.0, help='(classbal only)')
    ap.add_argument('--w_wrong', type=float, default=0.7, help='(classbal only)')
    # ---- abstain gate ----
    ap.add_argument('--abstain_init', type=float, default=1.5)
    ap.add_argument('--harm_penalty', type=float, default=1.0)
    # ---- CV ----
    ap.add_argument('--cv_folds', type=int, default=5)
    # ---- framework params ----
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    # ---- io / debug ----
    ap.add_argument('--save_dir', default='models/drl_gate_v4')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--debug_every', type=int, default=500)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    args.use_patch = not args.no_patch

    if args.stage == 'split':
        print(json.dumps(make_split(), indent=2))
    elif args.stage == 'precompute':
        precompute(args)
    elif args.stage == 'train':
        train(args)
    elif args.stage == 'test':
        test(args)
    elif args.stage == 'cv':
        cross_validate(args)


if __name__ == '__main__':
    main()