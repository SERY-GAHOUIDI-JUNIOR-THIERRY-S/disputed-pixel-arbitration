#!/usr/bin/env python3
"""
================================================================================
LIMU IN-DOMAIN LEAVE-ONE-SCENE-OUT CV  (framework vs +MLP vs +agent vs oracle)
================================================================================
The in-domain LIMU counterpart of your CDnet cross-validation, with the SAME
rigorous cross-scene protocol -> NO leakage:

  for each held-out LIMU scene s in {CameraParameter, Intersection, LightSwitch}:
      train the DRL agent AND a capacity-matched supervised MLP on the OTHER two
      scenes' disputed pixels, then score four systems on scene s (all labeled
      ROI frames, no subsampling):
          framework          = e.baseline_mask()  (CPB Stages 1-2-3 = Liang)
          framework + MLP     = supervised arbiter resolves the disputes
          framework + agent   = DRL arbiter resolves the disputes
          oracle              = disputes resolved by GT (the reachable ceiling)

Reuses YOUR trainer (train_one, build_scenes, evaluate helpers). The supervised
helpers are defined here so it works with any trainer build. Because a held-out
scene is never in training, the arbiter never sees the test scene -> the same
honest cross-scene claim as CDnet, now on a second dataset.

NOTE: only 3 LIMU scenes, so each fold trains on 2 -> results are small-sample and
noisy; report as a "small-scale in-domain confirmation", not a headline. With 2
training scenes there is no separate val set, so DRL best-checkpoint selection
uses the training scenes (in-sample); the TEST scene is still fully held out,
which is what the reported number depends on.

USAGE
  python opus_limu_indomain_cv.py --cache_dir drl_cache_limu --models_dir models/cpb_limu \
         --steps 8000 --sup_balance --sup_cap 20000 --save_dir models/limu_indomain
================================================================================
"""
import argparse, json, os, sys, importlib
import numpy as np

T = None
for _n in ('opus_final_project_metrics', 'opus_final_project_metrics_ablation',
           'opus_final_drl_train_junior_enhanced_with_patch_cnn',
           'opus_final_drl_train_junior'):
    try:
        T = importlib.import_module(_n); break
    except Exception:
        continue
if T is None:
    sys.exit('Could not import your trainer module. Run from the project folder.')
F = T.F
ENV = T.ENV
SEED = getattr(T, 'SEED', 1234)

LIMU_SCENES = ['limu/CameraParameter', 'limu/Intersection', 'limu/LightSwitch']


# --- supervised helpers (local, so we do not depend on the trainer build) ---
def _gather_disputes(scenes, cap):
    rng = np.random.default_rng(SEED)
    Xs, ys = [], []
    for sc in scenes:
        for i in range(len(sc.nums)):
            e = sc.env(i); r, c = e.dispute_ij()
            if len(r) == 0:
                continue
            keep = e.valid_fit[r, c]; r, c = r[keep], c[keep]
            if len(r) == 0:
                continue
            if cap and len(r) > cap:
                sel = rng.choice(len(r), cap, replace=False); r, c = r[sel], c[sel]
            guid = e.guid_fg[r, c] > 0; gt = e.gtfg_fit[r, c]
            ys.append((guid == gt).astype(np.int64))
            Xs.append(e.state(r, c).astype(np.float32))
    if not Xs:
        return None, None
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def _fit_supervised(X, y, args, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    dim = X.shape[1]
    net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 2))
    cw = None
    if args.sup_balance:
        cnt = np.bincount(y, minlength=2).astype(np.float64)
        cw = torch.tensor(cnt.sum() / (2.0 * np.maximum(cnt, 1.0)), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=args.sup_lr)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y.astype(np.int64))
    N = len(yt); bs = 8192
    net.train()
    for ep in range(args.sup_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    return net, mu.astype(np.float32), sd.astype(np.float32)


def f1c(a):
    return ENV.f1_from_counts(*a)


def score_scene(sc, gate, norm, sup, use_patch, patch_k):
    """Return per-system count vectors on one scene, all labeled frames."""
    import torch
    systems = ['framework', 'framework+mlp', 'framework+agent', 'oracle']
    acc = {s: [0, 0, 0, 0] for s in systems}
    snet, smu, ssd = (sup if sup else (None, None, None))
    for i in range(len(sc.nums)):
        e = sc.env(i); r, c = e.dispute_ij()
        base = e.baseline_mask()
        # framework
        for k, v in enumerate(e.counts(base)):
            acc['framework'][k] += v
        # framework + agent
        if len(r) == 0:
            am = base
        else:
            Sn = norm(e.state(r, c))
            Pt = torch.from_numpy(e.patches(r, c, patch_k)) if use_patch else None
            with torch.no_grad():
                act = gate(torch.from_numpy(Sn), Pt).argmax(-1).numpy()
            am = T._segment_gate3(e, act, r, c)
        for k, v in enumerate(e.counts(am)):
            acc['framework+agent'][k] += v
        # framework + mlp
        if snet is None or len(r) == 0:
            mm = base
        else:
            Xn = ((e.state(r, c).astype(np.float32) - smu) / ssd).astype(np.float32)
            with torch.no_grad():
                sact = snet(torch.from_numpy(Xn)).argmax(-1).numpy()
            mm = T._segment_gate3(e, sact, r, c)
        for k, v in enumerate(e.counts(mm)):
            acc['framework+mlp'][k] += v
        # oracle
        of = e.s1_default.copy()
        if len(r):
            of[r, c] = (e.gtfg_fit[r, c].astype(np.uint8) * 255)
        om = e._to_native(of)
        for k, v in enumerate(e.counts(om)):
            acc['oracle'][k] += v
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', default='drl_cache_limu')
    ap.add_argument('--models_dir', default='models/cpb_limu')
    ap.add_argument('--save_dir', default='models/limu_indomain')
    ap.add_argument('--win_size', type=int, default=10)
    # DRL
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--batch_frames', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--entropy_final', type=float, default=0.003)
    ap.add_argument('--bc_coef', type=float, default=0.0)
    ap.add_argument('--bc_final', type=float, default=0.2)
    ap.add_argument('--no_patch', action='store_true', default=True)
    ap.add_argument('--with_patch', dest='no_patch', action='store_false')
    ap.add_argument('--patch_k', type=int, default=4)
    ap.add_argument('--reward_mode', choices=['f1marg', 'classbal'], default='f1marg')
    ap.add_argument('--baseline', choices=['frame', 'ema'], default='frame')
    ap.add_argument('--w_fix', type=float, default=0.5)
    ap.add_argument('--w_break', type=float, default=0.5)
    ap.add_argument('--w_conf', type=float, default=0.05)
    ap.add_argument('--fg_weight', type=float, default=10.0)
    ap.add_argument('--w_correct', type=float, default=1.0)
    ap.add_argument('--w_wrong', type=float, default=0.7)
    ap.add_argument('--abstain_init', type=float, default=1.5)
    ap.add_argument('--harm_penalty', type=float, default=1.0)
    # supervised
    ap.add_argument('--sup_epochs', type=int, default=15)
    ap.add_argument('--sup_lr', type=float, default=1e-3)
    ap.add_argument('--sup_balance', action='store_true')
    ap.add_argument('--sup_cap', type=int, default=20000)
    ap.add_argument('--skip_drl', action='store_true')
    # framework params (must match how the LIMU cache/models were built)
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--debug_every', type=int, default=500)
    args = ap.parse_args()
    args.use_patch = not args.no_patch
    os.makedirs(args.save_dir, exist_ok=True)
    root = T.find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)

    systems = ['framework', 'framework+mlp', 'framework+agent', 'oracle']
    pool = {s: [0, 0, 0, 0] for s in systems}
    per_scene = {}

    for held in LIMU_SCENES:
        train_ids = [s for s in LIMU_SCENES if s != held]
        print('\n' + '=' * 74)
        print(f'  FOLD  held-out = {held}   train = {train_ids}')
        print('=' * 74)
        train_sc = T.build_scenes(train_ids, args, root, envargs)
        test_sc = T.build_scenes([held], args, root, envargs)
        if not train_sc or not test_sc:
            print('  [skip] missing cached scenes for this fold'); continue

        # supervised MLP
        sup = None
        X, y = _gather_disputes(train_sc, args.sup_cap)
        if X is not None:
            print(f'  supervised: {len(y):,} disputed px  trust-guid={float(y.mean()):.3f}  fitting MLP ...')
            net, mu, sd = _fit_supervised(X, y, args, seed=SEED)
            sup = (net, mu, sd); del X, y

        # DRL agent (val = train scenes, in-sample selection; test scene held out)
        gate = norm = None
        if not args.skip_drl:
            print(f'  training DRL agent on {len(train_sc)} scenes (steps={args.steps}) ...')
            gate, norm, _ = T.train_one(args, train_sc, train_sc,
                                        os.path.join(args.save_dir, held.replace('/', '_')),
                                        seed=SEED, tag=f'-{held.split("/")[1][:4]}')

        # score the held-out scene (all labeled frames)
        acc = score_scene(test_sc[0], gate, norm, sup, args.use_patch, args.patch_k)
        row = {s: f1c(acc[s]) for s in systems}
        per_scene[held] = row
        for s in systems:
            for k in range(4):
                pool[s][k] += acc[s][k]
        print('  scene F1: ' + '  '.join(f'{s}={row[s]:.4f}' for s in systems))

    # ---- aggregate ----
    print('\n' + '=' * 74)
    print('  LIMU IN-DOMAIN (leave-one-scene-out)  —  all labeled ROI frames')
    print('=' * 74)
    print(f"  {'system':<18}{'pooled':>10}{'scene-mean':>12}")
    pooled = {s: f1c(pool[s]) for s in systems}
    scene_mean = {s: float(np.mean([per_scene[h][s] for h in per_scene])) if per_scene else 0.0
                  for s in systems}
    for s in systems:
        print(f"  {s:<18}{pooled[s]:>10.4f}{scene_mean[s]:>12.4f}")
    print('  ' + '-' * 46)
    fw = pooled['framework']
    print(f"  framework+agent - framework : pooled {pooled['framework+agent']-fw:+.4f}"
          f"   scene {scene_mean['framework+agent']-scene_mean['framework']:+.4f}")
    print(f"  framework+mlp   - framework : pooled {pooled['framework+mlp']-fw:+.4f}"
          f"   scene {scene_mean['framework+mlp']-scene_mean['framework']:+.4f}")
    print(f"  agent - mlp                 : pooled {pooled['framework+agent']-pooled['framework+mlp']:+.4f}")
    print('  (compare against OUR framework reproduction, not Liang''s printed 0.793;')
    print('   the gap to 0.793 is reproduction quality, separate from the arbiter.)')
    print('\n  PER-SCENE:')
    print(f"  {'scene':<26}" + ''.join(f'{s[:12]:>14}' for s in systems))
    for h in per_scene:
        print(f"  {h:<26}" + ''.join(f'{per_scene[h][s]:>14.4f}' for s in systems))
    print('=' * 74)

    out = {'protocol': 'leave-one-scene-out', 'dataset': 'limu',
           'pooled': pooled, 'scene_mean': scene_mean, 'per_scene': per_scene}
    dst = os.path.join(args.save_dir, 'limu_indomain_cv.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'  saved -> {dst}')


if __name__ == '__main__':
    main()