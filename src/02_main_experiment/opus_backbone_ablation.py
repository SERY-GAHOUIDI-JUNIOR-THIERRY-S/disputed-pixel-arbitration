#!/usr/bin/env python3
"""
================================================================================
BACKBONE ABLATION  (framework vs framework+MLP under a given guidance cache)
================================================================================
Self-contained: does NOT need the trainer's `ablation` stage. For each CV fold
it trains a class-balanced supervised MLP arbiter on that fold's training scenes
(disputed pixels) and scores four systems on the held-out scenes, on identical
pixels, over all 5 folds (every scene tested once):
    guidance | framework (Liang fixed rule) | framework+MLP | oracle

Point it at the PSPNet guidance cache to fill the PSPNet row of Table 6; point it
at the DeepLab cache to reproduce that row as a sanity check.

USAGE
  python opus_backbone_ablation.py --cache_dir drl_cache_pspnet \
     --models_dir models/cpb_numba --save_dir models/ablation_pspnet --sup_balance
  # sanity check on DeepLab (should match the paper's 0.678 / 0.764):
  python opus_backbone_ablation.py --cache_dir drl_cache_junior \
     --models_dir models/cpb_numba --save_dir models/ablation_deeplab --sup_balance
================================================================================
"""
import argparse, json, os, sys, importlib
import numpy as np

T = None
for _n in ('opus_final_project_metrics', 'opus_final_project_metrics_ablation',
           'opus_final_drl_train_junior_enhanced_with_patch_cnn', 'opus_final_drl_train_junior'):
    try:
        T = importlib.import_module(_n); break
    except Exception:
        continue
if T is None:
    sys.exit('Could not import trainer module. Run from the project folder.')
F = T.F; ENV = T.ENV
SEED = getattr(T, 'SEED', 1234)
SYS = ['guidance', 'framework', 'framework+mlp', 'oracle']


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
            # --- 3-way oracle label (matches the DRL agent's action space) ---
            #   action 0 = trust CPB, 1 = trust guidance, 2 = keep baseline (Stage-3)
            # At a disputed pixel CPB and guidance disagree, so exactly one of them
            # equals GT -> the final `else 1` is always the correct guidance case.
            # Prefer action 2 whenever the frozen framework is ALREADY right: that is
            # what stops the arbiter from bulldozing clean scenes (e.g. LightSwitch).
            gt = e.gtfg_fit[r, c].astype(bool)
            cpb = (e.s1_default[r, c] > 0)
            base = e.base_pred_fit[r, c].astype(bool)
            base_ok = (base == gt); cpb_ok = (cpb == gt)
            lab = np.where(base_ok, 2, np.where(cpb_ok, 0, 1)).astype(np.int64)
            ys.append(lab); Xs.append(e.state(r, c).astype(np.float32))
    if not Xs:
        return None, None
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def _fit_supervised(X, y, args, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32); dim = X.shape[1]
    n_act = 3  # 3-way gate: {0:CPB, 1:guidance, 2:keep baseline}
    net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_act))
    cw = None
    if args.sup_balance:
        cnt = np.bincount(y, minlength=n_act).astype(np.float64)
        cw = torch.tensor(cnt.sum() / (n_act * np.maximum(cnt, 1.0)), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=args.sup_lr)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y.astype(np.int64))
    N = len(yt); bs = 8192; net.train()
    for ep in range(args.sup_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    return net, mu.astype(np.float32), sd.astype(np.float32)


def score_scene(sc, sup):
    import torch
    net, mu, sd = sup
    acc = {s: [0, 0, 0, 0] for s in SYS}
    for i in range(len(sc.nums)):
        e = sc.env(i); r, c = e.dispute_ij()
        base = e.baseline_mask()
        guid = e._to_native((e.guid_fg > 0).astype(np.uint8) * 255)
        if len(r) == 0:
            mm = base
        else:
            Xn = ((e.state(r, c).astype(np.float32) - mu) / sd).astype(np.float32)
            with torch.no_grad():
                sact = net(torch.from_numpy(Xn)).argmax(-1).numpy()
            mm = T._segment_gate3(e, sact, r, c)
        of = e.s1_default.copy()
        if len(r):
            of[r, c] = (e.gtfg_fit[r, c].astype(np.uint8) * 255)
        om = e._to_native(of)
        for name, mask in (('guidance', guid), ('framework', base), ('framework+mlp', mm), ('oracle', om)):
            for k, v in enumerate(e.counts(mask)):
                acc[name][k] += v
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--save_dir', default='models/ablation_backbone')
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--sup_epochs', type=int, default=15)
    ap.add_argument('--sup_lr', type=float, default=1e-3)
    ap.add_argument('--sup_balance', action='store_true')
    ap.add_argument('--sup_cap', type=int, default=3000)
    ap.add_argument('--af', type=float, default=2.5); ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0); ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    root = T.find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    folds = T.make_kfold(args.cv_folds, SEED)

    pool = {s: [0, 0, 0, 0] for s in SYS}
    scene_f1 = {}; cat_counts = {}
    for kf in range(args.cv_folds):
        test_ids = folds[kf]
        rest = [s for j in range(args.cv_folds) if j != kf for s in folds[j]]
        val = T.carve_val(rest, seed=SEED + kf)
        train_ids = [s for s in rest if s not in val]
        print(f'\n==== FOLD {kf}  train={len(train_ids)} test={len(test_ids)} ====')
        train_sc = T.build_scenes(train_ids, args, root, envargs)
        if not train_sc:
            print('  [skip] no train scenes cached'); continue
        X, y = _gather_disputes(train_sc, args.sup_cap)
        if X is None:
            print('  [skip] no disputes'); continue
        _fr = np.bincount(y, minlength=3) / max(len(y), 1)
        print(f'  supervised: {len(y):,} disputed px  [CPB {_fr[0]:.2f} | guid {_fr[1]:.2f} | keep {_fr[2]:.2f}]  fitting 3-way MLP ...')
        net, mu, sd = _fit_supervised(X, y, args, seed=SEED + kf); del X, y, train_sc
        test_sc = T.build_scenes(test_ids, args, root, envargs)
        for sc in test_sc:
            cat = sc.sid.split('/')[0]
            acc = score_scene(sc, (net, mu, sd))
            scene_f1[sc.sid] = {s: ENV.f1_from_counts(*acc[s]) for s in SYS}
            cat_counts.setdefault(cat, {s: [0, 0, 0, 0] for s in SYS})
            for s in SYS:
                for k in range(4):
                    pool[s][k] += acc[s][k]; cat_counts[cat][s][k] += acc[s][k]
        del test_sc

    pooled = {s: ENV.f1_from_counts(*pool[s]) for s in SYS}
    scene_mean = {s: float(np.mean([scene_f1[x][s] for x in scene_f1])) if scene_f1 else 0.0 for s in SYS}
    percat = {c: {s: ENV.f1_from_counts(*cat_counts[c][s]) for s in SYS} for c in cat_counts}
    cat_mean = {s: float(np.mean([percat[c][s] for c in percat])) if percat else 0.0 for s in SYS}

    print('\n' + '=' * 74)
    print(f'  BACKBONE ABLATION  cache={args.cache_dir}  (every scene tested once)')
    print('=' * 74)
    print(f"  {'system':<16}{'pooled':>10}{'scene-mean':>13}{'category-mean':>16}")
    for s in SYS:
        print(f"  {s:<16}{pooled[s]:>10.4f}{scene_mean[s]:>13.4f}{cat_mean[s]:>16.4f}")
    print('  ' + '-' * 60)
    fw = pooled['framework']
    print(f"  framework+mlp - framework : pooled {pooled['framework+mlp']-fw:+.4f}   "
          f"cat {cat_mean['framework+mlp']-cat_mean['framework']:+.4f}")
    print(f"  framework - guidance      : pooled {fw-pooled['guidance']:+.4f}")
    print('=' * 74)
    out = {'cache': args.cache_dir, 'pooled': pooled, 'scene_mean': scene_mean,
           'category_mean': cat_mean, 'per_category': percat, 'scene_f1': scene_f1}
    dst = os.path.join(args.save_dir, 'backbone_ablation.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print('  saved -> ' + dst)


if __name__ == '__main__':
    main()