#!/usr/bin/env python3
"""
================================================================================
TRAIN TRANSFER ARBITERS  (DRL agent + supervised MLP on the SAME CDnet scenes)
================================================================================
Produces the two checkpoints the transfer eval needs for a FAIR MLP-vs-DRL
comparison on LIMU / WallFlower:

    <drl_save_dir>/best.pth      the DRL agent   (3-way: CPB / guidance / keep)
    <sup_save_dir>/sup.pth       the supervised MLP (3-way: CPB / guidance / keep)

Both are trained on the IDENTICAL set of CDnet scenes, so any difference on the
target dataset is the method, not the data. Reuses your trainer's own functions
(train_one, _gather_disputes, _fit_supervised, build_scenes) so the math is
exactly what --stage cv/ablation already used.

TWO PROTOCOLS
  --scenes all        train on ALL 53 CDnet scenes (minus a 1-per-category val).
                      The paper's headline "trained on CDnet, tested on LIMU".
  --scenes fold --fold K
                      reproduce fold K's EXACT training scenes (same as the CV /
                      ablation split). Use this with --skip_drl to fit ONLY the
                      MLP that matches your already-trained fold{K}/best.pth --
                      a fair MLP-vs-DRL check in minutes, no DRL retrain.

TYPICAL USE
  # (fast) fair MLP for your existing fold0 agent, no DRL retrain:
  python opus_train_transfer_arbiters.py --scenes fold --fold 0 --skip_drl \
         --cache_dir drl_cache_junior --models_dir models/cpb_numba \
         --sup_save_dir models/sup_gate_fold0 --sup_balance

  # (full) both arbiters on ALL of CDnet -- the paper version (DRL part is slow):
  python opus_train_transfer_arbiters.py --scenes all \
         --cache_dir drl_cache_junior --models_dir models/cpb_numba \
         --drl_save_dir models/drl_gate_all --sup_save_dir models/sup_gate_all \
         --sup_balance --steps 40000 --no_patch
================================================================================
"""
import argparse, os, sys, importlib
from types import SimpleNamespace

# --- import the trainer module (same fallback order as the transfer evaluator) ---
T = None
for _n in ('opus_final_project_metrics', 'opus_final_project_metrics_ablation',
           'opus_final_drl_train_junior_enhanced_with_patch_cnn',
           'opus_final_drl_train_junior'):
    try:
        T = importlib.import_module(_n); break
    except Exception:
        continue
if T is None:
    sys.exit('Could not import your trainer module (opus_final_project_metrics.py). '
             'Run this from the project folder.')
F = T.F
ENV = T.ENV
SEED = getattr(T, 'SEED', 1234)


# ---------------------------------------------------------------------------
# Supervised-arbiter helpers -- defined HERE so the script does not depend on
# whether the installed trainer is the ablation build. Logic is identical to
# the ablation's _gather_disputes / _fit_supervised.
# ---------------------------------------------------------------------------
def _gather_disputes(scenes, cap):
    """Collect (state, 3-way oracle action) over all valid disputed pixels.
    action 0 = trust CPB, 1 = trust guidance, 2 = keep baseline (Stage-3 label).
    Identical label scheme to opus_backbone_ablation._gather_disputes so the
    transfer MLP matches the ablation MLP and the DRL agent's action space."""
    import numpy as np
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
            gt = e.gtfg_fit[r, c].astype(bool)
            cpb = (e.s1_default[r, c] > 0)
            base = e.base_pred_fit[r, c].astype(bool)
            base_ok = (base == gt); cpb_ok = (cpb == gt)
            lab = np.where(base_ok, 2, np.where(cpb_ok, 0, 1)).astype(np.int64)
            ys.append(lab)
            Xs.append(e.state(r, c).astype(np.float32))
    if not Xs:
        return None, None
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def _fit_supervised(X, y, args, seed):
    """Standardised logistic (linear) or capacity-matched MLP, class-balanced CE."""
    import numpy as np, torch, torch.nn as nn
    torch.manual_seed(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    dim = X.shape[1]
    n_act = 3  # 3-way gate: {0:CPB, 1:guidance, 2:keep baseline}
    if args.sup_model == 'linear':
        net = nn.Linear(dim, n_act)
    else:  # capacity-matched to the DRL scalar policy (64-64)
        net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_act))
    cw = None
    if args.sup_balance:
        cnt = np.bincount(y, minlength=n_act).astype(np.float64)
        cw = torch.tensor(cnt.sum() / (n_act * np.maximum(cnt, 1.0)), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=args.sup_lr)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y.astype(np.int64))
    N = len(yt); bs = 8192
    net.train()
    for ep in range(args.sup_epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            tot += float(loss.item())
        if args.debug:
            print(f'      [sup ep {ep+1}/{args.sup_epochs}] mean CE={tot/max(1,(N+bs-1)//bs):.4f}')
    net.eval()
    return net, mu.astype(np.float32), sd.astype(np.float32)


def all_cdnet_ids():
    return [f'{cat}/{s}' for cat, scenes in F.CDNET2014.items() for s in scenes]


def pick_scene_ids(args):
    """Return (train_ids, val_ids, tag) for the chosen protocol."""
    if args.scenes == 'all':
        allids = all_cdnet_ids()
        val = T.carve_val(allids, seed=SEED)                   # 1 per category
        train = [s for s in allids if s not in val]
        return train, list(val), 'all-cdnet'
    # fold: reproduce fold K's EXACT training split (matches CV / ablation)
    folds = T.make_kfold(args.cv_folds, SEED)
    test = folds[args.fold]
    rest = [s for j in range(args.cv_folds) if j != args.fold for s in folds[j]]
    val = T.carve_val(rest, seed=SEED + args.fold)
    train = [s for s in rest if s not in val]
    print(f'  [fold {args.fold}] held-out test scenes (NOT trained on): {test}')
    return train, list(val), f'fold{args.fold}'


def save_supervised(net, mu, sd, args, train_ids, tag):
    import torch
    os.makedirs(args.sup_save_dir, exist_ok=True)
    path = os.path.join(args.sup_save_dir, 'sup.pth')
    torch.save({'sup_state': net.state_dict(),
                'sup_model': args.sup_model,          # 'linear' | 'mlp'
                'mu': mu, 'sd': sd,
                'state_dim': int(ENV.FrameEnv.STATE_DIM),
                'n_actions': 3,                       # {0:CPB, 1:guidance, 2:keep baseline}
                'trained_on': tag,
                'n_train_scenes': len(train_ids),
                'sup_balance': bool(args.sup_balance)}, path)
    print(f'  supervised MLP saved -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', choices=['all', 'fold'], default='all')
    ap.add_argument('--fold', type=int, default=0, help='(--scenes fold) which fold to mirror')
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--skip_drl', action='store_true', help='do not (re)train the DRL agent')
    ap.add_argument('--skip_sup', action='store_true', help='do not train the supervised MLP')
    ap.add_argument('--drl_save_dir', default='models/drl_gate_all')
    ap.add_argument('--sup_save_dir', default='models/sup_gate_all')
    # ---- data / cache ----
    ap.add_argument('--cache_dir', default='drl_cache_junior')
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--win_size', type=int, default=10)
    # ---- DRL training (mirror the trainer defaults so train_one is happy) ----
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--batch_frames', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--entropy_final', type=float, default=0.003)
    ap.add_argument('--bc_coef', type=float, default=0.0)
    ap.add_argument('--bc_final', type=float, default=0.2)
    ap.add_argument('--no_patch', action='store_true')
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
    # ---- supervised MLP ----
    ap.add_argument('--sup_model', choices=['linear', 'mlp'], default='mlp')
    ap.add_argument('--sup_epochs', type=int, default=15)
    ap.add_argument('--sup_lr', type=float, default=1e-3)
    ap.add_argument('--sup_balance', action='store_true')
    ap.add_argument('--sup_cap', type=int, default=3000)
    # ---- framework params (must match how the cache/models were built) ----
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--debug_every', type=int, default=500)
    args = ap.parse_args()
    args.use_patch = not args.no_patch

    root = T.find_root()
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels,
                   ref=args.ref, sigema=args.sigema)
    train_ids, val_ids, tag = pick_scene_ids(args)
    print('=' * 78)
    print(f'  TRANSFER ARBITERS  protocol={args.scenes}({tag})  '
          f'train_scenes={len(train_ids)}  val_scenes={len(val_ids)}')
    print(f'  cache={args.cache_dir}  cpb_models={args.models_dir}  patch={args.use_patch}')
    print('=' * 78)

    train_sc = T.build_scenes(train_ids, args, root, envargs)
    if not train_sc:
        sys.exit('No training scenes cached — check --cache_dir / --models_dir '
                 '(these must be your CDnet cache + CPB models).')

    # ---- supervised MLP (cheap) ----
    if not args.skip_sup:
        X, y = _gather_disputes(train_sc, args.sup_cap)
        if X is None:
            print('  [warn] no disputes gathered — skipping supervised MLP')
        else:
            import numpy as _np
            _fr = _np.bincount(y, minlength=3) / max(len(y), 1)
            print(f'  supervised: {len(y):,} disputed px  [CPB {_fr[0]:.2f} | guid {_fr[1]:.2f} | keep {_fr[2]:.2f}]  '
                  f'fitting 3-way {args.sup_model} (balance={args.sup_balance}) ...')
            net, mu, sd = _fit_supervised(X, y, args, seed=SEED)
            save_supervised(net, mu, sd, args, train_ids, tag)
            del X, y

    # ---- DRL agent (expensive) ----
    if not args.skip_drl:
        val_sc = T.build_scenes(val_ids, args, root, envargs)
        if not val_sc:
            sys.exit('No val scenes cached — needed for DRL best-checkpoint selection.')
        os.makedirs(args.drl_save_dir, exist_ok=True)
        print(f'\n  training DRL agent on {len(train_sc)} scenes -> {args.drl_save_dir} '
              f'(steps={args.steps}) ...')
        T.train_one(args, train_sc, val_sc, args.drl_save_dir, seed=T.SEED, tag=f'-{tag}')
        print(f'  DRL agent saved -> {os.path.join(args.drl_save_dir, "best.pth")}')

    print('\n' + '=' * 78)
    print('  DONE. Next — transfer eval with BOTH arbiters:')
    drl_ck = os.path.join(args.drl_save_dir, 'best.pth')
    sup_ck = os.path.join(args.sup_save_dir, 'sup.pth')
    print(f'    python opus_transfer_limu_evaluation.py --stage eval --dataset limu \\')
    print(f'      --models_dir models/cpb_limu --cache_dir drl_cache_limu \\')
    print(f'      --agent_ckpt {drl_ck} --sup_ckpt {sup_ck}')
    print('=' * 78)


if __name__ == '__main__':
    main()