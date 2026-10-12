#!/usr/bin/env python3
"""
Run ONE cross-validation fold in its own process (memory-safe on a laptop).
Reuses opus_final_project_metrics's own functions, so results are identical to
the monolithic --stage cv, just without cross-fold memory accumulation.

If a fold's best.pth already exists it is LOADED (fold is only re-evaluated, not
retrained) — so a fold that already finished is not repeated. Use --retrain to force.

Driver (PowerShell), agent then MLP:
  for ($k=0; $k -le 4; $k++) { python cv_fold.py --fold $k --models_dir models/cpb_numba_rgb --cache_dir drl_cache --steps 20000 --win_size 10 --bc_coef 1.0 --save_dir models/drl_gate_rgb_cv }
  for ($k=0; $k -le 4; $k++) { python cv_fold.py --fold $k --models_dir models/cpb_numba_rgb --cache_dir drl_cache --steps 20000 --win_size 10 --bc_coef 1.0 --no_patch --save_dir models/drl_gate_rgb_cv_mlp }
"""
import argparse, json, os, gc
import opus_final_project_metrics as T


def build_args(cli):
    class A:
        pass
    a = A()
    a.fold = cli.fold; a.cv_folds = cli.cv_folds
    a.models_dir = cli.models_dir; a.cache_dir = cli.cache_dir
    a.win_size = cli.win_size; a.steps = cli.steps
    a.bc_coef = cli.bc_coef; a.bc_final = cli.bc_final
    a.use_patch = not cli.no_patch; a.patch_k = cli.patch_k
    a.save_dir = cli.save_dir
    a.af = cli.af; a.floor_min = cli.floor_min; a.bth_pixels = cli.bth_pixels
    a.ref = cli.ref; a.sigema = cli.sigema
    a.eval_every = cli.eval_every
    # trainer defaults (mirror opus_final_project_metrics.main)
    a.batch_frames = 8; a.lr = 3e-4
    a.entropy = 0.01; a.entropy_final = 0.003
    a.reward_mode = 'f1marg'; a.baseline = 'frame'
    a.w_fix = 0.5; a.w_break = 0.5; a.w_conf = 0.05
    a.fg_weight = 10.0; a.w_correct = 1.0; a.w_wrong = 0.7
    a.abstain_init = 1.5; a.harm_penalty = 1.0
    a.debug = False; a.debug_every = 500; a.force = False; a.n_frames = 60
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--models_dir', default='models/cpb_numba_rgb')
    ap.add_argument('--cache_dir', default='drl_cache')
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--bc_coef', type=float, default=1.0)
    ap.add_argument('--bc_final', type=float, default=0.2)
    ap.add_argument('--no_patch', action='store_true')
    ap.add_argument('--patch_k', type=int, default=4)
    ap.add_argument('--save_dir', default='models/drl_gate_rgb_cv')
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--retrain', action='store_true', help='force retrain even if best.pth exists')
    cli = ap.parse_args()
    a = build_args(cli)

    import torch
    root = T.find_root()
    envargs = dict(af=a.af, floor=a.floor_min, bth=a.bth_pixels, ref=a.ref, sigema=a.sigema)
    folds = T.make_kfold(a.cv_folds, T.SEED)
    kf = a.fold
    test_ids = folds[kf]
    rest = [s for j in range(a.cv_folds) if j != kf for s in folds[j]]
    val_ids = T.carve_val(rest, seed=T.SEED + kf)
    train_ids = [s for s in rest if s not in val_ids]
    fold_dir = os.path.join(a.save_dir, f'fold{kf}')
    os.makedirs(fold_dir, exist_ok=True)
    ckpt = os.path.join(fold_dir, 'best.pth')

    print(f'==== FOLD {kf}/{a.cv_folds-1}  train={len(train_ids)} val={len(val_ids)} '
          f'test={len(test_ids)}  patch={a.use_patch} ====')

    if os.path.exists(ckpt) and not cli.retrain:
        ck = torch.load(ckpt, map_location='cpu')
        a.use_patch = ck.get('use_patch', a.use_patch); a.patch_k = ck.get('patch_k', a.patch_k)
        gate = T.build_gate(ck['state_dim'], ck.get('n_actions', T.N_ACTIONS), 0.0,
                            a.use_patch, a.patch_k)
        gate.load_state_dict(ck['gate'])
        norm = T.RunningNorm(T.ENV.FrameEnv.STATE_DIM)
        norm.mean = ck['norm_mean']; norm.M2 = ck['norm_M2']; norm.n = ck['norm_n']
        norm.frozen = True
        print(f'  loaded existing checkpoint (no retrain): {ckpt}')
    else:
        train_sc = T.build_scenes(train_ids, a, root, envargs)
        val_sc = T.build_scenes(list(val_ids), a, root, envargs)
        gate, norm, _ = T.train_one(a, train_sc, val_sc, fold_dir, seed=T.SEED + kf, tag=f'-f{kf}')
        del train_sc, val_sc; gc.collect()

    test_sc = T.build_scenes(test_ids, a, root, envargs)
    per_scene = {}
    for sc in test_sc:
        rr = T.evaluate(gate, norm, [sc], a)
        per_scene[sc.sid] = {'cat': sc.sid.split('/')[0], 'base': rr['base'], 'agent': rr['f1'],
                             'A': [int(x) for x in rr['A']], 'B': [int(x) for x in rr['B']],
                             'act_mix': rr['act_mix']}
        print(f'  {sc.sid:34s} base={rr["base"]:.4f} agent={rr["f1"]:.4f} gain={rr["f1"]-rr["base"]:+.4f}')

    out = os.path.join(a.save_dir, f'fold{kf}_results.json')
    json.dump({'fold': kf, 'use_patch': a.use_patch, 'per_scene': per_scene}, open(out, 'w'), indent=2)
    print(f'  saved -> {out}')


if __name__ == '__main__':
    main()
