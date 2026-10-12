#!/usr/bin/env python3
"""
================================================================================
CROSS-FOLD TRANSFER ROBUSTNESS  (agent vs supervised MLP on LIMU, all K folds)
================================================================================
Turns the single-fold LIMU transfer result into a statistical claim. For each CV
fold k it:
   1. trains a supervised MLP on fold-k's EXACT training scenes (if not present),
      via opus_train_transfer_arbiters.py  (--scenes fold --fold k --skip_drl);
   2. runs the LIMU transfer eval with fold-k's agent + fold-k's MLP,
      via opus_transfer_limu_evaluation.py  (--stage eval), saving per-fold JSON;
   3. collects pooled / scene-mean / category-mean for base/guidance/supervised/
      agent, plus per-scene F1.
Then it reports mean +/- std across folds for every system and, crucially, for
the deltas  agent-supervised , agent-guidance , supervised-guidance.

It shells out to the two CLIs you already use (no re-implementation), so the math
is identical to what you ran by hand.

PRE-REQS (all already on disk from your CDnet CV + LIMU setup):
   <drl_dir>/fold{k}/best.pth        the trained CV agents  (default models/drl_gate_cv_metrics)
   <cache_dir>                        CDnet guidance cache   (drl_cache_junior)
   <models_dir>                       CDnet CPB models       (models/cpb_numba)
   <limu_models_dir>                  LIMU CPB models        (models/cpb_limu)
   <limu_cache>                       LIMU guidance cache    (drl_cache_limu)

USAGE
   python opus_transfer_folds.py --sup_balance
   python opus_transfer_folds.py --cv_folds 5 --drl_dir models/drl_gate_cv_metrics \
          --cache_dir drl_cache_junior --models_dir models/cpb_numba \
          --limu_models_dir models/cpb_limu --limu_cache drl_cache_limu
================================================================================
"""
import argparse, json, os, subprocess, sys
import numpy as np

ARBITER = 'opus_train_transfer_arbiters.py'
EVAL = 'opus_transfer_limu_evaluation.py'
SYSTEMS = ['base', 'guidance', 'supervised', 'agent']


def sh(cmd):
    print('  $ ' + ' '.join(cmd))
    r = subprocess.run(cmd)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_folds', type=int, default=5)
    ap.add_argument('--drl_dir', default='models/drl_gate_cv_metrics')
    ap.add_argument('--cache_dir', default='drl_cache_junior')
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--limu_models_dir', default='models/cpb_limu')
    ap.add_argument('--limu_cache', default='drl_cache_limu')
    ap.add_argument('--sup_root', default='models/sup_gate_folds')
    ap.add_argument('--save_root', default='models/transfer_folds')
    ap.add_argument('--dataset', default='limu')
    ap.add_argument('--sup_model', choices=['linear', 'mlp'], default='mlp')
    ap.add_argument('--sup_balance', action='store_true')
    ap.add_argument('--retrain_sup', action='store_true', help='refit the MLP even if sup.pth exists')
    args = ap.parse_args()
    py = sys.executable

    rows = []            # per-fold dict of system -> pooled
    scene_rows = {}      # sid -> system -> [per-fold f1]
    for k in range(args.cv_folds):
        agent_ck = os.path.join(args.drl_dir, f'fold{k}', 'best.pth')
        if not os.path.exists(agent_ck):
            print(f'[fold {k}] SKIP: agent checkpoint missing ({agent_ck})'); continue
        sup_dir = os.path.join(args.sup_root, f'fold{k}')
        sup_ck = os.path.join(sup_dir, 'sup.pth')
        print(f'\n================  FOLD {k}  ================')
        if args.retrain_sup or not os.path.exists(sup_ck):
            cmd = [py, ARBITER, '--scenes', 'fold', '--fold', str(k), '--skip_drl',
                   '--cache_dir', args.cache_dir, '--models_dir', args.models_dir,
                   '--sup_save_dir', sup_dir, '--sup_model', args.sup_model]
            if args.sup_balance:
                cmd.append('--sup_balance')
            if sh(cmd) != 0 or not os.path.exists(sup_ck):
                print(f'[fold {k}] SKIP: supervised training failed'); continue
        else:
            print(f'  [fold {k}] reuse existing {sup_ck}')

        save_dir = os.path.join(args.save_root, f'fold{k}')
        cmd = [py, EVAL, '--stage', 'eval', '--dataset', args.dataset,
               '--models_dir', args.limu_models_dir, '--cache_dir', args.limu_cache,
               '--agent_ckpt', agent_ck, '--sup_ckpt', sup_ck, '--save_dir', save_dir]
        if sh(cmd) != 0:
            print(f'[fold {k}] SKIP: eval failed'); continue
        jp = os.path.join(save_dir, f'transfer_{args.dataset}.json')
        if not os.path.exists(jp):
            print(f'[fold {k}] SKIP: no json at {jp}'); continue
        d = json.load(open(jp))
        rows.append({s: float(d['systems'][s]['pooled']) for s in SYSTEMS})
        for sid, sd in d.get('scene_f1', {}).items():
            for s in SYSTEMS:
                scene_rows.setdefault(sid, {}).setdefault(s, []).append(float(sd[s]))

    if not rows:
        raise SystemExit('No folds completed — check --drl_dir and that CDnet/LIMU caches exist.')

    def ms(vals):
        a = np.array(vals, float)
        return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

    n = len(rows)
    print('\n' + '=' * 78)
    print(f'  CROSS-FOLD TRANSFER on {args.dataset.upper()}   (K={n} folds, pooled F1)')
    print('=' * 78)
    print(f"  {'system':<14}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
    for s in SYSTEMS:
        v = [r[s] for r in rows]; m, sd = ms(v)
        print(f"  {s:<14}{m:>10.4f}{sd:>10.4f}{min(v):>10.4f}{max(v):>10.4f}")

    print('\n  DELTAS (per-fold, then aggregated):')
    def delta(a, b):
        return [r[a] - r[b] for r in rows]
    for name, aa, bb in [('agent - supervised', 'agent', 'supervised'),
                         ('agent - guidance', 'agent', 'guidance'),
                         ('supervised - guidance', 'supervised', 'guidance'),
                         ('agent - base', 'agent', 'base')]:
        v = delta(aa, bb); m, sd = ms(v)
        pos = sum(1 for x in v if x > 0)
        star = '  ***' if (m - sd > 0 and name == 'agent - supervised') else ''
        print(f"    {name:<24} {m:+.4f} +/- {sd:.4f}   (>0 in {pos}/{n} folds){star}")

    print('\n  PER-SCENE (mean +/- std across folds):')
    print(f"  {'scene':<28}{'base':>16}{'guidance':>16}{'supervised':>16}{'agent':>16}")
    for sid in scene_rows:
        cells = ''
        for s in SYSTEMS:
            m, sd = ms(scene_rows[sid][s]); cells += f"{m:>8.3f}+/-{sd:<5.3f}"
        print(f"  {sid:<28}{cells}")

    out = {'dataset': args.dataset, 'folds': n,
           'pooled': {s: list(map(lambda r: r[s], rows)) for s in SYSTEMS},
           'delta_agent_minus_supervised': delta('agent', 'supervised')}
    os.makedirs(args.save_root, exist_ok=True)
    dst = os.path.join(args.save_root, f'transfer_{args.dataset}_folds_summary.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print('\n  saved -> ' + dst)
    print('=' * 78)
    print('  READ: if `agent - supervised` is positive in most folds and its mean-std > 0,')
    print('  the RL transfer advantage is stable, not fold-0 luck -> a defensible claim.')
    print('=' * 78)


if __name__ == '__main__':
    main()