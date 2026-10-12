#!/usr/bin/env python3
"""
Combine per-fold results from cv_fold.py into the final CV summary — the same
pooled / scene-mean / category-mean numbers the monolithic --stage cv prints.

  python cv_aggregate.py --save_dir models/drl_gate_rgb_cv     --out cv_rgb_agent.json
  python cv_aggregate.py --save_dir models/drl_gate_rgb_cv_mlp --out cv_rgb_mlp.json
"""
import argparse, glob, json, os
import numpy as np
import opus_final_project_metrics as T


def f1(c):
    return T.ENV.f1_from_counts(*c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--out', default='cv_summary.json')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.save_dir, 'fold*_results.json')))
    if not files:
        raise SystemExit(f'no fold*_results.json in {args.save_dir} — run cv_fold.py first')

    scene = {}            # sid -> {cat, base, agent, A, B}
    per_fold = {}         # fold -> list of (cat, base_f1, agent_f1)
    for fp in files:
        d = json.load(open(fp))
        kf = d['fold']
        per_fold.setdefault(kf, [])
        for sid, r in d['per_scene'].items():
            scene[sid] = r
            per_fold[kf].append((r['cat'], r['base'], r['agent']))
    print(f'loaded {len(files)} folds, {len(scene)} held-out scenes total')

    # ---- category-mean over ALL scenes (macro of macro) ----
    def catmean(rows):
        by = {}
        for cat, b, a in rows:
            by.setdefault(cat, []).append((b, a))
        pc = {c: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
              for c, v in by.items()}
        return float(np.mean([v[0] for v in pc.values()])), \
               float(np.mean([v[1] for v in pc.values()])), pc

    all_rows = [(r['cat'], r['base'], r['agent']) for r in scene.values()]
    base_cm, agent_cm, per_cat = catmean(all_rows)

    # ---- pooled (micro over all held-out pixels) ----
    poolA = [0, 0, 0, 0]; poolB = [0, 0, 0, 0]
    for r in scene.values():
        for k in range(4):
            poolA[k] += r['A'][k]; poolB[k] += r['B'][k]
    pooled_base, pooled_agent = f1(poolB), f1(poolA)

    # ---- scene-mean (macro over scenes) ----
    sm_base = float(np.mean([r['base'] for r in scene.values()]))
    sm_agent = float(np.mean([r['agent'] for r in scene.values()]))

    # ---- per-fold category-mean ----
    fold_b, fold_a = [], []
    for kf in sorted(per_fold):
        b, a, _ = catmean(per_fold[kf])
        fold_b.append(b); fold_a.append(a)
    fb, fa = np.array(fold_b), np.array(fold_a)

    print('\n' + '=' * 72)
    print(f'  CROSS-VALIDATION RESULT  ({len(files)}-fold, every scene tested once)')
    print('=' * 72)
    print(f"  {'category':<26}{'base':>9}{'agent':>9}{'gain':>9}")
    for c in sorted(per_cat):
        b, a = per_cat[c]
        print(f"  {c:<26}{b:>9.4f}{a:>9.4f}{a-b:>+9.4f}")
    print('  ' + '-' * 52)
    gains = [a - b for b, a in per_cat.values()]
    print(f"  {'CATEGORY-MEAN (CDnet)':<26}{base_cm:>9.4f}{agent_cm:>9.4f}{agent_cm-base_cm:>+9.4f}")
    print(f"  worst category gain = {min(gains):+.4f}   |   improved = {sum(g>0 for g in gains)}/{len(gains)}")
    print(f"\n  Per-fold category-mean ({len(fa)} folds):")
    print(f"    base  = {fb.mean():.4f} +/- {fb.std():.4f}")
    print(f"    agent = {fa.mean():.4f} +/- {fa.std():.4f}")
    print(f"    gain  = {(fa-fb).mean():+.4f} +/- {(fa-fb).std():.4f}")

    print(f"\n  {'aggregation':<28}{'base':>10}{'agent':>10}{'gain':>10}")
    print(f"  {'pooled (= Liang Overall)':<28}{pooled_base:>10.4f}{pooled_agent:>10.4f}{pooled_agent-pooled_base:>+10.4f}")
    print(f"  {'scene-mean':<28}{sm_base:>10.4f}{sm_agent:>10.4f}{sm_agent-sm_base:>+10.4f}")
    print(f"  {'category-mean (CDnet)':<28}{base_cm:>10.4f}{agent_cm:>10.4f}{agent_cm-base_cm:>+10.4f}")
    print('=' * 72)

    out = {'n_folds': len(files), 'n_scenes': len(scene),
           'pooled': {'base': pooled_base, 'agent': pooled_agent, 'gain': pooled_agent - pooled_base},
           'scene_mean': {'base': sm_base, 'agent': sm_agent, 'gain': sm_agent - sm_base},
           'category_mean': {'base': base_cm, 'agent': agent_cm, 'gain': agent_cm - base_cm},
           'per_fold_catmean': {'base': fold_b, 'agent': fold_a},
           'per_category': {c: {'base': b, 'agent': a, 'gain': a - b} for c, (b, a) in per_cat.items()},
           'worst_category_gain': float(min(gains)),
           'scene_f1': {sid: {'cat': r['cat'], 'base': r['base'], 'agent': r['agent']}
                        for sid, r in scene.items()}}
    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'  saved -> {args.out}')


if __name__ == '__main__':
    main()
