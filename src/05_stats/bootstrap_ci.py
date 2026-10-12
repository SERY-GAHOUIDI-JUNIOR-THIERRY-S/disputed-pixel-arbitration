#!/usr/bin/env python3
"""
Bootstrap confidence intervals over the 53 CDnet scenes for the primary metrics,
to accompany the paired Wilcoxon p-values (addresses the reviewer request for CIs;
Efron & Tibshirani is already ref [30]).

Resamples SCENES with replacement (B=10000), recomputes the category-mean for each
system on each resample, and reports the 95% percentile CI of the differences
agent-framework and agent-MLP.

  python bootstrap_ci.py --file headtohead_rgb.json
  python bootstrap_ci.py --file headtohead_rgb.json --B 10000 --seed 0

Schema-tolerant: locates the per-scene dict and reads framework / agent / mlp F1 and
the scene's category under the common key spellings we used.
"""
import argparse, json
import numpy as np

FW  = ['framework', 'fw', 'comm', 'stage3', 'base_fw']
AG  = ['agent', 'f1', 'drl', 'rl']
MLP = ['mlp', 'supervised', 'sup', 'bc']
CAT = ['cat', 'category']


def find_ps(d):
    for k in ('per_scene', 'scene_f1', 'scenes'):
        if isinstance(d.get(k), dict):
            return d[k]
    if all(isinstance(v, dict) for v in d.values()):
        return d
    raise SystemExit('no per-scene dict found')


def pick(rec, keys):
    for k in keys:
        if k in rec and isinstance(rec[k], (int, float)):
            return float(rec[k])
    return None


def catmean(rows):
    """rows: list of (cat, f1). category-mean = mean over categories of per-scene mean."""
    by = {}
    for c, v in rows:
        by.setdefault(c, []).append(v)
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    ps = find_ps(json.load(open(a.file)))
    scenes = []
    for sid, rec in ps.items():
        cat = None
        for k in CAT:
            if isinstance(rec.get(k), str):
                cat = rec[k]; break
        if cat is None:
            cat = sid.split('/')[0]
        fw, ag, mlp = pick(rec, FW), pick(rec, AG), pick(rec, MLP)
        scenes.append((cat, fw, ag, mlp))
    n = len(scenes)
    print(f'{a.file}: {n} scenes, B={a.B}')

    def cm(sel, idx):
        rows = [(scenes[i][0], scenes[i][idx]) for i in sel if scenes[i][idx] is not None]
        return catmean(rows) if rows else float('nan')

    d_agfw, d_agmlp, agent_abs = [], [], []
    for _ in range(a.B):
        sel = rng.integers(0, n, n)
        fw, ag, mlp = cm(sel, 1), cm(sel, 2), cm(sel, 3)
        agent_abs.append(ag)
        d_agfw.append(ag - fw)
        if not np.isnan(mlp):
            d_agmlp.append(ag - mlp)

    def ci(x):
        x = np.array([v for v in x if not np.isnan(v)])
        return np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)

    for name, arr in [('agent category-mean', agent_abs),
                      ('agent - framework', d_agfw),
                      ('agent - MLP', d_agmlp)]:
        m, lo, hi = ci(arr)
        print(f'  {name:22s}: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    main()
