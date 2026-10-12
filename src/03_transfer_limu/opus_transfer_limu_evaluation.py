#!/usr/bin/env python3
"""
================================================================================
CROSS-DATASET TRANSFER EVALUATION
================================================================================
Applies a CDnet-trained arbiter ZERO-SHOT to another dataset (LIMU / WallFlower),
with CPB trained per-scene on that dataset (as Liang does). Reports
base / guidance / agent / oracle under all three aggregations, so it is directly
comparable to your CDnet table and to Liang Tables 3 & 5.

It REUSES your existing modules unchanged:
  - opus_final_hod_framework            (CPB/HoD kernels, metrics, guidance)
  - opus_drl_env_junior*                (FrameEnv, f1_from_counts)
  - your trainer module                 (Scene, build_scenes, build_gate, RunningNorm)

STAGES
  precompute : compute + cache the guidance (binary mask + soft prob) for the
               target scenes  (needs the per-scene CPB .pkl for the ROI + size).
  eval       : load the CDnet agent checkpoint and score base/guid/agent/oracle.

USAGE
  # 0) SELF-TEST the harness on CDnet TODAY (before LIMU exists):
  python opus_transfer_limu_evaluation.py --stage eval --dataset cdnet \
      --models_dir models/cpb_numba --cache_dir drl_cache_junior \
      --agent_ckpt models/drl_gate_cv_metrics/fold0/best.pth

  # 1) once LIMU is converted + its CPB models trained + guidance cached:
  python opus_transfer_limu_evaluation.py --stage precompute --dataset limu \
      --models_dir models/cpb_limu --cache_dir drl_cache_limu --n_frames 0
  python opus_transfer_limu_evaluation.py --stage eval --dataset limu \
      --models_dir models/cpb_limu --cache_dir drl_cache_limu \
      --agent_ckpt models/drl_gate_full_cdnet/best.pth --sup_ckpt models/sup_gate_all/sup.pth
================================================================================
"""
import argparse, json, os, pickle
from types import SimpleNamespace
import numpy as np
import cv2

try:
    import opus_final_hod_framework as F
except Exception:
    import opus_liang_hod_framework as F
try:
    import opus_drl_env_junior_enhanced_with_patch_cnn as ENV
except Exception:
    import opus_drl_env_junior as ENV

T = None
for _name in ('opus_final_project_metrics',
              'opus_final_drl_train_junior_enhanced_with_patch_cnn',
              'opus_final_drl_train_junior'):
    try:
        T = __import__(_name); break
    except Exception:
        continue
if T is None:
    raise SystemExit('Could not import your trainer module (Scene/build_scenes/build_gate).')

# ---- target dataset scene registries (cat/scene ids, CDnet-style on disk) ----
LIMU_SCENES = {'limu': ['CameraParameter', 'Intersection', 'LightSwitch']}
WALLFLOWER_SCENES = {'wallflower': ['Bootstrap', 'Camouflage', 'ForegroundAperture',
                                    'LightSwitch', 'MovedObject', 'TimeOfDay', 'WavingTrees']}


def dataset_ids(name):
    if name == 'cdnet':
        d = F.CDNET2014
    elif name == 'limu':
        d = LIMU_SCENES
    elif name == 'wallflower':
        d = WALLFLOWER_SCENES
    else:
        raise SystemExit(f'unknown dataset {name}')
    return [f'{cat}/{sc}' for cat, scs in d.items() for sc in scs]


def resolve_root(args):
    return args.dataset_root or F.find_root()


# ============================================================================
# PRECOMPUTE guidance (binary mask + soft prob) for the target scenes
# ============================================================================
def precompute(args):
    root = resolve_root(args)
    os.makedirs(args.cache_dir, exist_ok=True)
    guidance = F.make_guidance(args.guidance, False, models_dir=args.guidance_models_dir,
                               allow_fallback=False, device=args.device)
    for sid in dataset_ids(args.dataset):
        cat, scene = sid.split('/')
        out = os.path.join(args.cache_dir, f'{cat}__{scene}.npz')
        if os.path.exists(out) and not args.force:
            print(f'  [skip cached] {sid}'); continue
        mp = os.path.join(args.models_dir, f'cpb_paper_{cat}_{scene}.pkl')
        sp = os.path.join(root, cat, scene)
        if not (os.path.exists(mp) and os.path.isdir(sp)):
            print(f'  [skip missing] {sid}  (model={os.path.exists(mp)} dir={os.path.isdir(sp)})'); continue
        model = pickle.load(open(mp, 'rb'))
        rs, re_ = int(model['params']['roi_start']), int(model['params']['roi_end'])
        idir = os.path.join(sp, 'input'); gdir = os.path.join(sp, 'groundtruth')
        nums = [n for n in F.all_nums(sp) if rs <= n <= re_
                and F.load_gt(gdir, n) is not None]
        if args.n_frames and args.n_frames > 0 and len(nums) > args.n_frames:
            nums = nums[::max(1, len(nums) // args.n_frames)][:args.n_frames]
        if not nums:
            print(f'  [skip no ROI frames] {sid}'); continue
        guids, probs = [], []
        for n in nums:
            bgr = F.load_bgr(idir, n)
            guids.append(guidance.predict(bgr) if bgr is not None else None)
            probs.append(T._guidance_prob(guidance, bgr) if bgr is not None else None)
        H, W = guids[0].shape
        arr = np.stack([g if g is not None else np.zeros((H, W), np.uint8) for g in guids]).astype(np.uint8)
        payload = {'nums': np.array(nums), 'guid': arr}
        if all(p is not None for p in probs):
            payload['prob'] = np.stack([cv2.resize(p, (W, H)) if p.shape != (H, W) else p
                                        for p in probs]).astype(np.float16)
        np.savez_compressed(out, **payload)
        print(f'  cached {sid}  ({len(nums)} frames{" +prob" if "prob" in payload else " binary-only"})')


# ============================================================================
# EVAL : base / guidance / agent / oracle  (three aggregations)
# ============================================================================
def _reduce(d):
    pooled = ENV.f1_from_counts(*d['A'])
    scene_mean = float(np.mean(list(d['scene'].values()))) if d['scene'] else 0.0
    percat = {c: ENV.f1_from_counts(*cnts) for c, cnts in d['cat'].items()}
    cat_mean = float(np.mean(list(percat.values()))) if percat else 0.0
    return {'pooled': pooled, 'scene_mean': scene_mean, 'category_mean': cat_mean, 'per_cat': percat}


def eval_transfer(args):
    import torch
    root = resolve_root(args)
    envargs = dict(af=args.af, floor=args.floor_min, bth=args.bth_pixels, ref=args.ref, sigema=args.sigema)
    sc_args = SimpleNamespace(cache_dir=args.cache_dir, models_dir=args.models_dir, win_size=args.win_size)
    scenes = T.build_scenes(dataset_ids(args.dataset), sc_args, root, envargs)
    if not scenes:
        raise SystemExit('No scenes built — check --models_dir / --cache_dir (run precompute first).')

    ck = torch.load(args.agent_ckpt, map_location='cpu')
    if ck.get('state_dim', ENV.FrameEnv.STATE_DIM) != ENV.FrameEnv.STATE_DIM:
        raise SystemExit(f"checkpoint state_dim={ck.get('state_dim')} != env {ENV.FrameEnv.STATE_DIM}")
    use_patch = ck.get('use_patch', False); patch_k = ck.get('patch_k', 4)
    n_act = ck.get('n_actions', 3)
    gate = T.build_gate(ck['state_dim'], n_act, 0.0, use_patch, patch_k)
    gate.load_state_dict(ck['gate']); gate.eval()
    print(f"  [agent] use_patch={use_patch} patch_k={patch_k} n_actions={n_act} "
          f"bc_coef={ck.get('bc_coef','?')}->{ck.get('bc_final','?')}  "
          f"(confound check: patch-CNN gives the agent spatial features the scalar MLP lacks)")
    act_tot = np.zeros(int(n_act), np.int64)   # LIMU action histogram
    norm = T.RunningNorm(ENV.FrameEnv.STATE_DIM)
    norm.mean = ck['norm_mean']; norm.M2 = ck['norm_M2']; norm.n = ck['norm_n']; norm.frozen = True

    # optional supervised arbiter (loads 2-way OR 3-way MLP; head width from ckpt)
    sup = None
    if args.sup_ckpt:
        import torch.nn as nn
        sck = torch.load(args.sup_ckpt, map_location='cpu')
        if sck.get('state_dim', ENV.FrameEnv.STATE_DIM) != ENV.FrameEnv.STATE_DIM:
            raise SystemExit(f"sup_ckpt state_dim={sck.get('state_dim')} != env {ENV.FrameEnv.STATE_DIM}")
        sdim = sck['state_dim']
        s_nact = int(sck.get('n_actions', 2))           # 2-way (legacy) or 3-way (keep-baseline)
        if sck.get('sup_model', 'mlp') == 'linear':
            snet = nn.Linear(sdim, s_nact)
        else:
            snet = nn.Sequential(nn.Linear(sdim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, s_nact))
        snet.load_state_dict(sck['sup_state']); snet.eval()
        smu = np.asarray(sck['mu'], np.float32); ssd = np.asarray(sck['sd'], np.float32)
        sup = (snet, smu, ssd)
        sup_act = np.zeros(s_nact, np.int64)             # supervised action histogram
        print(f"  [supervised] loaded {args.sup_ckpt}  (trained_on={sck.get('trained_on','?')}, "
              f"scenes={sck.get('n_train_scenes','?')}, model={sck.get('sup_model','mlp')}, "
              f"n_actions={s_nact}{' (3-way keep-baseline)' if s_nact >= 3 else ' (2-way)'})")

    systems = ['base', 'guidance'] + (['supervised'] if sup else []) + ['agent', 'oracle']
    acc = {s: {'A': [0, 0, 0, 0], 'scene': {}, 'cat': {}} for s in systems}
    for sc in scenes:
        cat = sc.sid.split('/')[0]
        per = {s: [0, 0, 0, 0] for s in systems}
        for i in range(len(sc.nums)):
            e = sc.env(i); r, c = e.dispute_ij()
            base_mask = e.baseline_mask()
            guid_mask = e._to_native((e.guid_fg > 0).astype(np.uint8) * 255)
            if len(r) == 0:
                agent_mask = base_mask
            else:
                Sn = norm(e.state(r, c))
                Pt = torch.from_numpy(e.patches(r, c, patch_k)) if use_patch else None
                with torch.no_grad():
                    actions = gate(torch.from_numpy(Sn), Pt).argmax(-1).numpy()
                agent_mask = T._segment_gate3(e, actions, r, c)
                bc = np.bincount(actions, minlength=len(act_tot))
                act_tot[:len(bc)] += bc[:len(act_tot)]
            # supervised MLP on the identical disputed pixels
            if sup is not None:
                if len(r) == 0:
                    sup_mask = base_mask
                else:
                    snet, smu, ssd = sup
                    Xn = ((e.state(r, c).astype(np.float32) - smu) / ssd).astype(np.float32)
                    with torch.no_grad():
                        sact = snet(torch.from_numpy(Xn)).argmax(-1).numpy()
                    sup_mask = T._segment_gate3(e, sact, r, c)
                    sbc = np.bincount(sact, minlength=len(sup_act))
                    sup_act[:len(sbc)] += sbc[:len(sup_act)]
            oracle_fit = e.s1_default.copy()
            if len(r):
                oracle_fit[r, c] = (e.gtfg_fit[r, c].astype(np.uint8) * 255)
            oracle_mask = e._to_native(oracle_fit)
            masks = [('base', base_mask), ('guidance', guid_mask)]
            if sup is not None:
                masks.append(('supervised', sup_mask))
            masks += [('agent', agent_mask), ('oracle', oracle_mask)]
            for name, mask in masks:
                for k, v in enumerate(e.counts(mask)):
                    per[name][k] += v
        for s in systems:
            a = per[s]
            acc[s]['scene'][sc.sid] = ENV.f1_from_counts(*a)
            acc[s]['cat'].setdefault(cat, [0, 0, 0, 0])
            for k in range(4):
                acc[s]['cat'][cat][k] += a[k]; acc[s]['A'][k] += a[k]

    red = {s: _reduce(acc[s]) for s in systems}
    print(f'\n{"="*82}\n  TRANSFER EVAL  dataset={args.dataset}  agent={os.path.basename(args.agent_ckpt)}\n{"="*82}')
    print(f"  {'system':<12}{'pooled':>12}{'scene-mean':>14}{'category-mean':>16}")
    for s in systems:
        print(f"  {s:<12}{red[s]['pooled']:>12.4f}{red[s]['scene_mean']:>14.4f}{red[s]['category_mean']:>16.4f}")
    print('  ' + '-' * 78)
    b, g, a = red['base'], red['guidance'], red['agent']
    print(f"  agent - base       : pooled {a['pooled']-b['pooled']:+.4f}   cat {a['category_mean']-b['category_mean']:+.4f}")
    print(f"  agent - guidance   : pooled {a['pooled']-g['pooled']:+.4f}   cat {a['category_mean']-g['category_mean']:+.4f}")
    if sup is not None:
        s = red['supervised']
        print(f"  supervised - guid  : pooled {s['pooled']-g['pooled']:+.4f}   cat {s['category_mean']-g['category_mean']:+.4f}")
        print(f"  agent - supervised : pooled {a['pooled']-s['pooled']:+.4f}   cat {a['category_mean']-s['category_mean']:+.4f}"
              f"   <-- KEY: RL vs supervised under transfer")
    print(f"  (Liang LIMU CPB+DeepLabV3+ pooled 'Overall' ~= 0.793; guidance-alone ~= 0.50)")
    print('=' * 82)
    # per-scene detail (columns follow the active systems)
    hdr = ''.join(f"{s[:9]:>9}" for s in systems)
    print(f"  {'scene':<32}{hdr}")
    for sid in acc['base']['scene']:
        row = ''.join(f"{acc[s]['scene'][sid]:>9.3f}" for s in systems)
        print(f"  {sid:<32}{row}")

    # agent action mix on the target dataset (confound check: is action-2 used?)
    tot = int(act_tot.sum())
    if tot:
        names = ['trust-CPB', 'trust-guid', 'keep-baseline'][:len(act_tot)]
        mix = '  '.join(f'{nm}={act_tot[k]/tot:.3f}' for k, nm in enumerate(names))
        print('  ' + '-' * 78)
        print(f"  agent action mix on {args.dataset}:  {mix}")
        if len(act_tot) >= 3:
            k2 = act_tot[2] / tot
            print(f"  -> keep-baseline used {k2:.1%}. If ~0, the agent is effectively 2-way like the MLP,")
            print(f"     so its win is NOT from the extra action. If large, part of the gain is action-space.")
    if sup is not None:
        stot = int(sup_act.sum())
        if stot:
            snames = ['trust-CPB', 'trust-guid', 'keep-baseline'][:len(sup_act)]
            smix = '  '.join(f'{nm}={sup_act[k]/stot:.3f}' for k, nm in enumerate(snames))
            print(f"  supervised action mix on {args.dataset}:  {smix}")
    print('=' * 82)

    out = {'dataset': args.dataset, 'agent_ckpt': args.agent_ckpt, 'agent_action_mix': act_tot.tolist(),
           'systems': {s: {k: v for k, v in red[s].items() if k != 'per_cat'} for s in systems},
           'per_category': {s: red[s]['per_cat'] for s in systems},
           'scene_f1': {sid: {s: acc[s]['scene'][sid] for s in systems} for sid in acc['base']['scene']}}
    if sup is not None:
        out['supervised_action_mix'] = sup_act.tolist()
    os.makedirs(args.save_dir, exist_ok=True)
    dst = os.path.join(args.save_dir, f'transfer_{args.dataset}.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'\n  saved -> {dst}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['precompute', 'eval'])
    ap.add_argument('--dataset', required=True, choices=['cdnet', 'limu', 'wallflower'])
    ap.add_argument('--dataset_root', default=None, help='defaults to F.find_root() (the cdnet2014 folder)')
    ap.add_argument('--models_dir', default='models/cpb_numba')
    ap.add_argument('--cache_dir', default='drl_cache_transfer')
    ap.add_argument('--agent_ckpt', default='models/drl_gate_cv_metrics/fold0/best.pth')
    ap.add_argument('--sup_ckpt', default='', help='supervised MLP checkpoint (sup.pth) to score alongside the agent')
    ap.add_argument('--guidance', default='deeplab')
    ap.add_argument('--guidance_models_dir', default='./models')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_frames', type=int, default=0, help='0 = all ROI frames (precompute)')
    ap.add_argument('--win_size', type=int, default=10)
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--save_dir', default='models/transfer')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.stage == 'precompute':
        precompute(args)
    else:
        eval_transfer(args)


if __name__ == '__main__':
    main()