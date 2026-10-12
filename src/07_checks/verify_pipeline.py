#!/usr/bin/env python3
"""
Verify the DRL CV will use the FIXED pipeline: RGB + all-64 + paper Stage-2.
Run:  python verify_pipeline.py
Prints a PASS/FAIL line for each check and a final GO / NO-GO.
"""
import inspect, glob, pickle, sys

ok = True

def line(name, passed, detail=''):
    global ok
    tag = 'PASS' if passed else 'FAIL  <-- FIX THIS'
    if not passed:
        ok = False
    print(f'  [{tag}] {name}' + (f'   ({detail})' if detail else ''))

print('=' * 68)
print('  PIPELINE VERIFICATION  (RGB + all-64 + paper Stage-2)')
print('=' * 68)

# ---- 1. F.k_stage2 is the paper version ----
try:
    import opus_final_hod_framework as F
    s2 = inspect.getsource(F.k_stage2)
    is_paper = ('gbar' in s2) or ('gam' in s2)
    line('F.k_stage2 is PAPER (not port)', is_paper,
         'paper markers found' if is_paper else 'looks like the old port')
except Exception as e:
    line('import opus_final_hod_framework / read k_stage2', False, repr(e))

# ---- 2. the env routes through F and has NO stale local stage-2 ----
try:
    import opus_drl_env_junior_enhanced_with_patch_cnn as E
    se = inspect.getsource(E)
    counts = {t: se.count(t) for t in
              ['def k_stage2', 'k_stage2', 'HoDPipeline', 'k_stage1', 'k_stage3',
               '_block_flags', '_flags_blockmean', 'blockmean', 'DBLK_r', 'compute_stage1',
               'opus_final_hod_framework', 'import opus_liang']}
    print('  --- env token counts ---')
    for t, c in counts.items():
        print(f'      {t:26s}: {c}')
    line('env does NOT define its own k_stage2', counts['def k_stage2'] == 0,
         f"def k_stage2 = {counts['def k_stage2']}")
    line('env routes to the framework (F)',
         counts['opus_final_hod_framework'] >= 1 and
         (counts['HoDPipeline'] >= 1 or counts['k_stage2'] >= 1),
         'imports/uses opus_final_hod_framework')
    line('env uses all-64 (not block-mean)',
         counts['_flags_blockmean'] == 0 and counts['blockmean'] == 0,
         f"blockmean={counts['blockmean']} flags_blockmean={counts['_flags_blockmean']}")
    line('env is RGB-aware', counts['DBLK_r'] >= 1, f"DBLK_r refs = {counts['DBLK_r']}")
    if counts['import opus_liang'] >= 1:
        print('      NOTE: env can fall back to opus_liang_hod_framework — make sure '
              'opus_final_hod_framework imports cleanly so the fallback is NOT used.')
except Exception as e:
    line('import env / read source', False, repr(e))

# ---- 3. CPB model is RGB ----
try:
    mp = sorted(glob.glob('models/cpb_numba_rgb/*.pkl'))
    m = pickle.load(open(mp[0], 'rb'))
    is_rgb = all(k in m for k in ('DBLK_r', 'DBLK_g', 'DBLK_b'))
    line('CPB model is RGB (DBLK_r/g/b present)', is_rgb, f'{len(mp)} models found')
except Exception as e:
    line('load RGB CPB model', False, repr(e))

# ---- 4. which framework module actually imports (fallback guard) ----
try:
    import opus_final_hod_framework as F2
    line('opus_final_hod_framework imports cleanly (no fallback)', True, F2.__name__)
except Exception as e:
    line('opus_final_hod_framework imports cleanly', False, repr(e))

print('=' * 68)
print('  RESULT:', 'GO  — the CV will use RGB + all-64 + paper Stage-2'
      if ok else 'NO-GO — fix the FAIL line(s) above before running the CV')
print('=' * 68)
sys.exit(0 if ok else 1)
