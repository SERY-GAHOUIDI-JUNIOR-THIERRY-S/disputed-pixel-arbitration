#!/usr/bin/env python3
"""
================================================================================
MODEL-COMMUNICATION DIAGNOSTIC  —  localize where F1 is lost, stage by stage.
================================================================================
Reuses your framework file (default liang_v7.py) and the corrected guidance.
For each scene it reports, on CDnet-valid pixels:

  guidance-FG% , CPB(stage1)-FG% , agreement% , Case1% (CPB=FG/guid=BG) , Case2%
  per-stage F1 :  guidance | stage1(CPB) | stage2 | stage3(final) | ORACLE
  Case-1 transitions (stage1->final):  kept-FG , flipped-BG   (+ how many were TRUE fg)
  Case-2 transitions (stage1->final):  promoted-FG , left-BG  (+ how many were TRUE fg)
  support stats:  mean live supports , zero-support%

Reading it:
  * guidance-FG% ~0 on a scene with real motion  -> guidance blind (OOD)
  * CPB-FG% huge with low precision               -> CPB flooding (base problem)
  * big Case1 with 'flipped-BG' >> 'kept-FG' and many were TRUE fg -> Stage2 deletes real fg
  * big Case2 'promoted-FG' but mostly NOT true fg -> Stage3 over-promotes (flood)
  * stageN F1 << oracle with small Case%          -> resolvable dispute (arbiter headroom)
  * stageN F1 << oracle with Case% ~ whole frame  -> oracle is GT-inflated (NOT reachable)

USAGE
  python cpb_comm_diagnose.py --framework liang_v7.py --models_dir models/cpb_mad \
      --guidance deeplab --device cpu --decision blockmean --stage2_mode paper \
      --floor_min 6.0 --n_frames 30 --category turbulence
================================================================================
"""
import argparse, importlib.util, os, sys, time
import numpy as np, cv2

def load_framework(path):
    spec = importlib.util.spec_from_file_location("fw", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def case_stats(s1_fg, guid_fg, s_final_fg, gt_fg, valid):
    """Return dict of dispute counts + transition breakdown, all on valid pixels."""
    v = valid
    agree = (s1_fg == guid_fg) & v
    case1 = s1_fg & (~guid_fg) & v            # CPB=FG, guid=BG
    case2 = (~s1_fg) & guid_fg & v            # CPB=BG, guid=FG
    d = {}
    d['valid'] = int(v.sum())
    d['guidFG'] = int((guid_fg & v).sum())
    d['cpbFG']  = int((s1_fg & v).sum())
    d['agree']  = int(agree.sum())
    d['case1']  = int(case1.sum())
    d['case2']  = int(case2.sum())
    # Case1 transitions: stage1 said FG; what did final say?
    d['c1_keptFG']   = int((case1 & s_final_fg).sum())
    d['c1_flipBG']   = int((case1 & (~s_final_fg)).sum())
    d['c1_true_fg']  = int((case1 & gt_fg).sum())          # of disputed, how many really fg
    d['c1_keptFG_correct'] = int((case1 & s_final_fg & gt_fg).sum())
    d['c1_flipBG_correct'] = int((case1 & (~s_final_fg) & (~gt_fg)).sum())
    # Case2 transitions: stage1 said BG; what did final say?
    d['c2_promoteFG'] = int((case2 & s_final_fg).sum())
    d['c2_leftBG']    = int((case2 & (~s_final_fg)).sum())
    d['c2_true_fg']   = int((case2 & gt_fg).sum())
    d['c2_promote_correct'] = int((case2 & s_final_fg & gt_fg).sum())
    return d

def f1_of(counts):
    tp,fp,fn,tn = counts
    e=1e-10; pr=tp/(tp+fp+e); rc=tp/(tp+fn+e); return 2*pr*rc/(pr+rc+e), pr, rc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--framework', default='liang_v7.py')
    ap.add_argument('--models_dir', default='models/cpb_mad')
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--guidance', default='deeplab')
    ap.add_argument('--guidance_models_dir', default='./models')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--category', default=None)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--decision', default='blockmean', choices=['all64','blockmean'])
    ap.add_argument('--stage2_mode', default='paper', choices=['port','paper'])
    ap.add_argument('--floor_min', type=float, default=6.0)
    ap.add_argument('--block_frac', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--af', type=float, default=2.5)
    ap.add_argument('--sigema', type=float, default=0.8)
    ap.add_argument('--bth_pixels', type=float, default=7.0)
    ap.add_argument('--ref', type=int, default=20)
    ap.add_argument('--n_frames', type=int, default=30)
    ap.add_argument('--channel_mode', default='auto')
    args=ap.parse_args()

    fw=load_framework(args.framework)
    root=args.dataset_root or fw.find_root()
    if not root: raise SystemExit('CDnet not found; use --dataset_root')
    guid=fw.make_guidance(args.guidance, False, models_dir=args.guidance_models_dir,
                          allow_fallback=False, device=args.device)
    cats = ({args.category:([args.scene] if args.scene else fw.CDNET2014.get(args.category,[]))}
            if args.category else fw.CDNET2014)
    dm = 1 if args.decision=='blockmean' else 0
    s2m = 1 if args.stage2_mode=='paper' else 0
    GT_FG=fw.GT_FG; EXCL=fw.GT_EXCLUDE

    print(f"{'='*100}\n  MODEL-COMMUNICATION DIAGNOSTIC   framework={args.framework}  guidance={guid.name}")
    print(f"  decision={args.decision}  stage2_mode={args.stage2_mode}  floor={args.floor_min}  models={args.models_dir}\n{'='*100}")

    for cat,scenes in cats.items():
        for sc in scenes:
            mp=os.path.join(args.models_dir, f'cpb_paper_{cat}_{sc}.pkl')
            if not os.path.exists(mp):
                print(f"  [skip] {cat}/{sc} model missing"); continue
            import pickle; model=pickle.load(open(mp,'rb')); p=model['params']
            rs,re_=int(p['roi_start']),int(p['roi_end'])
            sp=os.path.join(root,cat,sc); idir=os.path.join(sp,'input'); gdir=os.path.join(sp,'groundtruth')
            nums=[n for n in fw.all_nums(sp) if rs<=n<=re_ and os.path.exists(os.path.join(gdir,f'gt{n:06d}.png'))]
            if args.n_frames>0 and len(nums)>args.n_frames:
                nums=nums[::max(1,len(nums)//args.n_frames)][:args.n_frames]
            if not nums: print(f"  [skip] {cat}/{sc} no ROI frames"); continue
            def mk(mode):
                return fw.HoDPipeline(model, mode, args.floor_min, args.af, args.sigema, args.bth_pixels,
                                      args.ref, True, args.channel_mode, args.block_frac, args.lam, dm, s2m)
            pipe2=mk('stage2'); pipe3=mk('stage3')
            agg={}; c_s1=[0,0,0,0]; c_s2=[0,0,0,0]; c_s3=[0,0,0,0]; c_g=[0,0,0,0]; c_or=[0,0,0,0]
            supp_sum=0.0; supp_zero=0; supp_n=0; t0=time.time()
            for n in nums:
                bgr=fw.load_bgr(idir,n); gt=fw.load_gt(gdir,n)
                if bgr is None or gt is None: continue
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                gm=guid.predict(bgr)
                s3=pipe3.run(rgb,gm); s1=pipe3.last_s1; supp=pipe3.last_cpb_supp
                s2=pipe2.run(rgb,gm)
                for arr in (s1,s2,s3):
                    pass
                def R(m):
                    return cv2.resize(m,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST) if m.shape!=gt.shape else m
                s1=R(s1); s2=R(s2); s3=R(s3); gmR=R(gm); suppR=R(supp.astype(np.float32))
                valid=np.ones(gt.shape,bool)
                for e in EXCL: valid&=(gt!=e)
                gt_fg=(gt==GT_FG); g_fg=(gmR>127); s1f=(s1==255); s2f=(s2==255); s3f=(s3==255)
                # per-stage confusion
                def acc(c, predfg):
                    c[0]+=int((predfg&gt_fg&valid).sum()); c[1]+=int((predfg&(~gt_fg)&valid).sum())
                    c[2]+=int(((~predfg)&gt_fg&valid).sum()); c[3]+=int(((~predfg)&(~gt_fg)&valid).sum())
                acc(c_g,g_fg); acc(c_s1,s1f); acc(c_s2,s2f); acc(c_s3,s3f)
                agree=(s1f==g_fg); orc=np.where(agree,s1f,gt_fg); acc(c_or,orc)
                # case stats (use final s3)
                d=case_stats(s1f,g_fg,s3f,gt_fg,valid)
                for k,v in d.items(): agg[k]=agg.get(k,0)+v
                supp_sum+=float(suppR[valid].sum()); supp_zero+=int((suppR[valid]==0).sum()); supp_n+=int(valid.sum())
            V=max(agg.get('valid',1),1)
            f1g,_,_=f1_of(c_g); f1s1,ps1,rs1=f1_of(c_s1); f1s2,_,_=f1_of(c_s2); f1s3,p3,r3=f1_of(c_s3); f1o,_,_=f1_of(c_or)
            print(f"\n  {cat}/{sc}   ({time.time()-t0:.0f}s, {len(nums)} frames)")
            print(f"    masks:  guidFG={100*agg['guidFG']/V:5.1f}%  cpbFG(s1)={100*agg['cpbFG']/V:5.1f}%  "
                  f"agree={100*agg['agree']/V:5.1f}%  Case1(cpbFG/guidBG)={100*agg['case1']/V:5.1f}%  Case2={100*agg['case2']/V:5.1f}%")
            print(f"    F1   :  guid={f1g:.3f}  stage1={f1s1:.3f}(P{ps1:.2f}/R{rs1:.2f})  stage2={f1s2:.3f}  "
                  f"stage3={f1s3:.3f}(P{p3:.2f}/R{r3:.2f})  ORACLE={f1o:.3f}")
            c1=max(agg['case1'],1); c2=max(agg['case2'],1)
            print(f"    Case1 ({agg['case1']} px, {100*agg['c1_true_fg']/c1:.0f}% truly fg):  keptFG={agg['c1_keptFG']} "
                  f"(correct {agg['c1_keptFG_correct']})  flipBG={agg['c1_flipBG']} (correct {agg['c1_flipBG_correct']})")
            print(f"    Case2 ({agg['case2']} px, {100*agg['c2_true_fg']/c2:.0f}% truly fg):  promoteFG={agg['c2_promoteFG']} "
                  f"(correct {agg['c2_promote_correct']})  leftBG={agg['c2_leftBG']}")
            print(f"    support: mean={supp_sum/max(supp_n,1):.1f}  zero-support={100*supp_zero/max(supp_n,1):.1f}%")

if __name__=='__main__':
    main()