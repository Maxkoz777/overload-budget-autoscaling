from baselines import *
import pandas as pd, time, json
DS=("alibaba200","huawei2023","azure2019"); BUD=(0.01,0.05); TAUS=(0,1,5)
rows=[]; sel={}
# Step 1: gamma selection on calibration days (tau=0)
for ds in DS:
    D=r7.load(ds,"cal")
    for d in BUD:
        W=SEL_W[(ds,d)]
        cand=[(gm,alpha_star(d,gm,0,D['T'])) for gm in GAMMAS]
        cand=[c for c in cand if c[1]>0]
        decs,sat,err=aci_tracks(D,W,0,cand)
        res=[]
        for k,(gm,ast) in enumerate(cand):
            f,c=r7.evaluate(D,decs[k],0,0)
            res.append(dict(dataset=ds,budget=d,gamma=gm,astar=ast,within=int((f<=d+1e-12).sum()),mean_cost=float(c.mean()),sat_mean=float(sat[k].mean())))
        r=pd.DataFrame(res).sort_values(["within","mean_cost"],ascending=[False,True])
        print(r.to_string(index=False),flush=True)
        sel[(ds,d)]=float(r.iloc[0].gamma)
        pd.DataFrame(res).to_csv(f"out/cal_{ds}_{d}.csv",index=False)
json.dump({f"{k[0]}|{k[1]}":v for k,v in sel.items()},open("out/gamma_selected.json","w"),indent=1)
print("SELECTED",sel,flush=True)
# Step 2: test
for ds in DS:
    D=r7.load(ds,"test")
    for d in BUD:
        W=SEL_W[(ds,d)]
        for tau in TAUS:
            t=time.time()
            out={}
            f,c=pac_h(D,W,d,tau); out["PAC-h"]=(f,c,None)
            gm=sel[(ds,d)]; ast=alpha_star(d,gm,tau,D['T'])
            tracks=[(0.005,d)]+([(gm,ast)] if ast>0 else [])
            decs,sat,err=aci_tracks(D,W,tau,tracks)
            f,c=r7.evaluate(D,decs[0],tau,tau); out["ACI"]=(f,c,sat[0])
            if ast>0:
                f,c=r7.evaluate(D,decs[1],tau,tau); out["ACI-G"]=(f,c,sat[1])
            f,c=r7.evaluate(D,bacc(D,d,tau,5),tau,tau); out["BACC-r"]=(f,c,None)
            f,c=r7.evaluate(D,bacc(D,d,tau,1),tau,tau); out["BACC-r h=1"]=(f,c,None)
            for p,(f,c,s) in out.items():
                for u in range(len(f)):
                    rows.append((ds,d,tau,p,D["uid"][u],f[u],c[u],np.nan if s is None else s[u]))
                print(f"{ds} d={d} tau={tau} {p:11s} within {int((f<=d+1e-12).sum())}/{len(f)} cost mean {c.mean():.3f} med {np.median(c):.3f}"+("" if s is None else f" sat {s.mean():.1f}"),flush=True)
            print(f"   [{time.time()-t:.0f}s]",flush=True)
        pd.DataFrame(rows,columns=["dataset","budget","tau","policy","unit_id","overload_fraction","relative_cost","saturated_minutes"]).to_csv("out/test_per_unit.csv.gz",index=False)
print("DONE",flush=True)
