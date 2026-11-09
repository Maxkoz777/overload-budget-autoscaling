from baselines import *
import pandas as pd, json, time
V={"P":{}, "V1_c_small":dict(c_mult=0.1),"V2_c_large":dict(c_mult=10.0),"V3_no_limit":dict(max_change=np.inf),
   "V4_hpa_limit":dict(max_change='hpa'),"V5_A_missed":dict(A_mode='missed'),"V6_no_trigger":dict(trigger=False),
   "V7_no_limit_no_trigger":dict(max_change=np.inf,trigger=False),"V8_hpa_limit_no_trigger":dict(max_change='hpa',trigger=False)}
sel={tuple([k.split('|')[0],float(k.split('|')[1])]):v for k,v in json.load(open('out/gamma_selected.json')).items()}
rows=[]; diag=[]
for ds in ("alibaba200","huawei2023","azure2019"):
    D=r7.load(ds,"test"); T=D["T"]; dem=D["y"][:,D["t0"]:D["t0"]+T]
    for d in (0.01,0.05):
        W=SEL_W[(ds,d)]
        for tau in (0,1,5):
            t=time.time()
            for name,kw in V.items():
                f,c=r7.evaluate(D,bacc(D,d,tau,5,**kw),tau,tau)
                for u in range(len(f)): rows.append((ds,d,tau,name,D["uid"][u],f[u],c[u]))
                print(f"{ds} d={d} tau={tau} {name:24s} within {int((f<=d+1e-12).sum())}/{len(f)} mean {c.mean():.3f} med {np.median(c):.3f}",flush=True)
            gm=sel[(ds,d)]; ast=alpha_star(d,gm,tau,T)
            tracks=[(0.005,d)]+([(gm,ast)] if ast>0 else [])
            decs,sat,err=aci_tracks(D,W,tau,tracks)
            for k,lab in enumerate(["ACI","ACI-G"][:len(tracks)]):
                dec=decs[k]; cap=dec[:, :T] if D["warm"] else np.concatenate([np.repeat(D["init"][:,None],tau,1),dec[:,tau:T]],1) if tau else dec[:,:T]
                M=Mcap(D); ov=dem>D["mu"][:,None]*cap; satm=cap==M[:,None]
                for u in range(len(err[k])):
                    diag.append((ds,d,tau,lab,D["uid"][u],err[k][u],int(ov[u].sum()),int((ov[u]&~satm[u]).sum()),int((ov[u]&satm[u]).sum()),int(sat[k][u])))
                e=err[k]; print(f"   {lab}: max err {e.max():.4f} (target {tracks[k][1]:.4f}, delta {d}); units with err>delta {(e>d+1e-12).sum()}; overload in saturated minutes: units {(np.array([r[8] for r in diag[-len(e):]])>0).sum()}",flush=True)
            print(f"   [{time.time()-t:.0f}s]",flush=True)
    pd.DataFrame(rows,columns=["dataset","budget","tau","variant","unit_id","overload_fraction","relative_cost"]).to_csv("out/A1_bacc_variants.csv.gz",index=False)
    pd.DataFrame(diag,columns=["dataset","budget","tau","policy","unit_id","aci_err","overload_min","overload_outside_sat","overload_in_sat","saturated_min"]).to_csv("out/A1_aci_diag.csv.gz",index=False)
print("DONE",flush=True)
