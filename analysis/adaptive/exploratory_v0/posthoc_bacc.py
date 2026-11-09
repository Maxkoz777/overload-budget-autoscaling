from baselines import *
import pandas as pd
rows=[]
for ds in ("alibaba200","huawei2023","azure2019"):
    D=r7.load(ds,"test")
    for d in (0.01,0.05):
        tau=1
        f,c=r7.evaluate(D,bacc(D,d,tau,5,max_change=np.inf),tau,tau)
        print(f"POSTHOC {ds} d={d} tau=1 BACC-r no-limit within {int((f<=d+1e-12).sum())}/{len(f)} cost mean {c.mean():.3f} med {np.median(c):.3f}",flush=True)
        for u in range(len(f)): rows.append((ds,d,tau,"BACC-r no-limit (post hoc)",D["uid"][u],f[u],c[u]))
pd.DataFrame(rows,columns=["dataset","budget","tau","policy","unit_id","overload_fraction","relative_cost"]).to_csv("out/posthoc_bacc.csv.gz",index=False)
