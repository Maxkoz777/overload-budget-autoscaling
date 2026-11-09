from common import *
import time
for ds in ("alibaba200","huawei2023","azure2019"):
    t=time.time(); D=r7.load(ds,"test")
    print(ds, D["y"].shape, "t0",D["t0"],"T",D["T"], f"{time.time()-t:.1f}s", flush=True)
    for d in (0.01,0.05):
        W=SEL_W[(ds,d)]
        for tau in (0,1,5):
            f,c=pac_h(D,W,d,tau)
            print(f"  delta={d} W={W} tau={tau}: within {int((f<=d+1e-12).sum())}/{len(f)} meancost {c.mean():.3f}", flush=True)
