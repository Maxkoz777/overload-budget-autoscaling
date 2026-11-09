import sys
import pandas as pd
sys.path.insert(0,'/home/claude/ws/nb')
import pq
pd.read_parquet=lambda p,*a,**k: pd.DataFrame(pq.read(str(p),['timestamp','replica_count','cpu_sum','was_missing']))
from pathlib import Path
AUD = Path('/home/claude/ws/overload-budget-autoscaling/analysis/audit')
sys.path.insert(0, str(AUD))
from canonical_inputs import install
install()
import pac_window_grid as r7
import numpy as np
SEL_W = {('alibaba200',0.01):480, ('alibaba200',0.05):1440, ('huawei2023',0.01):2880, ('huawei2023',0.05):2880, ('azure2019',0.01):480, ('azure2019',0.05):240}
ETA = 0.05

def pac_h(D, W, delta, tau, maxtau=None, rank=None):
    maxtau = tau if maxtau is None else maxtau
    r = rank if rank is not None else r7.pac_rank(W, delta, ETA)
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    m = r7.margins(D, W, tau + 1, (r,), start=-maxtau)
    nom = np.maximum(np.ceil((forecast + m[r]) / D["mu"][:, None]), 1.0)
    return r7.evaluate(D, nom, tau, maxtau)
