"""Bootstrap the test set: does adding combine+college to draft pick actually help?"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
import train as T
df = pd.read_csv(T.DATA)
tr = df[df.season <= T.TRAIN_END]; va = df[(df.season > T.TRAIN_END) & (df.season <= T.VAL_END)]; te = df[df.season > T.VAL_END]
y = te[T.TARGET].to_numpy()
P = {}
for g in (["draft"], ["combine","college","draft"], ["combine","college"]):
    prep = T.Prep([c for k in g for c in T.GROUPS[k]]).fit(tr)
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva = tr[T.TARGET].to_numpy(np.float32), va[T.TARGET].to_numpy(np.float32)
    ps = []
    for s in range(10):
        m, _ = T.fit_one(s, Xtr, ytr, Xva, yva)
        with torch.no_grad(): ps.append(m(torch.from_numpy(Xte)).numpy())
    P["nn_" + "+".join(g)] = np.mean(ps, 0)
    P["ridge_" + "+".join(g)] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva]).predict(Xte)
rng = np.random.default_rng(0); n = len(y)
def r2(y, p): return 1 - ((y-p)**2).sum()/((y-y.mean())**2).sum()
for a, b in [("ridge_combine+college+draft", "ridge_draft"), ("nn_combine+college+draft", "nn_draft"), ("ridge_draft", "nn_draft")]:
    d = []
    for _ in range(2000):
        i = rng.integers(0, n, n); d.append(r2(y[i], P[a][i]) - r2(y[i], P[b][i]))
    lo, hi = np.percentile(d, [2.5, 97.5]); print(f"R2[{a}] - R2[{b}]: {np.mean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
# Does college add info *given* draft? partial correlation of college-only model with residual of draft-only
res = y - P["ridge_draft"]
print("corr(residual of draft model, combine+college prediction):", round(spearmanr(res, P["ridge_combine+college"]).statistic, 3))
