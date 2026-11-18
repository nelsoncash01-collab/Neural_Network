"""Leave-one-draft-class-out CV (2010-21) for the POE target, plus single-feature correlations."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
import train as T
df, _ = T.add_poe(pd.read_csv(T.DATA))
dev = df[df.season <= T.VAL_END]
rng = np.random.default_rng(0)
for g in (["combine"], ["college"], ["combine", "college"]):
    feats = [c for k in g for c in T.GROUPS[k]]
    preds = pd.Series(index=dev.index, dtype=float)
    for yr in sorted(dev.season.unique()):  # leave-one-draft-class-out
        tr, te = dev[dev.season != yr], dev[dev.season == yr]
        prep = T.Prep(feats).fit(tr)
        m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(prep(tr), tr.poe)
        preds[te.index] = m.predict(prep(te))
    y, p = dev.poe.to_numpy(), preds.to_numpy()
    rho = spearmanr(y, p).statistic
    boot = [spearmanr(y[i], p[i]).statistic for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    print(f"{'+'.join(g):18s} n={len(y)}  rho={rho:+.3f}  95% CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")
# drafted only (undrafted POE is mostly ~0 noise)
d = dev[dev.undrafted == 0]
feats = T.GROUPS["college"]; preds = pd.Series(index=d.index, dtype=float)
for yr in sorted(d.season.unique()):
    tr, te = d[d.season != yr], d[d.season == yr]; prep = T.Prep(feats).fit(tr)
    preds[te.index] = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(prep(tr), tr.poe).predict(prep(te))
print(f"college, drafted only n={len(d)} rho={spearmanr(d.poe, preds).statistic:+.3f}")
print("single-feature spearman with poe (drafted, 2010-21):")
for c in T.GROUPS["combine"] + T.GROUPS["college"]:
    x = d[[c, "poe"]].dropna(); print(f"  {c:14s} {spearmanr(x[c], x.poe).statistic:+.3f}  (n={len(x)})")
print("test-era drift: mean poe by class", df.groupby("season").poe.mean().round(3).loc[2016:].to_dict())
