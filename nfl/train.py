"""Predict a WR's rookie share of league-wide WR PPR points.

Network: inputs -> Linear(4) -> ReLU -> Linear(4) -> ReLU -> Linear(1)
Inputs (by feature set, see --features):
  combine: 8 measures (standardized; missing -> train median + missing flag;
           top prospects often skip drills, so skipping is informative)
  college: final/best-season production, dominator rating, career yards,
           seasons, Power-5, age at rookie season (CFBD; FBS only)
  draft:   log(overall pick) with undrafted = 300, plus an undrafted flag
Split:   by draft class, so the test set is the future relative to training.
         Classes 2010-2025: train 2010-2019, val 2020-2021, test 2022-2025.
Tiny nets are very seed-sensitive, so we train an ensemble of N seeds (same
architecture) and average them. Each member is early-stopped on val.

Usage: python train.py [--seeds 10] [--features combine,college,draft]
       python train.py --compare     # every feature set, side by side
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "wr_full.csv")
ART = os.path.join(HERE, "artifacts")
GROUPS = {
    "combine": ["height_in", "weight_lb", "forty", "bench", "vertical", "broad_jump",
                "cone", "shuttle"],
    "college": ["c_final_rec", "c_final_yds", "c_final_td", "c_final_ypr", "c_final_dom",
                "c_best_dom", "c_career_yds", "c_seasons", "c_p5", "age_rookie"],
    "draft": ["log_pick", "undrafted"],
}
TARGET = "rookie_share_pct"
TRAIN_END, VAL_END = 2019, 2021


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 4), nn.ReLU(),
                                 nn.Linear(4, 4), nn.ReLU(),
                                 nn.Linear(4, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Prep:
    """Median-impute + standardize (fit on train only) and append missing flags.
    Columns that always go missing together (all college stats) share one flag."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, df):
        self.med = df[self.feats].median()
        filled = df[self.feats].fillna(self.med)
        self.mu, self.sd = filled.mean(), filled.std().replace(0, 1)
        miss = df[self.feats].isna()
        flags = {}
        for c in self.feats:
            if miss[c].any():
                flags.setdefault(tuple(miss[c]), c)  # dedupe identical patterns
        self.flag_cols = list(flags.values())
        return self

    def __call__(self, df):
        x = ((df[self.feats].fillna(self.med) - self.mu) / self.sd).to_numpy()
        miss = df[self.flag_cols].isna().to_numpy().astype(float)
        return np.hstack([x, miss]).astype(np.float32)


def fit_one(seed, Xtr, ytr, Xva, yva, epochs=2000, patience=100, lr=3e-3, wd=1e-3, bs=32):
    torch.manual_seed(seed)
    Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
    m = Net(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(delta=0.5)  # target is skewed: half zeros, long right tail
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(m(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = loss_fn(m(Xva), yva).item()
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m.eval(), best


def report(y, p):
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = spearmanr(y, p).statistic if np.ptp(p) > 0 else float("nan")
    return {"MAE": float(np.mean(np.abs(y - p))), "R2": float(r2), "spearman": float(rho)}


def run(df, groups, seeds, verbose=True):
    feats = [c for g in groups for c in GROUPS[g]]
    tr = df[df.season <= TRAIN_END]
    va = df[(df.season > TRAIN_END) & (df.season <= VAL_END)]
    te = df[df.season > VAL_END]
    prep = Prep(feats).fit(tr)
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva, yte = (d[TARGET].to_numpy(np.float32) for d in (tr, va, te))

    if verbose:
        print(f"features: {'+'.join(groups)} -> {Xtr.shape[1]} inputs")
        print(f"train {len(tr)} ({tr.season.min()}-{TRAIN_END})  val {len(va)}  "
              f"test {len(te)} ({VAL_END + 1}-{te.season.max()})")
    members, preds = [], []
    for s in range(seeds):
        m, vl = fit_one(s, Xtr, ytr, Xva, yva)
        with torch.no_grad():
            preds.append(m(torch.from_numpy(Xte)).numpy())
        members.append(m)
    p_nn = np.mean(preds, axis=0)
    single = [report(yte, p) for p in preds]

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
    rows = {
        "baseline (train mean)": report(yte, np.full_like(yte, ytr.mean())),
        "ridge (linear, same inputs)": report(yte, ridge.predict(Xte)),
        "NN single seed (median)": {k: float(np.median([r[k] for r in single])) for k in single[0]},
        f"NN ensemble ({seeds} seeds)": report(yte, p_nn),
    }
    if not verbose:
        return rows
    print(f"\nTEST ({len(te)} WRs, target mean {yte.mean():.3f}%, std {yte.std():.3f}%)")
    print(f"{'':32s}{'MAE':>8s}{'R2':>8s}{'spearman':>10s}")
    for k, r in rows.items():
        print(f"{k:32s}{r['MAE']:8.3f}{r['R2']:8.3f}{r['spearman']:10.3f}")

    # Context only (NOT a model input): how much does draft slot alone rank rookies?
    pick = te.draft_ovr.fillna(300).to_numpy()
    print(f"\n[context] draft pick alone, spearman on test: {spearmanr(yte, -pick).statistic:.3f}")

    out = te[["season", "player_name", "draft_ovr", TARGET]].assign(pred=p_nn)
    print("\nTop 15 test-set predictions:")
    print(out.nlargest(15, "pred").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    os.makedirs(ART, exist_ok=True)
    for i, m in enumerate(members):
        torch.save(m.state_dict(), os.path.join(ART, f"net_seed{i}.pt"))
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump({"features": groups, "test": rows,
                   "prep": {"median": prep.med.to_dict(), "mean": prep.mu.to_dict(),
                            "std": prep.sd.to_dict(), "flags": prep.flag_cols}}, f, indent=2)
    out.to_csv(os.path.join(ART, "test_predictions.csv"), index=False)
    return rows


def main(args):
    df = pd.read_csv(DATA)
    if not args.compare:
        run(df, args.features.split(","), args.seeds)
        return
    sets = [["combine"], ["combine", "college"], ["combine", "draft"], ["draft"],
            ["combine", "college", "draft"]]
    print(f"{'features':26s}{'ridge R2':>9s}{'rho':>7s}{'NN R2':>8s}{'rho':>7s}{'MAE':>7s}")
    for g in sets:
        r = run(df, g, args.seeds, verbose=False)
        rd, nn_ = r["ridge (linear, same inputs)"], r[f"NN ensemble ({args.seeds} seeds)"]
        print(f"{'+'.join(g):26s}{rd['R2']:9.3f}{rd['spearman']:7.3f}"
              f"{nn_['R2']:8.3f}{nn_['spearman']:7.3f}{nn_['MAE']:7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--features", default="combine,college,draft")
    ap.add_argument("--compare", action="store_true")
    main(ap.parse_args())
