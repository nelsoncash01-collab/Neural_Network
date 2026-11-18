"""Predict a WR's rookie share of league-wide WR PPR points from combine results.

Network: 16 inputs -> Linear(4) -> ReLU -> Linear(4) -> ReLU -> Linear(1)
Inputs:  8 combine measures (standardized; missing -> train median) plus
         8 "skipped this drill" flags, since top prospects often skip drills.
Split:   by draft class, so the test set is the future relative to training.
         train 2000-2016, val 2017-2019, test 2020-2025.
Tiny nets are very seed-sensitive, so we train an ensemble of N seeds (same
architecture) and average them. Each member is early-stopped on val.

Usage: python train.py [--seeds 10]
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
DATA = os.path.join(HERE, "data", "wr_combine_rookie.csv")
ART = os.path.join(HERE, "artifacts")
FEATS = ["height_in", "weight_lb", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
TARGET = "rookie_share_pct"
TRAIN_END, VAL_END = 2016, 2019


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 4), nn.ReLU(),
                                 nn.Linear(4, 4), nn.ReLU(),
                                 nn.Linear(4, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Prep:
    """Median-impute + standardize, fit on train only; append missing flags."""

    def fit(self, df):
        self.med = df[FEATS].median()
        filled = df[FEATS].fillna(self.med)
        self.mu, self.sd = filled.mean(), filled.std()
        return self

    def __call__(self, df):
        x = ((df[FEATS].fillna(self.med) - self.mu) / self.sd).to_numpy()
        miss = df[FEATS].isna().to_numpy().astype(float)
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


def main(args):
    df = pd.read_csv(DATA)
    tr = df[df.season <= TRAIN_END]
    va = df[(df.season > TRAIN_END) & (df.season <= VAL_END)]
    te = df[df.season > VAL_END]
    print(f"train {len(tr)} ({tr.season.min()}-{TRAIN_END})  val {len(va)}  test {len(te)} "
          f"({VAL_END + 1}-{te.season.max()})")

    prep = Prep().fit(tr)
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva, yte = (d[TARGET].to_numpy(np.float32) for d in (tr, va, te))

    members, preds = [], []
    for s in range(args.seeds):
        m, vl = fit_one(s, Xtr, ytr, Xva, yva)
        with torch.no_grad():
            preds.append(m(torch.from_numpy(Xte)).numpy())
        members.append(m)
        print(f"  seed {s}: val huber {vl:.4f}")
    p_nn = np.mean(preds, axis=0)
    single = [report(yte, p) for p in preds]

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
    rows = {
        "baseline (train mean)": report(yte, np.full_like(yte, ytr.mean())),
        "ridge (linear, same inputs)": report(yte, ridge.predict(Xte)),
        "NN single seed (median)": {k: float(np.median([r[k] for r in single])) for k in single[0]},
        f"NN ensemble ({args.seeds} seeds)": report(yte, p_nn),
    }
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
        json.dump({"test": rows, "prep": {"median": prep.med.to_dict(), "mean": prep.mu.to_dict(),
                                          "std": prep.sd.to_dict()}}, f, indent=2)
    out.to_csv(os.path.join(ART, "test_predictions.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())
