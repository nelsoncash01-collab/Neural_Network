"""Predict age-adjusted win-share proportion from college stats.

Target:  ws_ratio = actual career WS / expected WS for a player with the same
         number of NBA seasons. Expected WS is the mean career WS of TRAINING
         players with that many seasons (so val/test never leak into it).
         ws_ratio > 1  -> outperformed peers with equal tenure.

Network: inputs -> Linear(4) -> ReLU -> Linear(4) -> ReLU -> Linear(1)

Usage:
    python ws_model.py train   --data data/synthetic_draft.csv
    python ws_model.py predict --data new_prospects.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC = [
    # college context
    "age", "height_in", "weight_lb", "efg_pct", "ft_pct", "sos",
    # per-game
    "orb", "drb", "ast", "stl", "blk", "tov", "pts",
]
POSITIONS = ["G", "F", "C"]
YEARS_COL, WS_COL = "nba_years", "nba_ws"
MIN_BUCKET = 10  # min players per tenure bucket for a stable expected-WS mean
ART_DIR = "artifacts"


# ---------------------------------------------------------------- features
def encode_position(pos: pd.Series) -> np.ndarray:
    """One-hot G/F/C. Hybrids like 'G-F' split weight across both."""
    out = np.zeros((len(pos), len(POSITIONS)), dtype=np.float32)
    for i, p in enumerate(pos.fillna("").astype(str).str.upper()):
        parts = [x for x in p.replace("/", "-").split("-") if x in POSITIONS]
        for x in parts:
            out[i, POSITIONS.index(x)] = 1.0 / len(parts)
    return out


def build_X(df: pd.DataFrame, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    num = df[NUMERIC].to_numpy(dtype=np.float32)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    return np.hstack([num, encode_position(df["position"])]).astype(np.float32)


# ------------------------------------------------------------------ target
def fit_expected_ws(train: pd.DataFrame) -> dict:
    """Mean career WS per tenure; sparse long-tenure buckets are pooled."""
    yrs = train[YEARS_COL].astype(int)
    counts = yrs.value_counts().sort_index()
    cap = int(counts.index.max())
    while cap > 1 and counts[counts.index >= cap].sum() < MIN_BUCKET:
        cap -= 1
    capped = yrs.clip(upper=cap)
    table = train.groupby(capped)[WS_COL].mean()
    # A tenure bucket with mean <= 0 would make the ratio meaningless/sign-flipped.
    table = table.clip(lower=0.5)
    return {"cap": cap, "table": {int(k): float(v) for k, v in table.items()}}


def expected_ws(years: pd.Series, exp: dict) -> np.ndarray:
    table = exp["table"]
    keys = np.array(sorted(table))
    vals = np.array([table[k] for k in keys])
    y = years.astype(int).clip(lower=keys.min(), upper=exp["cap"]).to_numpy()
    return np.interp(y, keys, vals)  # interpolates any gaps in the table


def ws_ratio(df: pd.DataFrame, exp: dict) -> np.ndarray:
    return (df[WS_COL].to_numpy() / expected_ws(df[YEARS_COL], exp)).astype(np.float32)


# ------------------------------------------------------------------- model
class WSNet(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 4), nn.ReLU(),
            nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(y, p) -> dict:
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.ptp(p) > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "corr": corr}


# ------------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    need = NUMERIC + ["position", YEARS_COL, WS_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    n0 = len(df)
    df = df.dropna(subset=need)
    df = df[df[YEARS_COL] >= 1].reset_index(drop=True)  # 0-year busts have no defined ratio
    print(f"rows: {n0} loaded, {len(df)} usable")

    tr, te = train_test_split(df, test_size=0.15, random_state=args.seed)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85, random_state=args.seed)

    exp = fit_expected_ws(tr)
    scaler = StandardScaler()
    Xtr, Xva, Xte = build_X(tr, scaler, fit=True), build_X(va, scaler), build_X(te, scaler)
    ytr, yva, yte = ws_ratio(tr, exp), ws_ratio(va, exp), ws_ratio(te, exp)

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

    model = WSNet(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Huber: the ratio is heavy-tailed (stars hit 3-5x), MSE would let a few
    # outliers dominate an 8-node network.
    loss_fn = nn.HuberLoss(delta=1.0)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  val_huber {vl:.4f}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (best val_huber {best:.4f})")
            break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pte = model(torch.from_numpy(Xte)).numpy()
    m_model = metrics(yte, pte)
    m_base = metrics(yte, np.full_like(yte, ytr.mean()))  # predict train mean
    print("\nexpected WS by tenure (train):",
          {k: round(v, 1) for k, v in exp["table"].items()})
    print(f"\ntest n={len(yte)}   target mean {yte.mean():.2f}, std {yte.std():.2f}")
    print(f"{'':10s}{'MAE':>8s}{'RMSE':>8s}{'R2':>8s}{'corr':>8s}")
    for name, m in [("baseline", m_base), ("WSNet", m_model)]:
        print(f"{name:10s}{m['MAE']:8.3f}{m['RMSE']:8.3f}{m['R2']:8.3f}{m['corr']:8.3f}")

    os.makedirs(ART_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{ART_DIR}/wsnet.pt")
    with open(f"{ART_DIR}/preprocess.json", "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "expected_ws": exp, "n_in": Xtr.shape[1],
                   "test_metrics": {"model": m_model, "baseline": m_base}}, f, indent=2)
    print(f"\nsaved -> {ART_DIR}/wsnet.pt, {ART_DIR}/preprocess.json")


# ----------------------------------------------------------------- predict
def predict(args):
    with open(f"{ART_DIR}/preprocess.json") as f:
        meta = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["mean"])
    scaler.scale_ = np.array(meta["scale"])
    scaler.n_features_in_ = len(NUMERIC)
    model = WSNet(meta["n_in"])
    model.load_state_dict(torch.load(f"{ART_DIR}/wsnet.pt"))
    model.eval()

    df = pd.read_csv(args.data)
    with torch.no_grad():
        df["pred_ws_ratio"] = model(torch.from_numpy(build_X(df, scaler))).numpy()
    cols = [c for c in ["player", "position", "age"] if c in df.columns] + ["pred_ws_ratio"]
    out = df.sort_values("pred_ws_ratio", ascending=False)[cols]
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a) if a.mode == "train" else predict(a)
