"""Generate a FAKE dataset with the expected schema so the pipeline runs end to end.

The numbers are made up. Replace data/synthetic_draft.csv with real data
(e.g. Sports-Reference college stats joined to Basketball-Reference career WS).
"""
import argparse

import numpy as np
import pandas as pd


def main(n: int, seed: int, out: str) -> None:
    rng = np.random.default_rng(seed)
    pos = rng.choice(["G", "F", "C"], size=n, p=[0.42, 0.38, 0.20])
    h_base = {"G": 76.0, "F": 80.0, "C": 83.5}
    w_base = {"G": 195.0, "F": 220.0, "C": 245.0}

    df = pd.DataFrame({
        "player": [f"Player_{i}" for i in range(n)],
        "draft_year": rng.integers(1995, 2024, size=n),
        "age": np.clip(rng.normal(20.8, 1.2, n), 18.5, 24.5).round(1),
        "position": pos,
        "height_in": [rng.normal(h_base[p], 1.8) for p in pos],
        "weight_lb": [rng.normal(w_base[p], 14) for p in pos],
        "efg_pct": np.clip(rng.normal(0.52, 0.045, n), 0.38, 0.70),
        "ft_pct": np.clip(rng.normal(0.72, 0.08, n), 0.40, 0.95),
        "sos": rng.normal(5.0, 3.5, n),
        "orb": np.clip(rng.gamma(2.0, 0.8, n) + (pos == "C") * 1.5, 0, None),
        "drb": np.clip(rng.gamma(3.0, 1.2, n) + (pos == "C") * 2.0, 0, None),
        "ast": np.clip(rng.gamma(2.0, 1.0, n) + (pos == "G") * 2.0, 0, None),
        "stl": np.clip(rng.gamma(3.0, 0.35, n), 0, None),
        "blk": np.clip(rng.gamma(1.2, 0.5, n) + (pos == "C") * 1.0, 0, None),
        "tov": np.clip(rng.gamma(4.0, 0.6, n), 0, None),
        "pts": np.clip(rng.normal(15.5, 4.5, n), 3, None),
    })
    df[["height_in", "weight_lb"]] = df[["height_in", "weight_lb"]].round(1)

    # Latent talent: younger, more efficient, tougher schedule, productive -> better.
    talent = (
        -0.45 * (df.age - 20.8)
        + 8.0 * (df.efg_pct - 0.52)
        + 2.0 * (df.ft_pct - 0.72)
        + 0.06 * (df.sos - 5.0)
        + 0.05 * (df.pts - 15.5)
        + 0.12 * (df.ast + df.stl + df.blk - df.tov)
        + rng.normal(0, 0.9, n)
    )
    max_years = (2025 - df.draft_year).clip(upper=20)
    years = np.clip(np.round(rng.normal(5 + 2.5 * talent, 3)), 1, max_years).astype(int)
    ws_per_year = np.exp(0.6 + 0.55 * talent + rng.normal(0, 0.5, n)) - 1.0
    df["nba_years"] = years
    df["nba_ws"] = (ws_per_year * years).round(1)
    df.to_csv(out, index=False)
    print(f"wrote {n} fake rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic_draft.csv")
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
