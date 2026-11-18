"""Build the WR combine -> rookie fantasy-share dataset from nflverse releases.

Data: nflverse-data (CC-BY 4.0), https://github.com/nflverse/nflverse-data
  - combine.csv            (originally from Pro Football Reference)
  - players.csv            (ID crosswalk: pfr_id <-> gsis_id)
  - stats_player_reg_*.csv (regular-season player stats from NFL play-by-play)

Target: rookie_share = rookie-season PPR points / total PPR points by all WRs
        in that regular season, reported in percent.
Population: every WR who attended the combine (drafted or not). A player who
            never recorded a stat as a rookie gets 0. The "rookie season" is the
            combine year.
"""
import os
import re
import urllib.request

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
RAW = os.path.join(os.path.dirname(__file__), "data_raw")
OUT = os.path.join(os.path.dirname(__file__), "data", "wr_combine_rookie.csv")
FIRST, LAST = 2000, 2025  # last *completed* regular season


def fetch(rel: str, name: str) -> str:
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        print("downloading", name)
        urllib.request.urlretrieve(f"{BASE}/{rel}/{name}", path)
    return path


def norm(name: str) -> str:
    n = re.sub(r"[^a-z ]", "", str(name).lower())
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def height_in(ht: str) -> float:
    try:
        ft, inch = str(ht).split("-")
        return int(ft) * 12 + float(inch)
    except ValueError:
        return float("nan")


def main():
    combine = pd.read_csv(fetch("combine", "combine.csv"))
    players = pd.read_csv(fetch("players", "players.csv"), low_memory=False)
    stats = pd.concat(
        [pd.read_csv(fetch("stats_player", f"stats_player_reg_{y}.csv"), low_memory=False)
         for y in range(FIRST, LAST + 1)],
        ignore_index=True,
    )

    wr = combine[(combine.pos == "WR") & combine.season.between(FIRST, LAST)].copy()
    wr["height_in"] = wr.ht.map(height_in)

    # pfr_id -> gsis_id; fall back to name + rookie season for players without pfr_id.
    pid = players.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    wr["gsis_id"] = wr.pfr_id.map(pid)
    fb = players[players.position.isin(["WR", "TE", "RB"]) | players.position.isna()].copy()
    fb["key"] = fb.display_name.map(norm) + "|" + fb.rookie_season.astype("Int64").astype(str)
    fb = fb.drop_duplicates("key", keep=False).set_index("key").gsis_id
    need = wr.gsis_id.isna()
    wr.loc[need, "gsis_id"] = (wr.loc[need, "player_name"].map(norm) + "|"
                               + wr.loc[need, "season"].astype(int).astype(str)).map(fb)

    # Points by player-season, and league WR totals per season.
    pts = stats.groupby(["player_id", "season"]).fantasy_points_ppr.sum()
    wr_total = stats[stats.position == "WR"].groupby("season").fantasy_points_ppr.sum()

    key = list(zip(wr.gsis_id, wr.season))
    wr["rookie_ppr"] = [pts.get(k, 0.0) if isinstance(k[0], str) else 0.0 for k in key]
    wr["season_wr_ppr"] = wr.season.map(wr_total)
    wr["rookie_share_pct"] = 100 * wr.rookie_ppr / wr.season_wr_ppr

    cols = ["season", "player_name", "school", "draft_round", "draft_ovr", "pfr_id", "gsis_id",
            "height_in", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
            "rookie_ppr", "season_wr_ppr", "rookie_share_pct"]
    wr = wr[cols].rename(columns={"wt": "weight_lb"}).sort_values(["season", "player_name"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wr.to_csv(OUT, index=False)

    print(f"{len(wr)} WRs, {wr.gsis_id.notna().mean():.0%} matched to an NFL id, "
          f"{(wr.rookie_ppr > 0).mean():.0%} scored as rookies")
    print(f"drafted & matched: {wr[wr.draft_ovr.notna()].gsis_id.notna().mean():.0%}")
    print("-> " + OUT)


if __name__ == "__main__":
    main()
