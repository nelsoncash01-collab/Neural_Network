"""Add college production + draft capital to the WR dataset.

College data: CollegeFootballData.com API (needs CFBD_API_KEY). Their terms
allow models/predictions built on the data but forbid redistributing the raw
data, so the cached API responses and the merged CSV are git-ignored.

Output: nfl/data/wr_full.csv (one row per combine WR, classes 2010+)
"""
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from build_dataset import norm

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data_raw", "cfbd")
BASE_CSV = os.path.join(HERE, "data", "wr_combine_rookie.csv")
PLAYERS = os.path.join(HERE, "data_raw", "players.csv")
OUT = os.path.join(HERE, "data", "wr_full.csv")
FIRST_CLASS = 2010            # CFBD player stats are sparse before ~2008
YEARS = range(FIRST_CLASS - 4, 2025)
P5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}
# CFBD only has FCS players' games vs FBS opponents (e.g. Kupp's 2016: 206 yds),
# so non-FBS college stats are treated as missing rather than fed in wrong.
FBS = P5 | {"Big East", "American Athletic", "Mountain West", "Conference USA",
            "Mid-American", "Sun Belt", "WAC", "FBS Independents"}


def cfbd(path: str, cache_name: str):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, cache_name)
    if not os.path.exists(f):
        req = urllib.request.Request(
            f"https://api.collegefootballdata.com{path}",
            headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(f, "wb").write(r.read())
        print("fetched", path)
    return json.load(open(f))


def load_college():
    rec = []
    for y in YEARS:
        rows = cfbd(f"/stats/player/season?year={y}&category=receiving", f"rec_{y}.json")
        rec += rows
    rec = pd.DataFrame(rec)
    rec["stat"] = pd.to_numeric(rec.stat, errors="coerce")
    rec = rec.pivot_table(index=["playerId", "player", "team", "conference", "season"],
                          columns="statType", values="stat", aggfunc="sum").reset_index()

    team = []
    for y in YEARS:
        team += cfbd(f"/stats/season?year={y}", f"team_{y}.json")
    team = pd.DataFrame(team)
    team = team[team.statName.isin(["netPassingYards", "passingTDs"])]
    team = team.pivot_table(index=["team", "season"], columns="statName",
                            values="statValue").reset_index()

    rec = rec.merge(team, on=["team", "season"], how="left")
    rec["yds_share"] = rec.YDS / rec.netPassingYards
    rec["td_share"] = rec.TD / rec.passingTDs.replace(0, np.nan)
    rec["dominator"] = rec[["yds_share", "td_share"]].mean(axis=1).clip(upper=1)
    rec["name_key"] = rec.player.map(norm)
    rec["last_key"] = rec.name_key.str.split().str[-1]
    return rec


def school_sim(school: str, team: str) -> float:
    a = set(re.findall(r"[a-z]+", str(school).lower())) - {"university", "of", "state", "the"}
    b = set(re.findall(r"[a-z]+", str(team).lower())) - {"university", "of", "state", "the"}
    return len(a & b) / max(1, len(a | b))


def match(base: pd.DataFrame, rec: pd.DataFrame) -> pd.Series:
    """Map each combine WR to a CFBD playerId (name + last-3-seasons window, school tiebreak)."""
    by_name, by_last = rec.groupby("name_key"), rec.groupby("last_key")
    ids = []
    for _, r in base.iterrows():
        key, last = norm(r.player_name), r.season - 1
        c = by_name.get_group(key) if key in by_name.groups else rec.iloc[:0]
        c = c[c.season.between(last - 2, last)]
        if c.empty:  # nickname fallback (Tutu/Chatarius Atwell): last name + same school
            lk = key.split()[-1] if key else ""
            c = by_last.get_group(lk) if lk in by_last.groups else rec.iloc[:0]
            c = c[c.season.between(last - 2, last)]
            c = c[c.team.map(lambda t: school_sim(r.school, t)) > 0]
            if c.playerId.nunique() != 1:
                ids.append(None)
                continue
        g = c.groupby("playerId").agg(team=("team", "last"), yds=("YDS", "sum"))
        g["sim"] = g.team.map(lambda t: school_sim(r.school, t))
        if len(g) > 1:
            g = g.sort_values(["sim", "yds"], ascending=False)
            if g.sim.iloc[0] == 0:  # ambiguous name and no school overlap: don't guess
                ids.append(None)
                continue
        ids.append(g.index[0])
    return pd.Series(ids, index=base.index)


def college_features(pid, last_season, rec_by_id) -> dict:
    if pid is None or pid not in rec_by_id.groups:
        return {}
    s = rec_by_id.get_group(pid)
    s = s[s.season <= last_season].sort_values("season")
    if s.empty:
        return {}
    f = s.iloc[-1]
    if f.conference not in FBS:
        return {}
    return {
        "c_final_rec": f.REC, "c_final_yds": f.YDS, "c_final_td": f.TD,
        "c_final_ypr": f.YDS / max(f.REC, 1), "c_final_dom": f.dominator,
        "c_best_dom": s.dominator.max(), "c_career_yds": s.YDS.sum(),
        "c_seasons": s.season.nunique(), "c_p5": float(f.conference in P5),
    }


def main():
    base = pd.read_csv(BASE_CSV)
    base = base[base.season >= FIRST_CLASS].reset_index(drop=True)
    rec = load_college()
    base["cfbd_id"] = match(base, rec)
    by_id = rec.groupby("playerId")
    feats = pd.DataFrame([college_features(p, s - 1, by_id)
                          for p, s in zip(base.cfbd_id, base.season)], index=base.index)
    df = pd.concat([base, feats], axis=1)

    # Age on Sept 1 of the rookie season (birth dates from nflverse players.csv).
    players = pd.read_csv(PLAYERS, low_memory=False)
    bd = players.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id").birth_date
    birth = pd.to_datetime(df.gsis_id.map(bd), errors="coerce")
    df["age_rookie"] = (pd.to_datetime(df.season.astype(str) + "-09-01") - birth).dt.days / 365.25

    # Draft capital: log pick, undrafted -> pick 300 + flag.
    df["undrafted"] = df.draft_ovr.isna().astype(float)
    df["log_pick"] = np.log(df.draft_ovr.fillna(300))

    df.to_csv(OUT, index=False)
    print(f"{len(df)} WRs (classes {FIRST_CLASS}+): college stats matched "
          f"{df.c_final_yds.notna().mean():.0%} "
          f"(drafted {df[df.undrafted == 0].c_final_yds.notna().mean():.0%}), "
          f"age known {df.age_rookie.notna().mean():.0%}")
    print("-> " + OUT)


if __name__ == "__main__":
    main()
