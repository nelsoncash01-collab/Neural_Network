# WR Prospect → Rookie Fantasy Share

Predicts a wide receiver's **rookie-season share of all WR PPR fantasy points** (in %). The network has two
hidden layers of 4 ReLU nodes each. Ten copies are trained with different seeds and their predictions are averaged.

## Data
| Inputs | Source | Terms |
|---|---|---|
| Combine (ht, wt, 40, bench, vert, broad, cone, shuttle) | [nflverse](https://github.com/nflverse/nflverse-data) (from PFR) | CC-BY 4.0 |
| Rookie PPR points, birth dates, ID crosswalk | nflverse (NFL play-by-play) | CC-BY 4.0 |
| Draft pick | nflverse combine file | CC-BY 4.0 |
| College production | [CollegeFootballData.com](https://collegefootballdata.com) API | Models/predictions allowed; raw data **may not be redistributed**, so `wr_full.csv` and the API cache are git-ignored |

Data provided by CollegeFootballData.com.

- Population: every combine WR in the 2010–2025 classes, drafted or not (757). Anyone with no rookie stats counts as 0.
- Target: rookie PPR points / total PPR points by all WRs that season × 100.
- College features (FBS only): final-season rec/yds/TD/YPR, final and best-season dominator rating
  (average of the player's share of team receiving yards and TDs), career yards, seasons, Power-5 flag, age at rookie season.
  CFBD only has FCS players' games against FBS opponents (Kupp's 2016 shows 206 yards), so FCS stats are treated as missing.
- Draft: log(overall pick), with undrafted = 300, plus an undrafted flag.
- Split by draft class: train 2010–19, val 2020–21, test 2022–25 (194 WRs).

```bash
python nfl/build_dataset.py     # nflverse -> data/wr_combine_rookie.csv
python nfl/build_college.py     # + CFBD (needs CFBD_API_KEY) -> data/wr_full.csv
python nfl/train.py --compare   # every feature set side by side
python nfl/train.py             # full model, saves artifacts/
```

## Results (test: 2022–25)
| inputs | ridge R² | ridge ρ | NN R² | NN ρ |
|---|---|---|---|---|
| combine | 0.05 | 0.35 | 0.02 | 0.31 |
| combine + college | 0.19 | 0.55 | 0.19 | 0.54 |
| combine + draft | 0.51 | 0.74 | 0.46 | 0.72 |
| **draft only** | **0.53** | **0.77** | **0.51** | **0.77** |
| combine + college + draft | 0.49 | 0.75 | 0.43 | 0.71 |

Bootstrap check (`bootstrap_check.py`, 2,000 resamples of the test set):
- Adding combine + college to draft **hurts**: NN R² −0.07, 95% CI [−0.13, −0.02]; ridge −0.04, 95% CI [−0.07, −0.02].
- Correlation between the combine+college model's predictions and what draft pick gets wrong: 0.06, i.e. almost none.
- The NN never beats ridge.

What this means: college production adds real signal over the combine alone (R² goes from 0.02 to 0.19). But NFL
teams already price that information into the draft pick, and the pick also determines opportunity. With about 500
training rows, the extra inputs mostly add noise.
