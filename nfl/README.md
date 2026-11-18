# WR Combine → Rookie Fantasy Share

Predicts a wide receiver's **rookie-season share of all WR PPR fantasy points** (in %) from NFL combine results.
The network has two hidden layers of 4 ReLU nodes each. Ten copies are trained with different seeds and their
predictions are averaged.

## Data
Source: [nflverse-data](https://github.com/nflverse/nflverse-data), licensed CC-BY 4.0. The combine file comes
from Pro Football Reference via nflverse; the fantasy points come from NFL play-by-play. Use it for private,
non-commercial work only.

- Population: every combine WR from 2000 to 2025, drafted or not (1,200 players). Anyone with no rookie stats counts as 0.
- Target: rookie PPR points / total PPR points by all WRs in that regular season × 100.
- Inputs: height, weight, 40, bench, vertical, broad, 3-cone, shuttle, plus a "skipped the drill" flag for each.
- Split by draft class: train 2000–16, val 2017–19, test 2020–25.

```bash
python nfl/build_dataset.py   # downloads nflverse files, writes nfl/data/wr_combine_rookie.csv
python nfl/train.py
```

## Results (test set: 2020–25, 295 WRs)
| model | MAE | R² | Spearman |
|---|---|---|---|
| always predict the mean | 0.265 | -0.02 | n/a |
| ridge (linear, same inputs) | 0.248 | 0.07 | 0.43 |
| NN, 10-seed ensemble | 0.238 | 0.07 | 0.45 |
| *draft pick alone (not an input, for context)* | | | *0.79* |

Combine numbers carry a real but weak signal. The network doesn't beat a linear model in any meaningful way.
Draft slot ranks rookies far better, because it builds in college film and production and because it sets
opportunity: high picks get targets.
