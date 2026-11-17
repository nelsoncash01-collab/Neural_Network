# College → NBA Win Shares Ratio Net

A small feed-forward network (two hidden layers of 4 ReLU nodes each) that predicts a draft prospect's
**age-adjusted win shares proportion** from college stats.

## Target
`ws_ratio = career WS / expected WS for players with the same number of NBA seasons`

Expected WS is the mean career WS of *training-set* players with that tenure (long-tenure buckets with fewer
than 10 players are pooled). 1.0 = average for tenure, 2.0 = double. Players with 0 NBA seasons are dropped,
since the ratio isn't defined for them.

## Inputs (16)
- College: `age` (at start of the pre-draft season), `position` (G/F/C one-hot; "G-F" splits it 50/50),
  `height_in`, `weight_lb`, `efg_pct`, `ft_pct`, `sos`
- Per game: `orb`, `drb`, `ast`, `stl`, `blk`, `tov`, `pts`

Training CSV also needs `nba_years`, `nba_ws`. `player` is optional.

## Run
```bash
pip install -r requirements.txt
python make_synthetic_data.py          # FAKE data, only to test the pipeline
python ws_model.py train --data data/synthetic_draft.csv
python ws_model.py predict --data prospects.csv --out preds.csv
```

Training uses a 70/15/15 split, standardized inputs, Adam, Huber loss, and early stopping on validation.
It prints test metrics next to a baseline that always predicts the mean; that baseline is the bar to beat.
