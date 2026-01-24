# Bike-sharing demand RAP (Python) — Hour-level (Peak Hours)

This repo is a small **Reproducible Analytical Pipeline (RAP)** that:
1) downloads the **UCI Bike Sharing Dataset** (`hour.csv` by default),
2) cleans/prepares the data,
3) analyzes **peak hours** (workingday vs weekend),
4) trains a fast baseline model,
5) writes **outputs** (tables + figures + a short report).

## One-command run
```bash
./run.sh
```

## Outputs (generated)
- `outputs/metrics.txt` (RMSE / R²)
- `outputs/summary.csv` (simple group summary)
- `outputs/peak_hours.csv` (top peak hours: workingday vs weekend)
- `outputs/fig_timeseries.png` (rentals over time)
- `outputs/fig_temp_vs_cnt.png` (temperature vs demand)
- `outputs/fig_avg_by_hour.png` (average rentals by hour)
- `reports/REPORT.md` (short human-readable report)

## Notes
- Default is **hour-level** (`hour.csv`). If you want day-level instead:
  ```bash
  BIKE_LEVEL=day ./run.sh
  ```
- Downloaded data is stored in `data/raw/` (not committed).
