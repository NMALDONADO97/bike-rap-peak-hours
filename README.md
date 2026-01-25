# Bike-sharing demand RAP (Python) — Hour-level (Peak Hours)

This repo is a small **Reproducible Analytical Pipeline (RAP)** that:
1) downloads the **UCI Bike Sharing Dataset** (`hour.csv` by default),
2) cleans/prepares the data,
3) analyzes **peak hours** (workingday vs weekend),
4) trains a fast baseline model,
5) writes **outputs** (tables + figures + a short report).

## Reproducibility
- **Python Version**: Specified in `.python-version` (pinned to 3.12.7). Supports Python 3.10 to 3.13.
- **Dependencies**: Locked in `requirements.lock` for exact environments.
- **Makefile**: Provides a standard interface for reproducibility on Unix systems.

## Quick Start (Unix/macOS/WSL)
If you have `make` installed:
```bash
make
```
This will install dependencies, run the pipeline for both 'hour' and 'day' levels, and launch the dashboard.

## One-command run (All Platforms)
If you don't have `make` or are on Windows:
```bash
python run.py
python3 run.py
```
This script automatically:
1. Detects and uses a compatible Python version (avoids 3.14 alpha).
2. Creates a local virtual environment (`.venv` or `.venv-win`).
3. Installs dependencies from `requirements.lock`.
4. Runs the analysis pipeline for both data levels.
5. Launches the **Streamlit dashboard** to visualize the results.

## Outputs (generated)
Generated files in `outputs/` are prefixed by level (e.g., `hour_` or `day_`):
- `*_metrics.txt` (RMSE / R²)
- `*_summary.csv` (simple group summary)
- `*_peak_hours.csv` (top peak hours: workingday vs weekend)
- `*_fig_timeseries.png` (rentals over time)
- `*_fig_temp_vs_cnt.png` (temperature vs demand)
- `*_fig_avg_by_hour.png` (average rentals by hour)
- `reports/*_REPORT.md` (short human-readable report)
- **Streamlit Dashboard**: A browser-based interactive view of all above.

## Notes
- Downloaded data is stored in `data/raw/` (not committed).
- To stop the dashboard, press `Ctrl+C` in your terminal.
