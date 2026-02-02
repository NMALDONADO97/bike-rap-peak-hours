# Bike-sharing Demand RAP — Peak Hours Analysis

A **Reproducible Analytical Pipeline (RAP)** that analyzes bike-sharing demand with a focus on **peak hours** (**workdays vs weekends**) using the **UCI Bike Sharing Dataset** (`hour.csv`).

## Start Here

### Option A — Run with Docker 
Go to: **[Quick Start (Docker)](#quick-start-docker)**

### Option B — No Docker available (run locally with Python)
Go to: **[Quick Start (Local Python — no Docker)](#quick-start-local-python--no-docker)**

---

## What this pipeline does

1. **Data Acquisition**: downloads the dataset (`hour.csv`)
2. **Data Processing**: cleans and prepares hourly records
3. **Peak Hours Analysis**: compares workdays vs weekends
4. **Baseline Modeling**: trains a fast baseline model
5. **Output Generation**: exports tables, figures, and a short report
6. **Interactive Dashboard**: launches a Streamlit app for exploration

---

## Reproducibility

- **Python Version**: specified in `.python-version` (pinned to **3.12.7**). Supports Python **3.10–3.14**.
- **Dependencies**: pinned in `requirements.txt`
- **Containerization**: Docker ensures consistent execution across platforms

---

## Quick Start (Docker): 

From the project root directory

### 1) Build the image

```bash
docker build -t bike-rap:latest .
```

### 2) Run the pipeline + dashboard

Choose the command that matches your shell:

**Git Bash (Windows):**
```bash
docker run --rm -p 8501:8501 --mount type=bind,source="$(pwd)",target=/app bike-rap:latest
```

**PowerShell, CMD, WSL, Linux, macOS:**
```bash
docker run --rm -p 8501:8501 -v .:/app bike-rap:latest
```

This will:
- run the complete analytical pipeline
- generate `outputs/`, `reports/`, and `processed_data/`
- launch Streamlit on port **8501**

### 3) Open the dashboard

```text
http://localhost:8501
```

To stop the dashboard: press `Ctrl+C`.

---

## Quick Start (Local Python — no Docker)

From the project root directory

### 1) Run the pipeline

Depending on your system:


```bash
python run.py
```
or

```bash
python3 run.py
```

## Outputs (generated)

Generated files in `outputs/` are prefixed by level (e.g., `hour_` or `day_`):

- `*_metrics.txt` (RMSE / R²)
- `*_summary.csv` (group summary)
- `*_peak_hours.csv` (top peak hours: workingday vs weekend)
- `*_fig_timeseries.png` (rentals over time)
- `*_fig_temp_vs_cnt.png` (temperature vs demand)
- `*_fig_avg_by_hour.png` (average rentals by hour)
- `reports/*_REPORT.md` (short human-readable report)
- **Streamlit Dashboard**: interactive view of all outputs

