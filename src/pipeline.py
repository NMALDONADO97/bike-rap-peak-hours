"""
Reproducible Analytical Pipeline (RAP) — Bike Sharing Peak Hours

Main script executed by `run.sh`.

It:
- downloads the UCI Bike Sharing dataset (hour/day CSV) if missing
- cleans + engineers a small set of features
- generates tables/figures (outputs/)
- trains baseline models (Ridge + RandomForest comparison)
- writes a short Markdown report (reports/)

Notes:
- Generated artefacts (data/raw, data/processed, outputs, reports) are gitignored.
- Use BIKE_LEVEL=hour (default) or BIKE_LEVEL=day.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils import ensure_ucibike_data


def main() -> None:
    BIKE_LEVEL = os.getenv("BIKE_LEVEL", "hour").strip().lower()
    if BIKE_LEVEL not in {"hour", "day"}:
        print(f"WARNING: BIKE_LEVEL='{BIKE_LEVEL}' not recognized. Falling back to 'hour'.")
        BIKE_LEVEL = "hour"

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    outputs_dir = Path("outputs")
    reports_dir = Path("reports")

    for d in [raw_dir, processed_dir, outputs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 1) Download / extract dataset ---
    csv_path = ensure_ucibike_data(raw_dir=raw_dir, level=BIKE_LEVEL)  # creates data/raw/bike_sharing_dataset.zip + CSV
    df = pd.read_csv(csv_path)

    # --- 2) Minimal cleaning + feature engineering ---
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

    # weekend feature based on weekday (0=Sunday in this dataset)
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # basic checks
    if "cnt" in df.columns:
        df = df.dropna(subset=["cnt"])

    # Save processed table (so the RAP has a clear "data -> processed -> outputs" flow)
    processed_path = processed_dir / f"{BIKE_LEVEL}_processed.csv"
    df.to_csv(processed_path, index=False)

    # --- 3) EDA figures ---
    # 3.1 Heatmap: avg rentals by month x hour
    if {"mnth", "hr", "cnt"}.issubset(df.columns):
        pivot = df.pivot_table(values="cnt", index="mnth", columns="hr", aggfunc="mean")
        plt.figure(figsize=(9, 4))
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="avg cnt")
        plt.xlabel("hour (hr)")
        plt.ylabel("month (mnth)")
        plt.title(f"Heatmap: average rentals by month x hour ({BIKE_LEVEL})")
        plt.xticks(range(0, pivot.shape[1], 2), range(0, 24, 2))
        plt.yticks(range(pivot.shape[0]), pivot.index.tolist())
        plt.tight_layout()
        plt.savefig(outputs_dir / f"{BIKE_LEVEL}_fig_heatmap_month_hour.png", dpi=160)
        plt.close()

    # 3.2 Timeseries: daily total rentals
    if {"dteday", "cnt"}.issubset(df.columns) and df["dteday"].notna().any():
        daily = df.groupby("dteday", as_index=False)["cnt"].sum()
        plt.figure(figsize=(9, 3))
        plt.plot(daily["dteday"], daily["cnt"])
        plt.xlabel("date")
        plt.ylabel("total cnt")
        plt.title(f"Daily rentals over time ({BIKE_LEVEL})")
        plt.tight_layout()
        plt.savefig(outputs_dir / f"{BIKE_LEVEL}_fig_timeseries.png", dpi=160)
        plt.close()

    # 3.3 Scatter: temp vs cnt
    if {"temp", "cnt"}.issubset(df.columns):
        plt.figure(figsize=(5, 4))
        plt.scatter(df["temp"], df["cnt"], s=8, alpha=0.35)
        plt.xlabel("temp (normalized)")
        plt.ylabel("cnt")
        plt.title(f"Temp vs rentals ({BIKE_LEVEL})")
        plt.tight_layout()
        plt.savefig(outputs_dir / f"{BIKE_LEVEL}_fig_temp_vs_cnt.png", dpi=160)
        plt.close()

    # 3.4 Average by hour curve
    if {"hr", "cnt"}.issubset(df.columns):
        avg_by_hr = df.groupby("hr", as_index=False)["cnt"].mean().rename(columns={"cnt": "avg_cnt"})
        plt.figure(figsize=(7, 3))
        plt.plot(avg_by_hr["hr"], avg_by_hr["avg_cnt"])
        plt.xlabel("hour (hr)")
        plt.ylabel("average cnt")
        plt.title(f"Average rentals by hour ({BIKE_LEVEL})")
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(outputs_dir / f"{BIKE_LEVEL}_fig_avg_by_hour.png", dpi=160)
        plt.close()

    # --- 4) Peak hours by day type (table + extra figure) ---
    peak_path = outputs_dir / f"{BIKE_LEVEL}_peak_hours.csv"
    peak_top = pd.DataFrame()

    if "hr" in df.columns:
        df["_daytype"] = np.where(df.get("is_weekend", 0) == 1, "weekend", "workingday")
        if "workingday" in df.columns and "is_weekend" in df.columns:
            df.loc[(df["workingday"] == 0) & (df["is_weekend"] == 0), "_daytype"] = "non-workingday"

        peak = (
            df.groupby(["_daytype", "hr"], as_index=False)["cnt"]
            .mean()
            .rename(columns={"cnt": "avg_cnt"})
            .sort_values(["_daytype", "avg_cnt"], ascending=[True, False])
        )
        peak_top = peak.groupby("_daytype", as_index=False).head(5)
        peak_top.to_csv(peak_path, index=False)

        # curves by day type
        avg_split = (
            df.groupby(["_daytype", "hr"], as_index=False)["cnt"]
            .mean()
            .rename(columns={"cnt": "avg_cnt"})
        )
        plt.figure(figsize=(7, 3))
        for daytype in avg_split["_daytype"].unique():
            tmp = avg_split[avg_split["_daytype"] == daytype]
            plt.plot(tmp["hr"], tmp["avg_cnt"], label=daytype)
        plt.xlabel("hour (hr)")
        plt.ylabel("average cnt")
        plt.title(f"Average rentals by hour split by day type ({BIKE_LEVEL})")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        plt.tight_layout()
        plt.savefig(outputs_dir / f"{BIKE_LEVEL}_fig_avg_by_hour_daytype.png", dpi=160)
        plt.close()

        df.drop(columns=["_daytype"], inplace=True, errors="ignore")
    else:
        peak_top = pd.DataFrame({"note": [f"No 'hr' column found ({BIKE_LEVEL}-level). Peak-hour analysis skipped."]})
        peak_top.to_csv(peak_path, index=False)

    # Human-readable peak-hours text
    summary_txt = outputs_dir / f"{BIKE_LEVEL}_peak_hours_summary.txt"
    if "_daytype" in peak_top.columns and not peak_top.empty:
        out_lines: list[str] = []
        for dt in peak_top["_daytype"].unique():
            tmp = peak_top[peak_top["_daytype"] == dt].sort_values("avg_cnt", ascending=False)
            out_lines.append(f"{dt.upper()}:")
            for _, r in tmp.iterrows():
                out_lines.append(f"  hr={int(r['hr'])}  avg_cnt={r['avg_cnt']:.2f}")
            out_lines.append("")
        summary_txt.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")
    else:
        summary_txt.write_text(f"No peak-hour summary available for {BIKE_LEVEL}-level dataset.\n", encoding="utf-8")

    # --- 5) Summary table (by season if available) ---
    summary_path = outputs_dir / f"{BIKE_LEVEL}_summary.csv"
    if "cnt" in df.columns:
        if "season" in df.columns:
            summary = (
                df.groupby("season")["cnt"]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .rename(columns={"count": "n"})
            )
        else:
            summary = pd.DataFrame(
                {"n": [len(df)], "mean": [df["cnt"].mean()], "median": [df["cnt"].median()], "std": [df["cnt"].std()]}
            )
        summary.to_csv(summary_path, index=False)

    # --- 6) Baseline models (Ridge + RandomForest comparison) ---
    metrics_path = outputs_dir / f"{BIKE_LEVEL}_metrics.txt"
    comp_path = outputs_dir / f"{BIKE_LEVEL}_model_comparison.csv"

    if "cnt" in df.columns:
        drop_cols = {"cnt", "casual", "registered", "instant", "dteday"}
        X = df[[c for c in df.columns if c not in drop_cols]]
        y = df["cnt"].astype(float)

        # heuristic: treat integer columns as categorical
        cat_cols = [c for c in X.columns if pd.api.types.is_integer_dtype(X[c])]
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Time-aware split if date exists
        if "dteday" in df.columns and df["dteday"].notna().any():
            split = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
        else:
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ridge (sparse OK)
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )
        ridge_sparse = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=42))])
        ridge_sparse.fit(X_train, y_train)
        preds = ridge_sparse.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        metrics_path.write_text(
            f"Dataset level: {BIKE_LEVEL}\n"
            f"Train size: {len(X_train)}\n"
            f"Test size: {len(X_test)}\n"
            f"Model: Ridge(alpha=1.0)\n"
            f"RMSE: {rmse:.3f}\n"
            f"R2: {r2:.3f}\n",
            encoding="utf-8",
        )

        # Comparison (dense OHE for RandomForest compatibility)
        pre_dense = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )
        ridge = Pipeline([("pre", pre_dense), ("model", Ridge(alpha=1.0, random_state=42))])
        rf = Pipeline(
            [
                ("pre", pre_dense),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )

        results = []
        for name, model in [("Ridge", ridge), ("RandomForest", rf)]:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results.append(
                {
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "r2": float(r2_score(y_test, pred)),
                }
            )

        pd.DataFrame(results).sort_values("rmse").to_csv(comp_path, index=False)

    # --- 7) Mini report (Markdown) ---
    report_path = reports_dir / f"{BIKE_LEVEL}_REPORT.md"
    report_path.write_text(
        f"""# Bike-sharing demand — peak hours (auto report: {BIKE_LEVEL})

This report is generated by `src/pipeline.py`.

## Question
Which **hours** are the peak-demand hours, and do they differ between **workingdays** and **weekends**?

## What I did 
- Downloaded the UCI Bike Sharing Dataset (`{BIKE_LEVEL}.csv`).
- Built an `is_weekend` feature from `weekday`.
- Produced peak-hours outputs (`outputs/{BIKE_LEVEL}_peak_hours.csv`) and EDA plots (`outputs/{BIKE_LEVEL}_fig_*.png`).
- Trained a baseline model (Ridge regression) and compared it with a RandomForest model.

## Key artefacts
- Environment snapshot: `outputs/{BIKE_LEVEL}_env.json`
- Processed data: `data/processed/{BIKE_LEVEL}_processed.csv`
- Peak hours: `outputs/{BIKE_LEVEL}_peak_hours.csv` + `outputs/{BIKE_LEVEL}_peak_hours_summary.txt`
- Summary table: `outputs/{BIKE_LEVEL}_summary.csv`
- Baseline metrics: `outputs/{BIKE_LEVEL}_metrics.txt`
- Model comparison: `outputs/{BIKE_LEVEL}_model_comparison.csv`
- Figures: `outputs/{BIKE_LEVEL}_fig_*.png`
""",
        encoding="utf-8",
    )

    # --- 8) Quick check / logging ---
    generated = [
        f"outputs/{BIKE_LEVEL}_metrics.txt",
        f"outputs/{BIKE_LEVEL}_model_comparison.csv",
        f"outputs/{BIKE_LEVEL}_summary.csv",
        f"outputs/{BIKE_LEVEL}_peak_hours.csv",
        f"outputs/{BIKE_LEVEL}_peak_hours_summary.txt",
        f"outputs/{BIKE_LEVEL}_fig_heatmap_month_hour.png",
        f"outputs/{BIKE_LEVEL}_fig_timeseries.png",
        f"outputs/{BIKE_LEVEL}_fig_temp_vs_cnt.png",
        f"outputs/{BIKE_LEVEL}_fig_avg_by_hour.png",
        f"outputs/{BIKE_LEVEL}_fig_avg_by_hour_daytype.png",
        f"reports/{BIKE_LEVEL}_REPORT.md",
        f"data/processed/{BIKE_LEVEL}_processed.csv",
    ]
    missing = [p for p in generated if not Path(p).exists()]
    if missing:
        print(f"WARNING: some expected artefacts for '{BIKE_LEVEL}' were not found:")
        for p in missing:
            print(" -", p)
    else:
        print(f"OK ✅ Pipeline executed for level: {BIKE_LEVEL}.")
        print("Check: outputs/, reports/, data/processed/")


if __name__ == "__main__":
    main()
