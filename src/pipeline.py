from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from utils import ensure_ucibike_data


def main() -> None:
    # Config
    level = os.environ.get("BIKE_LEVEL", "hour").strip().lower()  # hour | day
    if level not in {"day", "hour"}:
        raise ValueError("BIKE_LEVEL must be 'day' or 'hour'")

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    outputs_dir = Path("outputs")
    reports_dir = Path("reports")
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    csv_path = ensure_ucibike_data(raw_dir=raw_dir, level=level)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 2) Basic cleaning
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"])
        df = df.sort_values("dteday").reset_index(drop=True)

    if "cnt" not in df.columns:
        raise ValueError("Expected target column 'cnt' not found.")

    # Add is_weekend (unique touch)
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    # Save processed
    processed_path = processed_dir / f"{level}_processed.csv"
    df.to_csv(processed_path, index=False)

    # 3) Figures
    if "dteday" in df.columns:
        # Timeseries (hourly: line can be dense, but still OK)
        plt.figure()
        if level == "hour" and "hr" in df.columns:
            # Aggregate to daily to make it readable
            daily = df.groupby("dteday", as_index=False)["cnt"].sum()
            plt.plot(daily["dteday"], daily["cnt"])
            plt.title("Bike rentals over time (hour -> aggregated to day)")
        else:
            plt.plot(df["dteday"], df["cnt"])
            plt.title(f"Bike rentals over time ({level})")
        plt.xlabel("date")
        plt.ylabel("cnt (total rentals)")
        plt.tight_layout()
        plt.savefig(outputs_dir / "fig_timeseries.png", dpi=160)
        plt.close()

    if "temp" in df.columns:
        plt.figure()
        # sample points if very large (hour dataset)
        plot_df = df.sample(n=min(len(df), 5000), random_state=42)
        plt.scatter(plot_df["temp"], plot_df["cnt"], s=8)
        plt.xlabel("temp (normalized)")
        plt.ylabel("cnt (total rentals)")
        plt.title(f"Temperature vs demand ({level})")
        plt.tight_layout()
        plt.savefig(outputs_dir / "fig_temp_vs_cnt.png", dpi=160)
        plt.close()

    # Average rentals by hour (the "cool" part)
    if "hr" in df.columns:
        avg_by_hr = df.groupby("hr", as_index=False)["cnt"].mean().rename(columns={"cnt": "avg_cnt"})
        plt.figure()
        plt.plot(avg_by_hr["hr"], avg_by_hr["avg_cnt"])
        plt.xlabel("hour (hr)")
        plt.ylabel("average cnt")
        plt.title("Average rentals by hour")
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(outputs_dir / "fig_avg_by_hour.png", dpi=160)
        plt.close()

    # 4) Summary tables
    # (a) simple group summary
    group_col = "season" if "season" in df.columns else None
    if group_col:
        summary = (
            df.groupby(group_col)["cnt"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
    else:
        summary = pd.DataFrame({"n": [len(df)], "mean": [df["cnt"].mean()], "median": [df["cnt"].median()], "std": [df["cnt"].std()]})
    summary.to_csv(outputs_dir / "summary.csv", index=False)

    # (b) peak hours table: workingday vs weekend
    peak_path = outputs_dir / "peak_hours.csv"
    if "hr" in df.columns:
        # Define weekend via is_weekend; keep workingday if available
        df["_daytype"] = np.where(df["is_weekend"] == 1, "weekend", "workingday")
        if "workingday" in df.columns:
            # If workingday=0 but not weekend (e.g. holiday), keep as non-workingday
            df.loc[(df["workingday"] == 0) & (df["is_weekend"] == 0), "_daytype"] = "non-workingday"

        peak = (
            df.groupby(["_daytype", "hr"], as_index=False)["cnt"]
            .mean()
            .rename(columns={"cnt": "avg_cnt"})
            .sort_values(["_daytype", "avg_cnt"], ascending=[True, False])
        )

        # Top 5 hours per type
        peak_top = peak.groupby("_daytype", as_index=False).head(5)
        peak_top.to_csv(peak_path, index=False)
        df.drop(columns=["_daytype"], inplace=True, errors="ignore")
    else:
        # fallback
        pd.DataFrame({"note": ["No 'hr' column found; peak-hour analysis skipped."]}).to_csv(peak_path, index=False)

    # 5) Baseline model (fast)
    drop_cols = {"cnt", "casual", "registered", "instant", "dteday"}
    X = df[[c for c in df.columns if c not in drop_cols]]
    y = df["cnt"].astype(float)

    # Identify categorical vs numeric
    # Treat integer-coded columns as categorical (season, yr, mnth, hr, weekday, weathersit, etc.)
    cat_cols = [c for c in X.columns if pd.api.types.is_integer_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=42))])

    # Time-aware split if date exists
    if "dteday" in df.columns:
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
    else:
        # shouldn't happen here, but keep safe
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    (outputs_dir / "metrics.txt").write_text(
        f"Dataset level: {level}\n"
        f"Train size: {len(X_train)}\n"
        f"Test size: {len(X_test)}\n"
        f"Model: Ridge(alpha=1.0)\n"
        f"RMSE: {rmse:.3f}\n"
        f"R2: {r2:.3f}\n"
    )

    # 6) Mini report (markdown)
    report = reports_dir / "REPORT.md"
    report.write_text(
        f"""# Bike-sharing demand — peak hours (auto report)

This report is automatically generated by `src/pipeline.py`.

## Question
Which **hours** are the peak-demand hours, and do they differ between **workingdays** and **weekends**?

## What we did
- Downloaded the **UCI Bike Sharing Dataset** (`{level}.csv`).
- Built a simple `is_weekend` feature from `weekday`.
- Produced a peak-hours table (`outputs/peak_hours.csv`) and a plot (`outputs/fig_avg_by_hour.png`).
- Trained a baseline model (**Ridge regression**) to predict `cnt`.

## Key results
- Metrics: see `outputs/metrics.txt`
- Peak hours: see `outputs/peak_hours.csv`
- Summary table: `outputs/summary.csv`

## Figures
- `outputs/fig_timeseries.png`
- `outputs/fig_avg_by_hour.png`
- `outputs/fig_temp_vs_cnt.png`
"""
    )

    print("Done. Generated files:")
    print(f"- {processed_path}")
    print(f"- {outputs_dir / 'metrics.txt'}")
    print(f"- {outputs_dir / 'summary.csv'}")
    print(f"- {outputs_dir / 'peak_hours.csv'}")
    print(f"- {outputs_dir / 'fig_avg_by_hour.png'}")
    print(f"- {reports_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
