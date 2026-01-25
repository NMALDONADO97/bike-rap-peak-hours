import streamlit as st
import pandas as pd
from pathlib import Path
import json

st.set_page_config(page_title="Bike Sharing Peak Hours", layout="wide")

st.title("🚲 Bike Sharing Peak Hours Analysis")
st.markdown("""
This dashboard displays the results of the Reproducible Analytical Pipeline (RAP) 
for the UCI Bike Sharing dataset.
""")

# Sidebar for level selection
level = st.sidebar.selectbox("Select Analysis Level", ["hour", "day"], index=0)

# Check for processed data
processed_dir = Path("data/processed")
processed_path = processed_dir / f"{level}_processed.csv"

if not processed_path.exists():
    st.error(f"No processed data found for level: {level}.")
    st.info(f"Run the pipeline with: `BIKE_LEVEL={level} python run.py` to generate it.")
    st.stop()

df = pd.read_csv(processed_path)

# Main Dashboard
col1, col2 = st.columns(2)

with col1:
    st.header("Peak Hours Summary")
    peak_hours_path = Path(f"outputs/{level}_peak_hours.csv")
    if peak_hours_path.exists():
        peak_df = pd.read_csv(peak_hours_path)
        if "note" in peak_df.columns:
            st.warning(peak_df["note"].iloc[0])
        else:
            st.dataframe(peak_df, use_container_width=True)
    else:
        st.info(f"Peak hours analysis not found for {level}.")

    st.header("General Summary")
    summary_path = Path(f"outputs/{level}_summary.csv")
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df, use_container_width=True)

with col2:
    st.header("Model Performance")
    metrics_path = Path(f"outputs/{level}_metrics.txt")
    if metrics_path.exists():
        st.text_area("Baseline Metrics", metrics_path.read_text(), height=150)
    
    comp_path = Path(f"outputs/{level}_model_comparison.csv")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)

st.divider()
st.header(f"Visualizations ({level})")

# Grid for figures
fig_cols = st.columns(2)

figs = [
    (f"outputs/{level}_fig_heatmap_month_hour.png", "Heatmap: Month x Hour"),
    (f"outputs/{level}_fig_avg_by_hour_daytype.png", "Avg Rentals: Hour (Day Type Split)"),
    (f"outputs/{level}_fig_timeseries.png", "Total Rentals Over Time"),
    (f"outputs/{level}_fig_avg_by_hour.png", "Avg Rentals by Hour"),
    (f"outputs/{level}_fig_temp_vs_cnt.png", "Temperature vs Demand"),
]

for i, (path, title) in enumerate(figs):
    with fig_cols[i % 2]:
        if Path(path).exists():
            st.subheader(title)
            st.image(path)
        else:
            st.info(f"Figure not found: {title}")

st.divider()
st.header("Raw Data Preview")
st.dataframe(df.head(100), use_container_width=True)
