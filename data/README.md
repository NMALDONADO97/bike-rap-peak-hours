# Data

This project uses the **Bike Sharing Dataset** from the UCI Machine Learning Repository.

- The pipeline downloads a ZIP from UCI and extracts `hour.csv` (default) or `day.csv`.
- Downloaded files are stored in `data/raw/` (not committed to git).
- Processed data (cleaned) is written to `data/processed/`.
