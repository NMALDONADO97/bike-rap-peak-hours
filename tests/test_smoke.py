from pathlib import Path
import subprocess

def test_pipeline_generates_artifacts():
    # run the project command
    subprocess.run(["bash", "run.sh"], check=True)

    # check key artifacts exist
    assert Path("outputs/metrics.txt").exists()
    assert Path("outputs/summary.csv").exists()
    assert Path("outputs/peak_hours.csv").exists()
    assert Path("reports/REPORT.md").exists()
