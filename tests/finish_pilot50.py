"""Finish reporting the already-running small comparison without chat monitoring."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "test-results/final_test_1000"


def main():
    sources = json.loads((ROOT / "data/final_test_1000/prepared_sources.json").read_text(encoding="utf-8"))[:50]
    expected = [WORK / "predictions" / track / model / (row["id"] + ".json")
                for track in ("common_input", "full_article")
                for model in ("LSTM", "Transformer", "BART", "PEGASUS", "Lead-3")
                for row in sources]
    status_path = WORK / "pilot50_completion.json"
    for _ in range(30):
        done = sum(p.exists() for p in expected)
        status_path.write_text(json.dumps({"status": "running", "saved": done, "expected": 500}), encoding="utf-8")
        if done == 500:
            with (WORK / "pilot50_report.log").open("w", encoding="utf-8") as output:
                result = subprocess.run([sys.executable, "tests/report_pilot50.py"], cwd=ROOT,
                                        stdout=output, stderr=subprocess.STDOUT)
            status_path.write_text(json.dumps({"status": "complete" if result.returncode == 0 else "report_error",
                                              "saved": done, "expected": 500, "report_exit_code": result.returncode}), encoding="utf-8")
            return result.returncode
        time.sleep(60)
    status_path.write_text(json.dumps({"status": "incomplete_after_30_minutes", "saved": done,
                                      "expected": 500, "resume": ".\\.venv\\Scripts\\python.exe tests\\final_benchmark.py run --limit 50"}), encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
