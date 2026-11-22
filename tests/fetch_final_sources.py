"""Retrieve a fixed random official test sample; no generation or model changes."""
import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/final_test_1000"
DATASET = "abisee/cnn_dailymail"
REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
SELECTED = random.Random(42).sample(range(11490), 1000)


def fetch(offset):
    path = DEST / "blocks" / f"{offset:05}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = ("https://datasets-server.huggingface.co/rows?dataset=abisee%2Fcnn_dailymail"
           f"&config=3.0.0&split=test&offset={offset}&length=100")
    for attempt in range(5):
        try:
            with urlopen(url, timeout=60) as response:
                data = json.load(response)
            assert data["num_rows_total"] == 11490
            data["retrieved_from"] = url
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    (DEST / "blocks").mkdir(parents=True, exist_ok=True)
    selected = set(SELECTED)
    found = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, block in enumerate(pool.map(fetch, sorted({r // 100 * 100 for r in selected}))):
            for entry in block["rows"]:
                idx, row = entry["row_idx"], entry["row"]
                if idx in selected:
                    found[idx] = dict(row_idx=idx, id=row["id"], article=row["article"],
                                      reference=row["highlights"], retrieved_from=block["retrieved_from"])
            if i % 10 == 0:
                print(f"Fetched {i + 1} blocks; {len(found)}/1000 selected sources", flush=True)
    assert set(found) == selected
    rows = [found[i] for i in SELECTED]
    assert len({r["id"] for r in rows}) == 1000
    for row in rows:
        for field in ("article", "reference"):
            row[field + "_sha256"] = hashlib.sha256(row[field].encode()).hexdigest()
    (DEST / "sources.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved all 1000 fixed test sources; no predictions generated.", flush=True)


if __name__ == "__main__":
    main()
