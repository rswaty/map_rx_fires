"""Run the round-2 pipeline, unchanged, on the Arkansas records.

    python src/ar_records.py
    python src/run_arkansas.py            # all burns; skips burns already done
    python src/run_arkansas.py <key> ...  # selected burns
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pipeline_v2 as V  # noqa: E402

V.REF = V.ROOT / "data" / "arkansas"
V.OUT = V.ROOT / "outputs_arkansas"

if __name__ == "__main__":
    keys = sys.argv[1:] or list(json.loads((V.REF / "fires.json").read_text()))
    for k in keys:
        if (V.OUT / k / "run_log.json").exists():
            continue
        try:
            V.run(k)
        except Exception as e:  # one bad burn should not stop the batch; it is logged
            (V.OUT / k).mkdir(parents=True, exist_ok=True)
            (V.OUT / k / "error.txt").write_text(repr(e))
            print(k, "FAILED", repr(e), flush=True)
