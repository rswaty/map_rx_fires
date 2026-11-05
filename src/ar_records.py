"""Arkansas test: TNC burn records (xlsx) -> data/arkansas/burns.csv for run_csv.py.

Real records, not simulated: the recorded lat/long is used as-is, each burn is a single
recorded day, and "Acres Burned" is the reported size. No perimeters are used; the
provider reviews the maps afterwards.

    python src/ar_records.py
    python src/run_csv.py data/arkansas/burns.csv --out outputs_arkansas --title "TNC Arkansas prescribed burns"
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "tester_files" / "BurnsForRandy_ARFO_FY25.xlsx"
CSV = ROOT / "data" / "arkansas" / "burns.csv"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    df = pd.read_excel(XLSX)
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        lat, lon = (float(v) for v in str(r["Lat/Long"]).split(","))
        area = str(r["Managed Area Name"]).strip()
        unit = str(r["Burn Unit Name"]).strip()
        rows.append({
            "id": "ar_" + slug(area.split("/")[0].replace(" NA", "").replace("-Preserve", ""))[:22] + "_" + slug(unit)[:12],
            "site": area, "unit": unit, "lat": lat, "lon": lon,
            "date": pd.Timestamp(r["Date(s) of fire"]).strftime("%Y-%m-%d"),
            "acres": float(r["Acres Burned"]),
            "burn_type": str(r["Type of Burn"]).strip(),
        })
    CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
