"""Arkansas test: TNC burn records (xlsx) -> pipeline input records.

Real records, not simulated: the recorded lat/long is used as-is (no offset), each
burn is a single recorded day, and "Acres Burned" is the reported size. No
perimeters are used; the provider reviews the maps afterwards.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "tester_files" / "BurnsForRandy_ARFO_FY25.xlsx"
REF = ROOT / "data" / "arkansas"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    df = pd.read_excel(XLSX)
    df.columns = [c.strip() for c in df.columns]
    fires = {}
    for _, r in df.iterrows():
        lat, lon = (float(v) for v in str(r["Lat/Long"]).split(","))
        date = pd.Timestamp(r["Date(s) of fire"]).strftime("%Y-%m-%d")
        area = str(r["Managed Area Name"]).strip()
        unit = str(r["Burn Unit Name"]).strip()
        key = "ar_" + slug(area.split("/")[0].replace(" NA", "").replace("-Preserve", ""))[:22] + "_" + slug(unit)[:12]
        fires[key] = {
            "key": key,
            "label": f"{area} - {unit}",
            "managed_area": area,
            "burn_unit": unit,
            "source": "TNC Arkansas burn records (BurnsForRandy_ARFO_FY25.xlsx)",
            "lat": lat,
            "lon": lon,
            "point_offset_m": 0,
            "start": date,
            "end": date,
            "reported_acres": float(r["Acres Burned"]),
            "tnc_acres": None if pd.isna(r.get("TNC acres")) else float(r["TNC acres"]),
            "burn_type": str(r["Type of Burn"]).strip(),
            "burn_boss": str(r["Burn Boss name"]).strip(),
            "ops_number": None if pd.isna(r.get("OPS")) else int(r["OPS"]),
            "region": "Arkansas",
        }
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "fires.json").write_text(json.dumps(fires, indent=1))
    for k, f in fires.items():
        print(k, f["start"], f["reported_acres"], f["lat"], f["lon"])


if __name__ == "__main__":
    main()
