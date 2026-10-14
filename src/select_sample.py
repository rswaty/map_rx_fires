"""Round 2 test set, chosen automatically (no hand picking):

  * the 3 round-1 fires, plus
  * National Park Service prescribed burns from 2019 (public NPS perimeter service) that
    - were mapped with GPS (differential or autonomous),
    - are 60-1200 ac, lasted <= 4 days, and are not pile burns,
    - at most 1 per park and 5 per NPS region, largest first.

Each fire gets the same kind of simulated input record as round 1: a point 200-400 m
from the unit centre, the recorded dates, and the reported acres.
"""
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from get_reference import NPS, query, simulated_point

ROOT = Path(__file__).resolve().parents[1]
REF1 = ROOT / "data" / "reference"
REF2 = ROOT / "data" / "reference_v2"
REF2.mkdir(parents=True, exist_ok=True)

MIN_AC, MAX_AC, MAX_DAYS = 60, 1200, 4
PER_REGION = 5
SEED = 7


def main():
    fires = json.loads((REF1 / "fires.json").read_text())
    for k in fires:
        for suffix in ("target", "other_burns"):
            shutil.copy(REF1 / f"{k}_{suffix}.geojson", REF2 / f"{k}_{suffix}.geojson")
        fires[k]["region"] = {"konza_k1b": "LTER", "grsm_short_bunk": "SER", "seki_big_stump": "PWR"}[k]

    nps = query(NPS, "FeatureCategory='Treatment' AND FireCalendarYear=2019")
    nps["start"] = pd.to_datetime(nps["FireDiscoveryDateTime"], unit="ms").dt.normalize()
    nps["end"] = pd.to_datetime(nps["ControlDate"], unit="ms").dt.normalize()
    nps["park"] = nps["UNITCODE"].fillna(nps["IncidentName"].str.split().str[0])
    burns = nps.dissolve(by=["IncidentName", "start"], aggfunc="first").reset_index()
    burns["acres"] = burns.to_crs(5070).area / 4046.86
    ok = (burns.MAPMETHOD.str.contains("GPS", na=False)
          & burns.acres.between(MIN_AC, MAX_AC)
          & (burns.end - burns.start).dt.days.between(0, MAX_DAYS)
          & ~burns.IncidentName.str.contains("pile", case=False)
          & (burns.IncidentName != "Cataloochee Short Bunk RX"))
    cand = burns[ok].sort_values("acres", ascending=False).drop_duplicates("park")
    cand = cand.groupby("REGIONCODE", group_keys=False).head(PER_REGION)

    rng = np.random.default_rng(SEED)
    for _, r in cand.iterrows():
        key = "nps_" + r.park + "_" + "".join(c for c in r.IncidentName.lower() if c.isalnum())[:18]
        tgt = gpd.GeoDataFrame([r], geometry="geometry", crs=4326)
        other = nps[~((nps.IncidentName == r.IncidentName) & (nps.start == r.start))]
        tgt.drop(columns=["start", "end"]).to_file(REF2 / f"{key}_target.geojson", driver="GeoJSON")
        other.drop(columns=["start", "end"]).to_file(REF2 / f"{key}_other_burns.geojson", driver="GeoJSON")
        lat, lon, off = simulated_point(tgt, rng)
        fires[key] = dict(key=key, label=f"{r.park} - {r.IncidentName}", region=r.REGIONCODE,
                          source=f"NPS National Wildland Fire Perimeters ({r.MAPMETHOD})",
                          ecosystem="", lat=lat, lon=lon, point_offset_m=round(off),
                          start=str(r.start.date()), end=str(r.end.date()),
                          reported_acres=round(float(r.acres), 1))
    (REF2 / "fires.json").write_text(json.dumps(fires, indent=2))
    print(pd.DataFrame(fires.values())[["key", "region", "start", "end", "reported_acres"]].to_string(index=False))


if __name__ == "__main__":
    main()
