"""Pull VIIRS (S-NPP) 375 m active-fire detections near each fire from the
public NASA FIRMS yearly country archive (no API key needed). Keeps detections
within ~30 km of each simulated point for the year of the burn."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "firms"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_United_States.csv"
BOX_DEG = 0.3


def main():
    fires = json.loads((ROOT / "data" / "reference" / "fires.json").read_text())
    years = sorted({f["start"][:4] for f in fires.values()})
    for y in years:
        keep = {k: [] for k, f in fires.items() if f["start"][:4] == y}
        for chunk in pd.read_csv(URL.format(y=y), chunksize=500_000):
            for k in keep:
                f = fires[k]
                m = ((chunk.latitude - f["lat"]).abs() < BOX_DEG) & ((chunk.longitude - f["lon"]).abs() < BOX_DEG)
                if m.any():
                    keep[k].append(chunk[m])
        for k, parts in keep.items():
            df = pd.concat(parts) if parts else pd.DataFrame()
            df.to_csv(OUT / f"{k}_viirs.csv", index=False)
            print(k, y, len(df), "detections within", BOX_DEG, "deg")


if __name__ == "__main__":
    main()
