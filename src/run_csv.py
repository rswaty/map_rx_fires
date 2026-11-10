"""CSV of burns -> the five satellite maps per burn. The one command a partner needs.

    python src/run_csv.py burns.csv
    python src/run_csv.py burns.csv --out outputs_myregion --title "My region prescribed burns"
    python src/run_csv.py burns.csv --redo unit_7      # rerun one burn from scratch

CSV: one row per burn. Column names are not case-sensitive.
  required  lat, lon (decimal degrees, WGS84)   or one "lat/long" column ("34.58, -92.25")
            date        burn date (YYYY-MM-DD is safest; M/D/YYYY also works)
            acres       reported burned acres (sets the search window size)
  optional  end_date    last burn day for multi-day burns (default: same as date)
            id          short unique name; used for file names (default: from site + unit)
            site        preserve / managed area name (map title)
            unit        burn unit name (map title)
            burn_type   shown in the map header
  Any other columns are ignored. People's names (e.g. burn boss) are never read or shown.

Output folder (default outputs_<csv name>/):
  review/        the maps: <id>.pdf (5 pages) and PNGs, summary.pdf, summary.csv, burns.kmz, burns.gpkg
  <id>/          per-burn layers, reused on reruns (delete the folder or use --redo to redo a burn)
  fires.json     the burns as read from the CSV

Burns already processed are skipped, so rerunning after adding rows only processes new ones.
A burn whose CSV row changes is redone automatically. One failed burn does not stop the rest;
the reason is in <id>/error.txt and in summary.csv.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "id": ["id", "key", "burn_id"],
    "site": ["site", "managed_area", "managed area name", "preserve", "name", "site_name"],
    "unit": ["unit", "burn_unit", "burn unit name", "unit_name"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "long", "longitude", "lng", "x"],
    "latlon": ["lat/long", "lat/lon", "latlong", "latlon", "lat_long", "coordinates"],
    "date": ["date", "start", "start_date", "burn_date", "date(s) of fire", "ignition_date"],
    "end_date": ["end_date", "end", "end date"],
    "acres": ["acres", "reported_acres", "acres burned", "acres_burned", "area_acres"],
    "burn_type": ["burn_type", "type of burn", "type"],
}
POST_DAYS = 45  # days of imagery after the burn that the methods look at


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def pick(df, field):
    norm = lambda s: re.sub(r"[\s_]+", "_", str(s).strip().lower())  # noqa: E731
    cols = {norm(c): c for c in df.columns}
    for a in ALIASES[field]:
        if norm(a) in cols:
            return cols[norm(a)]
    return None


def read_burns(path, max_burns=None):
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    df = df.dropna(how="all")
    if max_burns and len(df) > max_burns:
        sys.exit(f"This CSV has {len(df)} burns; the limit here is {max_burns}. "
                 f"Split it into files of {max_burns} rows or fewer.")
    col = {k: pick(df, k) for k in ALIASES}
    problems = []
    if not (col["lat"] and col["lon"]) and not col["latlon"]:
        problems.append("no lat/lon columns (need 'lat' and 'lon', or one 'lat/long' column)")
    if not col["date"]:
        problems.append("no burn date column (need 'date')")
    if not col["acres"]:
        problems.append("no acres column (need 'acres')")
    if problems:
        sys.exit("CSV problem: " + "; ".join(problems) + f"\nColumns found: {list(df.columns)}")

    text = lambda r, k: "" if not col[k] or pd.isna(r[col[k]]) else str(r[col[k]]).strip()  # noqa: E731
    fires, bad = {}, []
    for i, r in df.iterrows():
        row = i + 2  # spreadsheet row number (header is row 1)
        try:
            if col["lat"] and col["lon"]:
                lat, lon = float(r[col["lat"]]), float(r[col["lon"]])
            else:
                lat, lon = (float(v) for v in str(r[col["latlon"]]).split(","))
            start = pd.to_datetime(r[col["date"]])
            end = pd.to_datetime(r[col["end_date"]]) if text(r, "end_date") else start
            acres = float(r[col["acres"]])
        except Exception as e:
            bad.append(f"row {row}: could not read lat/lon, date or acres ({e})")
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            bad.append(f"row {row}: lat/lon out of range ({lat}, {lon})")
            continue
        if 66 <= lon <= 125 and 24 <= lat <= 50:
            print(f"WARNING row {row}: longitude {lon} is positive; US longitudes are negative (west)")
        if acres <= 0 or end < start:
            bad.append(f"row {row}: acres must be > 0 and end date on/after the burn date")
            continue
        site, unit = text(r, "site"), text(r, "unit")
        key = slug(text(r, "id")) or slug(f"{site} {unit}")[:40] or f"burn_{row}"
        base, n = key, 2
        while key in fires:
            key, n = f"{base}_{n}", n + 1
        fires[key] = {"key": key, "label": " - ".join(x for x in (site, unit) if x) or key,
                      "managed_area": site or key, "burn_unit": unit, "lat": lat, "lon": lon,
                      "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}", "reported_acres": acres,
                      "burn_type": text(r, "burn_type"),
                      "source": Path(path).name, "csv_row": row}
    for b in bad:
        print("SKIPPED", b)
    if not fires:
        sys.exit("No usable rows in the CSV.")
    return fires


def same_burn(a, b):
    k = ("lat", "lon", "start", "end", "reported_acres")
    return all(a.get(x) == b.get(x) for x in k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV (or .xlsx) of burns")
    ap.add_argument("--out", help="output folder (default: outputs_<csv name>)")
    ap.add_argument("--title", help="title on the summary page and Google Earth file")
    ap.add_argument("--redo", nargs="*", default=[], metavar="ID", help="burn ids to redo from scratch")
    ap.add_argument("--maps-only", action="store_true", help="only remake maps from saved layers")
    ap.add_argument("--max-burns", type=int, help="refuse CSVs with more rows than this")
    a = ap.parse_args()

    fires = read_burns(a.csv, a.max_burns)
    out = Path(a.out) if a.out else ROOT / f"outputs_{slug(Path(a.csv).stem)}"
    out.mkdir(parents=True, exist_ok=True)
    old_path = out / "fires.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    for k, f in fires.items():
        if k in a.redo or (k in old and not same_burn(old[k], f)):
            shutil.rmtree(out / k, ignore_errors=True)
            print(k, "will be redone")
    old_path.write_text(json.dumps(fires, indent=1))

    import maps
    import methods
    import pipeline_v2 as V

    V.REF, V.OUT, V.CACHE = out, out, out / "cache"
    methods.set_output(out)
    today = pd.Timestamp.now().normalize()
    if not a.maps_only:
        todo = [k for k in fires if not (out / k / "methods.nc").exists()]
        print(f"{len(fires)} burns, {len(todo)} to process (a few minutes each)", flush=True)
        for i, k in enumerate(todo, 1):
            f = fires[k]
            t = time.time()
            if pd.Timestamp(f["end"]) > today:
                print(f"[{i}/{len(todo)}] {k}: burn date is in the future; skipped")
                continue
            try:
                (out / k / "error.txt").unlink(missing_ok=True)
                if not (out / k / "run_log.json").exists():
                    V.run(k)
                methods.run(k)
                print(f"[{i}/{len(todo)}] {k} done in {time.time() - t:.0f} s", flush=True)
            except Exception as e:  # one bad burn should not stop the batch; reason goes in summary.csv
                import traceback
                (out / k).mkdir(parents=True, exist_ok=True)
                msg = str(e).strip() or repr(e)
                (out / k / "error.txt").write_text(f"{e.__class__.__name__}: {msg}\n\n{traceback.format_exc()}")
                for p in ("run_log.json", "methods.nc"):
                    (out / k / p).unlink(missing_ok=True)
                print(f"::error title={k} failed::{e.__class__.__name__}: {msg}", flush=True)
                traceback.print_exc()
            if (today - pd.Timestamp(f["end"])).days < POST_DAYS:
                print(f"    note: burn was < {POST_DAYS} days ago; rerun later with --redo {k} for more after-images")

    title = a.title or f"{Path(a.csv).stem.replace('_', ' ')} prescribed burns"
    maps.main(out, title)
    print(f"\nMaps: {out / 'review'}")


if __name__ == "__main__":
    main()
