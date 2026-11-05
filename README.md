# Prescribed burn maps from a point, a date and acres

Upload a small CSV of burns (where, when, how big) and get five maps per burn made from free
satellite data. No burn perimeters and no software installs are needed.

1. **Landsat before and after** the burn, with the burned area Landsat finds
2. **Sentinel-2 before and after**, with the burned area Sentinel-2 finds
3. **Sentinel-2 + Landsat combined** (more clear looks)
4. **VIIRS active-fire detections** (NASA FIRMS hot spots)
5. **Best effort**: combined, plus satellite embeddings to remove seasonal green-up, with burn severity

Each map shows the recorded point, a 1 km search circle, mapped vs. reported acres and a
confidence rating. Results are satellite estimates and are not field-verified.

## Make maps (in your browser)

You need to be a collaborator on this repository.

1. **Make a CSV**, one row per burn, **5 burns at most**. Start from [`template.csv`](template.csv):

   | Column | Required | Example | Notes |
   |---|---|---|---|
   | `lat`, `lon` | yes | `34.5851`, `-92.2494` | decimal degrees; US longitudes are negative |
   | `date` | yes | `2026-01-15` | burn date (`1/15/2026` also works) |
   | `acres` | yes | `90` | reported acres; sets how far around the point to look |
   | `end_date` | no | `2026-01-16` | for multi-day burns |
   | `id` | no | `north_unit` | short unique name used for file names |
   | `site`, `unit` | no | `Example Preserve`, `North unit` | map title |
   | `burn_type` | no | `Broadcast` | shown in the map header |

   Other columns are ignored. Do not include people's names; they are never used.
   Give the file a name that says what it is, e.g. `ozarks_spring_2026.csv`.

2. **Upload it** into the [`inputs`](inputs) folder: open the folder, click
   **Add file → Upload files**, add the CSV, choose **Create a new branch**, and commit.

3. **Wait.** The upload starts a run automatically; follow it in the **Actions** tab
   (about 4 minutes per burn).

4. **Download.** Open the finished run and download **burn-maps** (a zip) at the bottom.
   Downloads are kept for 90 days. To redo a run, edit and re-upload the CSV.

The zip has a folder per CSV with:

- `<id>.pdf`: the five maps for one burn (each page also as a PNG)
- `summary.pdf`, `summary.csv`: all burns, acres by method, confidence
- `burns.kmz`: open in Google Earth
- `burns.gpkg`: GIS layers (QGIS/ArcGIS), one layer per method

If a run fails, open it in **Actions** and read the red step: it names the problem
(e.g. more than 5 rows, a missing column, a date it could not read).

## Good to know

- **Recent burns:** the methods use up to 45 days of images after the burn. A burn less than
  45 days old gets a better map if you run it again later.
- **VIIRS for the current year** needs a free NASA FIRMS key saved by the repository owner as
  the Actions secret `FIRMS_MAP_KEY` (<https://firms.modaps.eosdis.nasa.gov/api>). Burns from
  past years need no key.
- **Where it works:** the lower 48 US states, from about 2017 on.
- Mapped acres often differ from reported acres: light or patchy fire under trees is hard to
  see from space, and neighbouring burns from the same weeks can be included.

## Run on your own computer (optional, no burn limit)

Needs Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_csv.py my_burns.csv  # maps in outputs_my_burns/review/
```

Burns already done are skipped on reruns; `--redo <id>` redoes one burn, `--maps-only`
redraws the maps, `FIRMS_MAP_KEY=<key>` adds current-year VIIRS.

## Code

| File | What it does |
|---|---|
| `.github/workflows/maps.yml` | the browser workflow: runs on CSVs uploaded to `inputs/` |
| `src/run_csv.py` | reads the CSV and runs everything below |
| `src/pipeline_v2.py` | downloads Sentinel-2 + Landsat, Sentinel-2-only and best-effort burned areas |
| `src/methods.py` | Landsat-only and combined burned areas, Landsat before/after images |
| `src/maps.py` | the five maps, summary, Google Earth and GIS files |
| `src/pipeline.py`, `src/tessera.py` | shared helpers: Sentinel-2 loading, embeddings |
| `src/ar_records.py` | turns the TNC Arkansas spreadsheet into `data/arkansas/burns.csv` |
