# Prescribed burn maps from a point, a date and acres

Upload a small CSV of burns (where, when, how big) and get five maps per burn made from free
satellite data. No burn perimeters and no software installs are needed.

1. **Landsat before and after** the burn, with the burned area Landsat finds
2. **Sentinel-2 before and after**, with the burned area Sentinel-2 finds
3. **Sentinel-2 + Landsat combined** (more clear looks)
4. **VIIRS active-fire detections** (NASA FIRMS hot spots)
5. **Best effort**: combined, plus satellite embeddings to remove seasonal green-up, with burn severity

Each map shows the recorded point, a 1 km search circle, the mapped burned area (alongside
the approximate reported acres), and a confidence rating based on imagery timing and signal
strength. Results are satellite estimates and are not field-verified.

## Make maps (in your browser)

You need to be a collaborator on this repository.

1. **Make a CSV**, one row per burn, **5 burns at most**. Start from [`template.csv`](template.csv):

   | Column | Required | Example | Notes |
   |---|---|---|---|
   | `lat`, `lon` | yes | `34.5851`, `-92.2494` | decimal degrees; US longitudes are negative |
   | `date` | yes | `2026-01-15` | burn date (`1/15/2026` also works) |
   | `acres` | yes | `90` | approximate acres; only used to set how far around the point to look |
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

## Good to know (what satellites can and cannot see)

- **Burn size (best above ~3 acres):** Sentinel-2 pixels are 20 m × 20 m (~0.1 acre) and the
  mapper requires at least 5 connected pixels (~0.5 acre) so single noisy pixels are not mapped
  as fires. Burns under **3 acres**, pile burns, or narrow strip burns are often too small to
  form a clear multi-pixel scar.
- **Closed canopy and low-intensity surface burns:** optical satellites see the top of the
  canopy. A cool understory burn under dense conifer or hardwood canopy often leaves little or
  no change in Sentinel-2 / Landsat reflectance, even when VIIRS active-fire hot spots (`Map 4`)
  confirm fire was on the ground.
- **Snow cover:** if snow covers the unit during the 45 days after the burn (common for winter
  or early-spring mountain burns), the ground is hidden in the "After" image (shown as flat
  grey) and the burn scar cannot be mapped until snowmelt.
- **Recent burns (wait ~2–6 weeks when possible):** the methods use up to 45 days of images
  after the burn and check that the change persists across consecutive clear passes. A burn run
  within a couple of weeks of ignition may only have 1–3 post-burn passes (often cloudy or
  smoky); rerunning it a few weeks later usually gives a much better map.
- **Approximate acres:** reported acres are only used to size the search window, never as a
  benchmark. Confidence is rated from how soon a clear image was captured after the burn and
  how distinct the burn signal is. Neighbouring burns from the same weeks can also be included.
- **Fill in `id` or `site` and `unit`:** if those columns are blank, burns are named `burn_2`,
  `burn_3`, etc. after their spreadsheet row number.
- **VIIRS for the current year** needs a free NASA FIRMS key saved by the repository owner as
  the Actions secret `FIRMS_MAP_KEY` (<https://firms.modaps.eosdis.nasa.gov/api>). Burns from
  past years need no key.
- **Where it works:** the lower 48 US states, from about 2017 on.
- **Troubleshooting log:** see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for a record of edge
  cases found during testing and how they were fixed.

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
