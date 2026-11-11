# Troubleshooting & Reliability Log

This document tracks every issue uncovered during partner testing (starting with Sarah Byer's northern Sierra Nevada / Independence Lake test burns), why it happened, what was fixed in the code, and what is an inherent satellite remote-sensing limitation.

---

## Issue 1 — Empty report (`ValueError: dimension 'time' already exists as a scalar variable`)

### What happened
- **Test case:** `inputs/from_sarah_byer.csv` (`independence_lake`, `39.452314, -120.290426`, `3/6/2024`, `200 ac`).
- **Symptom:** The GitHub Actions run showed a green checkmark, but the downloaded `burn-maps.zip` contained no per-burn PDF or PNGs — only `summary.pdf`, `summary.csv`, and `burns.kmz`, with `summary.csv` showing:
  `no maps: ValueError("dimension 'time' already exists as a scalar variable")`

### Root cause
1. Independence Lake (~7,000 ft in the Sierra Nevada) had 84–89% snow cover throughout the 45-day post-burn window (March 6 – April 20, 2024).
2. `load_s2()` previously dropped any date where `< 20%` of the search window was snow- and cloud-free (`valid_frac > 0.2`). Because snow covered > 80% of the window on every clear March/April pass, **all post-burn Sentinel-2 dates were discarded**.
3. With 0 post-burn dates, `edge_composite()` fell back to `xr.full_like(da.isel(time=0), np.nan)`. In `xarray`, `.isel(time=0)` leaves `time` attached as a **0-dimensional scalar coordinate** (`time = 2024-01-09`), which was written into `layers.nc`.
4. When `methods.py` loaded `layers.nc` and ran `s2.assign_coords(x=grid.x, y=grid.y)`, `xarray` tried to merge the scalar `time` coordinate on `grid` with the 1-D `time` dimension on `s2` and raised `ValueError("dimension 'time' already exists as a scalar variable")`.
5. `run_csv.py` caught the exception so a batch could continue, and the GitHub Actions workflow still exited `0` (green checkmark).

### Fixes applied
- **No stray scalar `time` coordinate (`src/pipeline.py`, `src/pipeline_v2.py`, `src/methods.py`):**
  - `edge_composite()` now builds an explicit 2-D `(y, x)` `DataArray` when a before or after window has 0 images instead of calling `isel(time=0)`.
  - `pipeline_v2.py` uses `isel(time=0, drop=True)` and strips any non-spatial coordinates before writing `layers.nc`.
  - `methods.py` strips any non-spatial coordinates when reading `layers.nc`.
- **Snow no longer drops an entire scene (`src/pipeline.py`, `src/pipeline_v2.py`, `src/methods.py`):**
  - Scene-level filtering now distinguishes **cloud/nodata** (`< 80%` cloud required) from **snow** (`> 2%` snow-free/cloud-free pixels required to keep a date), while still masking snow pixels individually so snowmelt is not mistaken for burn.
- **Loud, plain-English failures (`src/run_csv.py`, `src/maps.py`, `.github/workflows/maps.yml`):**
  - `summary.pdf` now prints a `FAILED BURNS` block at the top right with the reason for any failed burn.
  - `.github/workflows/maps.yml` writes a Markdown summary table onto the GitHub Actions run page, still uploads any finished maps (`if: always()`), and **fails the workflow run (red X)** if any burn in the CSV did not produce maps.

---

## Issue 2 — Transient Microsoft Planetary Computer outage killed a burn (`STACError`)

### What happened
- **Test case:** `inputs/sarah_byer_tests_2.csv`, row 3 (`burn_3`, `39.45927, -120.282`, `2025-10-21`, `31.8 ac`).
- **Symptom:** `burn_3` produced no maps and failed with:
  `STACError: HREF: 'https://planetarycomputer.microsoft.com/api/stac/v1/' does not resolve to a STAC object`

### Root cause
- Every burn opened the Planetary Computer STAC catalog and ran multiple STAC searches and `odc.stac.load()` calls without any retry logic. A momentary HTTP 502/503 from Planetary Computer aborted `burn_3`.

### Fixes applied
- **Automatic retry with backoff (`src/pipeline.py`, `src/pipeline_v2.py`, `src/methods.py`):**
  - Added `retry()`, `open_stac()`, and `stac_search()` helpers in `src/pipeline.py` (4 attempts with linear backoff) wrapping STAC catalog connection, item searches, and raster loads across Sentinel-2, Landsat, and Copernicus DEM.
  - If all retries still fail, raises a plain-English `RuntimeError` explaining that the satellite archive had a temporary outage and suggesting a rerun.

---

## Issue 3 — Reported acres treated as a ground-truth benchmark

### What happened
- **Symptom:** Confidence was downgraded to `low` whenever mapped acres were `< 20%` of reported acres, and maps displayed `(% of reported)` plus flags like `mapped area < 1/3 of reported` or `mapped area > 2x reported`.

### Root cause
- Reported acres in field records are rough visual or planning estimates — often for an entire burn unit when only part of the unit or surface fuels under canopy actually burned.

### Fixes applied
- **Removed reported-acre ratio from confidence (`src/maps.py`, `README.md`):**
  - `confidence()` now depends only on **imagery timing** (`first_clear_after_days_merged`), **signal strength** (`dnbr_median`), **whether embeddings were used**, and **whether a burn patch was detected** — never on the ratio of mapped to reported acres.
  - Map headers, sidebars, KML descriptions, and summary tables now label reported acres as approximate (`reported ~X ac (approx.)` / `Approx. ac`) and state that reported acres are only used to size the search window.

---

## Issue 4 — Shoreline / water-level changes mapped as burns in Landsat (`burn_2`)

### What happened
- **Test case:** `inputs/sarah_byer_tests_2.csv`, row 2 (`burn_2`, `39.45496, -120.295`, `2025-10-08`, `92.6 ac`).
- **Symptom:** Landsat mapped 62.6 acres right along the shoreline of Independence Lake, while Sentinel-2 (0.8 ac) and VIIRS (0 detections within 1 km) showed no burn.

### Root cause
- Neither `load_s2()` nor `load_landsat()` masked water pixels (`SCL == 6` in Sentinel-2; `qa_pixel` bit 7 in Landsat) or mixed land/water shoreline pixels. Autumn reservoir/lake drawdown between August/September and October changed shoreline reflectance enough to trigger a false NBR drop in 30 m Landsat pixels.

### Fixes applied
- **Water + 2-pixel (40 m) shoreline buffer masked (`src/pipeline.py`, `src/pipeline_v2.py`, `src/methods.py`):**
  - Sentinel-2 now masks `SCL == 6` (water) dilated by 2 pixels (40 m).
  - Landsat 8/9 now masks `qa_pixel` bit 7 (water) dilated by 2 pixels (40 m).

---

## Issue 5 — Same-day Sentinel-2 + Landsat observations & TESSERA year fallback on 2026 burns (`burn_4`, `burn_5`)

### What happened
- **Test cases:** `burn_4` (`2026-10-27`, `11.3 ac`) and `burn_5` (`2026-10-28`, `36.8 ac`).
- **Symptoms:**
  1. Both burns occurred on Oct 27–28, 2026 (only 13 days before the run) and had 12–13 VIIRS active-fire detections confirming fire on Oct 28–29.
  2. Best-effort mapped `2.2 ac` (`burn_4`) and `1.8 ac` (`burn_5`), but the map text printed `Mapped: NOT DETECTED` and `low` confidence (`not detected (< 1 ha mapped)`).
  3. On October 30, 2026 (the first post-burn look), both Sentinel-2 and Landsat 9 passed on the same day, creating duplicate `2026-10-30` timestamps in the merged series.

### Root causes
1. **Hard 1 ha (`2.47 ac`) "NOT DETECTED" cutoff:** `clean()` keeps burn patches down to 5 pixels (`0.20 ha` / `~0.5 ac`), and those polygons were drawn on the map and saved to `burns.gpkg`, but `acres_lines()` and `confidence()` in `src/maps.py` labelled anything under `1 ha` (`2.47 ac`) as `NOT DETECTED`.
2. **Duplicate dates in `break_dates()`:** When Sentinel-2 and Landsat observed on the same calendar day (`2026-10-30`), `xr.concat([s2, ls], "time")` produced two entries with the same timestamp. `break_dates()` treated the second entry as the "next clear look" (`t + 1`) instead of checking persistence into a later date, and cross-sensor differences on that day could reduce the measured drop.
3. **TESSERA embedding year:** For a 2026 burn, `embedding_strata()` only tried `year - 1` (`2025`); if a tile is not yet in `year - 1`, it now falls back cleanly across `(year - 1, year - 2, 2024)`.
4. **Why Sentinel-2 mapped 11.7 ac on `burn_5` while best-effort mapped 1.8 ac:** Inspection of the `sentinel2_only` polygons in `burns.gpkg` showed that 10.9 of the 11.7 Sentinel-2 acres were two patches **900–1,310 m southeast** of the recorded point (caused by cloud/shadow/snow artifacts on Nov 1–4), while only **0.8 ac** was near the recorded point (140 m away). Best-effort's look-alike adjustment removed the distant cloud/snow false positive and mapped **1.8 ac** right at the recorded point.

### Fixes applied
- **Consistent detection threshold (`0.5 ac` / 5 pixels) & decimal formatting (`src/maps.py`):**
  - `acres_lines()` and `confidence()` now use the same 5-pixel (`~0.5 ac`) minimum as `clean()`, so a 1.8–2.2 ac mapped burn is reported as `1.8 ac` / `2.2 ac` (with 1 decimal place for burns `< 10 ac`) instead of `NOT DETECTED`.
  - Burns between `0.5 ac` and `3 ac` get a clear note (`small mapped patch (X.X ac)`) and `medium` confidence rather than `NOT DETECTED`.
  - Burns less than 45 days old are flagged in confidence notes (`burn is N days old (full post-burn window is 45 days)`).
- **Daily averaging on shared dates (`src/pipeline.py`):**
  - `break_dates()` now collapses duplicate dates (`nbr.groupby("time").mean("time")`) so same-day Sentinel-2 + Landsat observations act as one daily look and persistence is always checked against a subsequent day.
- **TESSERA year fallback (`src/pipeline_v2.py`):**
  - `embedding_strata()` tries `(year - 1, year - 2, 2024)` before falling back to a single stratum.

---

## Inherent Satellite Limitations (What Code Cannot Fix & How We Communicate It)

Even when the code works as designed, free 20–30 m optical satellites (Sentinel-2 and Landsat) have physical limits that users need to expect:

1. **Burn size (< 3 acres):**
   - One Sentinel-2 SWIR pixel is `20 m × 20 m` (`0.1 acres`), and the minimum patch filter (`clean()`) requires at least 5 connected pixels (`~0.5 acres`) so single noisy pixels are not mapped as fires. In practice, burns smaller than **~3 acres** (or narrow strip/pile burns) rarely leave a contiguous multi-pixel canopy scar.
2. **Low-intensity / under-canopy burns:**
   - Optical satellites see the **top of the forest canopy**. In closed-canopy conifer or hardwood stands, a cool surface burn that consumes needle litter without scorching tree crowns often produces almost no change in top-of-canopy NBR (`burn_2`, `burn_4`, `burn_5`). VIIRS 375 m active-fire hotspots (`Map 4 of 5`) can still confirm that fire was on the ground during a satellite overpass even when the canopy looks unchanged afterward.
3. **Snow cover after winter/spring burns:**
   - If snow covers the unit for the 45 days after ignition (`independence_lake` on March 6, 2024, at 7,000 ft), optical satellites cannot see the ground until snowmelt.
4. **Very recent burns (< 45 days old):**
   - The method looks up to **45 days after the burn** for clear post-fire images and requires a drop to persist across consecutive clear looks. Running a burn only 1–2 weeks after ignition (`burn_4` and `burn_5` at 13 days old) means only 1–3 post-burn passes exist; if those passes are partly cloudy or smoky, detection will be partial until more post-burn images arrive.
