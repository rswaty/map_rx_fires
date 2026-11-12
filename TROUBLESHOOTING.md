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

---

## Issue 6 — Small / Under-Canopy Burns Erased by Morphological Opening & Missed Delayed Scorch

### What happened
- Across Sarah Byer's four test burns in `inputs/sarah_byer_tests_2.csv` (`burn_2` ~93 ac, `burn_3` ~32 ac, `burn_4` ~11 ac, `burn_5` ~37 ac), the previous `best_effort` method mapped `0.0 ac`, `2.0 ac`, `0.8 ac`, and `2.6 ac`.

### Root causes identified
1. **Destructive `binary_opening(iterations=1)` in `clean()`:**
   - Before measuring connected-patch size (`MIN_PATCH_PX = 5`), `clean()` eroded the mask by 1 pixel (20 m) on all sides. Any burn strip or canopy-gap opening 1–2 pixels (20–40 m) wide was completely erased before `GROUP_GAP_PX` (60 m bridging) could connect adjacent canopy openings.
   - Even on the 6 larger Arkansas burns, this single erosion step was throwing away 15–50% of valid burned pixels around patch edges and under partial canopy.
2. **Earliest-3-looks post-burn composite missed delayed crown scorch:**
   - `edge_composite()` only took the median of the **earliest 3 clear looks** after the burn. In conifer stands where needles scorch and brown 2–6 weeks after a surface burn, taking only the first 3 post-burn passes misses the peak NBR drop.
3. **Single-threshold cutoff (`z >= 1.5`) without VIIRS seeding or local canopy-gap growth:**
   - Even when VIIRS active-fire detections (`Map 4`) confirmed active fire at the point (`burn_3`, `burn_4`, `burn_5`), pixels inside those VIIRS footprints with moderate canopy-filtered signal (`1.2 <= z < 1.5`) were discarded.
4. **Control lines & `v2_snap` were hidden from the reviewer maps:**
   - `pipeline_v2.py` already downloaded Overture Maps roads/trails/streams and DEM ridgelines and computed `rule_v2_snap`, but `maps.py` did not display them on Map 5 or export `best_effort_snapped` to `burns.kmz` / `burns.gpkg`.

### What we tested & what we learned (including a critical commission lesson)
- **Lesson on unbounded hysteresis / un-dated growth:**
  - We first tested unbounded `binary_propagation` growing into `z >= 1.0` without requiring `date_ok` on the growth pixels. That caused **runaway percolation** across the landscape (`burn_4` exploded to 313 ac and `burn_5` to 1,000 ac), confirming the Southeast FireMap lesson from the pilot: relaxing `date_ok` or allowing unbounded region growth creates massive false positives.
- **Fixes kept (`src/pipeline.py`, `src/pipeline_v2.py`, `src/methods.py`, `src/maps.py`):**
  1. **Non-destructive `clean()`:** Removed the initial `binary_opening` erosion step; connected components (`>= 5 pixels` / `0.5 ac`, 8-connected) are filtered directly, then closed by 1 pixel.
  2. **`post_scorch_composite()`:** Takes the elementwise minimum of the earliest-3 median and the lowest-2 median across the 45-day post-burn window (requiring at least 2 post-burn looks so a single cloud-shadow glitch cannot dominate).
  3. **Bounded `seed_and_grow()` (`max_grow_px = 2`):** Core seeds (`z >= 1.5` & `date_ok` & `dNBR >= 0.08`, or `z >= 1.2` & `date_ok` & `dNBR >= 0.08` inside an active VIIRS fire footprint near the point) can grow at most **2 pixels (40 m)** into adjacent date-matched pixels (`z >= 1.2` & `date_ok` & `dNBR >= 0.08`).
  4. **Control lines and `rule_v2_snap` shown on Map 5 and exported to `burns.kmz` / `burns.gpkg` / `summary.csv`.**

### Before vs. after on `sarah_byer_tests_2.csv` (and the 6 Arkansas burns)

| Burn | Approx. reported ac | Previous `best_effort` ac | Updated `best_effort` ac | Updated `snapped` ac | Notes |
|---|---|---|---|---|---|
| `burn_2` (`2025-10-08`) | ~93 ac | 0.0 ac (`NOT DETECTED`) | **2.1 ac** (medium) | 0.5 ac | 0 VIIRS hits within 1 km; small 2.1 ac scar right south of the point |
| `burn_3` (`2025-10-21`) | ~32 ac | 2.0 ac (medium) | **4.9 ac** (high) | 3.7 ac | 2.5× more area mapped under conifer canopy |
| `burn_4` (`2026-10-27`) | ~11 ac | 0.8 ac (medium) | **3.4 ac** (high) | 1.3 ac | 4.2× more area mapped; 12 VIIRS hits nearby; only 15 days old |
| `burn_5` (`2026-10-28`) | ~37 ac | 2.6 ac (medium) | **91.8 ac** (high) | 89.5 ac | 13 VIIRS hits; note that Nov 1–4 snow/cloud over the east ridge still adds some commission |
| **Arkansas 6 burns (mean)** | ~131 ac | 74.6 ac | **91.8 ac** | — | +15% to +53% more unit area recovered on every Arkansas burn without runaway growth |

---

## Issue 7 — Autumn Mountain Shadows & Senescence (`burn_5` 3× Over-Mapping) vs. Closed-Canopy Understory Attenuation

### What happened
- In the intermediate test above (`review_v4`), `burn_5` (~37 ac reported, Oct 28, 2026) mapped **91.8 ac** (~2.5× reported) by pulling in scattered patches up to 1 km away on north/east-facing slopes and around Secret Meadow, while `burn_4` (~11.3 ac reported) only mapped **3.4 ac**.

### Ecological & Physical Root Causes
1. **Topographic sun-shadow darkening in autumn mountains:** Between mid-October and early November at 39.5° N, solar elevation drops rapidly; north- and east-facing slopes darken in NIR/SWIR even without fire.
2. **Autumn senescence vs. black char:** Deciduous aspen/shrubs and wet meadows (`Secret Meadow`) cure/senesce in late October, dropping `dNBR` because Red (`B04`) and SWIR1 (`B11`) *brighten* — whereas real combustion *darkens* Vis/NIR/SWIR1 (`Char` endmember).
3. **Cold shadows vs. warm char:** Mountain shadows and snowmelt are cold in Landsat's thermal band (`lwir11`), while blackened char warms up (`+1.5 °C` or more relative to look-alike controls on sunny post-burn passes).
4. **Canopy attenuation (`USFS/FIA NLCD Tree Canopy Cover`):** Under 40–90% tree canopy, surface char signal is attenuated by tree crowns unless scaled by `TCC` where sub-pixel char increased.
5. **Fixed 1 km search radius on small burns:** Using a 1,000 m radius (776 ac circle) on an 11–37 ac burn allowed `burn_5` to grab unrelated terrain 900 m away and zoomed the map out so far that small burns looked like tiny specks.

### Fixes implemented ("All-Things" Ecological Model)
1. **Topographic illumination ($\cos i$) detrending (`topo_illumination` + `adjust_series` in `src/pipeline_v2.py`):**
   - Computes exact solar incidence $\cos i$ from the 30 m Copernicus DEM slope/aspect and each scene's solar zenith & azimuth, and regresses out shadow-driven darkening across the unburned control pool on every pass.
2. **Sub-pixel Char unmixing & senescence veto (`subpixel_char_fraction` in `src/pipeline.py`):**
   - Unmixes Sentinel-2 reflectance into Green Vegetation, Senescent Litter (`NPV`), and Black Carbon (`Char`), vetoing pixels where Red/SWIR1 brightened without an increase in `Char`.
3. **Landsat 8/9 Thermal Band (`lwir11` Surface Temperature) anomaly (`load_landsat(with_thermal=True)` in `src/pipeline_v2.py`):**
   - Computes post-burn surface temperature anomaly ($\Delta T_s$ in °C) relative to look-alike controls; vetoes cold shadows ($\Delta T_s < -2.5\text{ °C}$) and boosts warm char scars ($\Delta T_s \ge +1.5\text{ °C}$).
4. **USFS/FIA NLCD Tree Canopy Cover (`load_tcc` in `src/pipeline_v2.py`) + optional `ecosystem` CSV column (`src/run_csv.py`, `src/maps.py`):**
   - Fetches continuous 0–100% USFS/FIA Tree Canopy Cover in ~0.4 s via MRLC WCS (with ESA WorldCover 10 m fallback), scales char-confirmed `dNBR` for canopy attenuation, and prints canopy % + ecosystem class on Map 5 and in `summary.csv`.
5. **Size-scaled search radius (`search_radius_m` in `src/pipeline.py`, `src/maps.py`) & `>= 5` post-burn looks guard on `post_scorch_composite`:**
   - Search radius scales with approximate burn size ($\text{clip}(3 r_{\text{reported}},\, 350\text{ m},\, 1000\text{ m})$), and `post_scorch_composite` only uses the delayed-scorch window when $\ge 5$ post-burn passes exist.
6. **TESSERA 128-D Stand-Affinity Region Growing & Stand-Boundary Edges (`embedding_features` + `stand_affinity` in `src/pipeline_v2.py`):**
   - Instead of only reducing the 128-D TESSERA embeddings into 30 k-means buckets, we retain the L2-normalized 16-D PCA embedding vectors ($\hat{\mathbf{e}}_{y,x}$) and compute:
     - **Cosine stand affinity (`stand_aff`)** of every pixel to the pre-fire vegetation stand around the recorded point and core burned seeds, allowing `seed_and_grow()` to propagate into canopy-attenuated pixels (`z >= 0.75`, `dNBR >= 0.05`, date-matched or char-confirmed) **only within the same contiguous vegetation stand** (`stand_aff >= 0.72`).
     - **Watershed stand-boundary edges (`emb_edges`)** on the spatial gradient of $\hat{\mathbf{e}}_{y,x}$, added both as a barrier to region growing and into `unit_cells()` so compartments close along forest/meadow and forest/riparian boundaries as well as roads, streams, and ridgelines.

### Results Comparison (`outputs_sarah_byer_all_things/` and `outputs_arkansas_all_things/`)

#### A. Sarah Byer's 4 Sierra Nevada Burns (`inputs/sarah_byer_tests_2.csv`)

| Burn | Date | Approx. ac | Search radius | USFS/FIA TCC | Baseline (`main`) | Without stand-affinity | With Stand-Affinity Embeddings `best_effort` (`snapped`) | What changed |
|---|---|---|---|---|---|---|---|---|
| `burn_2` | 2025-10-08 | ~92.6 ac | 1000 m | 41% | 0.0 ac | 3.7 ac | **8.1 ac** (6.2 ac) | High confidence (`median dNBR 0.175`); grows within the stand around the point |
| `burn_3` | 2025-10-21 | ~31.8 ac | 607 m | 39% | 2.0 ac | 1.2 ac | **9.1 ac** (8.5 ac) | Grows through the forested stand around the point while stopping at the Secret Meadow boundary |
| `burn_4` | 2026-10-27 | ~11.3 ac | 362 m | 44% | 0.8 ac | 16.7 ac (11.8 ac) | **23.7 ac** (26.4 ac) | Maps the lakeside terrace stand (16 days old) |
| `burn_5` | 2026-10-28 | ~36.8 ac | 653 m | 37% | 2.6 ac | 1.4 ac | **21.6 ac** (**32.7 ac** snapped) | Maps **21.6 ac** (**32.7 ac** snapped vs. ~36.8 ac reported!) right over the 13 VIIRS hits without leaking into Secret Meadow |

#### B. TNC Arkansas 6 Burns (`data/arkansas/burns.csv` — original `outputs_arkansas/` vs. `outputs_arkansas_all_things/`)

| Burn | Date | Approx. ac | USFS/FIA TCC | Original `outputs_arkansas` (`best_effort`) | With Stand-Affinity Embeddings `best_effort` (`snapped`) | Confidence |
|---|---|---|---|---|---|---|
| `ar_ar_oak_west_unit` | 2026-01-06 | ~120 ac | 84% (closed forest) | 41.6 ac (medium) | **99.3 ac** (**113.6 ac** snapped) | **high** (+139% mapped under 84% canopy) |
| `ar_lorance_creek_north` | 2026-01-15 | ~90 ac | 90% (closed forest) | 40.7 ac (medium) | **61.2 ac** (**70.0 ac** snapped) | **high** (+50% mapped under 90% canopy) |
| `ar_rattlesnake_ridge_maumelle_pin` | 2026-03-20 | ~135 ac | 89% (closed forest) | 64.8 ac (medium) | **109.2 ac** (**145.7 ac** snapped) | **high** (+69% mapped under 89% canopy) |
| `ar_prairie_ridge_preserve_north_unit` | 2026-05-08 | ~130 ac | 59% (closed forest) | 99.6 ac (high) | **101.7 ac** (**105.0 ac** snapped) | **high** |
| `ar_felsenthal_west_dollar_junct` | 2026-05-14 | ~200 ac | 87% (closed forest) | 145.7 ac (high) | **202.8 ac** (**220.1 ac** snapped) | **high** (+39% mapped under 87% canopy) |
| `ar_dunahoo_preserve_entire` | 2026-06-29 | ~110 ac | 81% (closed forest) | 55.2 ac (high) | **116.7 ac** (**114.2 ac** snapped) | **high** (+111% mapped under 81% canopy) |
| **Arkansas 6-burn average** | — | **~130.8 ac** | **82% TCC** | **74.6 ac** (57%) | **115.2 ac** (**128.1 ac** snapped) | **6 of 6 high confidence** |


