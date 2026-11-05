"""Point + date -> burn polygon pipeline (free data only).

Input per fire (data/reference/fires.json): approximate lat/lon, recorded start
and end dates, reported acres. The reference perimeter is NOT used here; it is
only used later in validate.py.

Evidence layers (all on a common 20 m UTM grid):
  * Sentinel-2 L2A: pre/post NBR composites (dNBR) + per-pixel break date from
    the full time series ("did this pixel change when the record says it burned?")
  * Sentinel-1 RTC: VH/VV backscatter log-ratio post/pre (cloud/smoke-proof)
  * TESSERA annual embeddings: cosine change between the burn year and year before
  * VIIRS 375 m active fire detections (FIRMS archive) near the recorded dates

Mapping rules (to expose the omission/commission tradeoff):
  A  wall-to-wall  : dNBR > 0.10 anywhere in the window (classic low-severity threshold)
  B  point-guided  : locally adaptive dNBR threshold, kept only if connected to the point
  C  B + timing    : additionally require the break date to match the recorded burn dates
  D  C + embedding : C, plus weaker-dNBR pixels that are date-matched AND show embedding change
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
from pyproj import CRS, Transformer
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).parent))
import tessera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
OUT = ROOT / "outputs"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RES = 20  # m; native resolution of the S2 SWIR bands

PRE_DAYS, POST_DAYS = 60, 45
DATE_TOL_DAYS = 2          # slack on recorded dates
POINT_RADIUS_M = 1000      # how far the true burn may be from the recorded point
MIN_PATCH_PX = 5           # 0.2 ha
GROUP_GAP_PX = 3           # patches within 60 m are treated as one burn
SCL_INVALID = [0, 1, 8, 9, 10, 11]  # nodata, saturated, cloud med/high, cirrus, snow
SCL_SHADOW = 3


# ----------------------------------------------------------------------------- grid
def utm_epsg(lat, lon):
    return CRS.from_dict({"proj": "utm", "zone": int((lon + 180) // 6) + 1,
                          "south": lat < 0}).to_epsg()


def make_window(f):
    """Square search window scaled to reported size (min 5 km across)."""
    epsg = utm_epsg(f["lat"], f["lon"])
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    x, y = to_utm.transform(f["lon"], f["lat"])
    half = max(2500.0, 2.5 * np.sqrt(f["reported_acres"] * 4046.86))
    half = np.ceil(half / RES) * RES
    bb_utm = (x - half, y - half, x + half, y + half)
    lons, lats = to_ll.transform([bb_utm[0], bb_utm[2], bb_utm[0], bb_utm[2]],
                                 [bb_utm[1], bb_utm[1], bb_utm[3], bb_utm[3]])
    bb_ll = (min(lons), min(lats), max(lons), max(lats))
    return epsg, (x, y), bb_utm, bb_ll


# ----------------------------------------------------------------------------- S2
def _readable(item):
    """Some archived scenes have files without georeferencing, which crash the whole load."""
    try:
        with rasterio.open(item.assets["SCL"].href) as src:
            ok = src.crs is not None
    except Exception:
        ok = False
    if not ok:
        print("skipping unreadable scene", item.id, flush=True)
    return ok


def load_s2(cat, f, epsg, bb_ll, t0, t1):
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bb_ll,
                       datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                       query={"eo:cloud_cover": {"lt": 95}}).item_collection()
    items = [it for it in items if _readable(it)]
    baseline = {}
    for it in items:
        d = pd.Timestamp(it.datetime).tz_localize(None).normalize()
        baseline[d] = float(it.properties.get("s2:processing_baseline", "0") or 0)
    ds = odc.stac.load(items, bands=["B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL"],
                       crs=f"EPSG:{epsg}", resolution=RES, bbox=bb_ll,
                       groupby="solar_day", chunks={}, resampling={"SCL": "nearest", "*": "bilinear"})
    ds = ds.compute()
    ds["time"] = pd.DatetimeIndex(ds.time.values).normalize()

    scl = ds["SCL"]
    invalid = scl.isin(SCL_INVALID) | (scl == 0)
    # Sen2Cor often labels fresh char as "cloud shadow"; only trust the shadow
    # class when an actual cloud is nearby in the same scene.
    cloud = scl.isin([8, 9, 10]).values
    near_cloud = np.stack([ndi.binary_dilation(c, iterations=25) for c in cloud])
    invalid = invalid | ((scl == SCL_SHADOW) & near_cloud)

    refl = {}
    for b in ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]:
        off = xr.DataArray([1000.0 if baseline.get(t, 0) >= 4.0 else 0.0
                            for t in pd.DatetimeIndex(ds.time.values)], dims="time",
                           coords={"time": ds.time})
        v = (ds[b].astype("float32") - off) / 10000.0
        refl[b] = v.where(~invalid & (ds[b] > 0))
    s2 = xr.Dataset(refl)
    s2["NBR"] = (s2.B8A - s2.B12) / (s2.B8A + s2.B12)
    s2["NDVI"] = (s2.B08 - s2.B04) / (s2.B08 + s2.B04)
    valid_frac = s2.NBR.notnull().mean(("x", "y"))
    s2 = s2.sel(time=valid_frac.values > 0.2)
    return s2


def edge_composite(da, times, mask_time, n=3, from_end=False):
    """Per-pixel median of the n valid observations closest to the burn."""
    sub = da.sel(time=mask_time)
    if sub.sizes["time"] == 0:
        return xr.full_like(da.isel(time=0), np.nan), 0
    arr = sub.values
    if from_end:
        arr = arr[::-1]
    valid = ~np.isnan(arr)
    rank = np.cumsum(valid, axis=0)
    arr = np.where(valid & (rank <= n), arr, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmedian(arr, axis=0)
    return xr.DataArray(out, dims=("y", "x"), coords={"y": da.y, "x": da.x}), sub.sizes["time"]


def break_dates(nbr):
    """Per-pixel largest *persistent* NBR drop between consecutive clear looks.

    drop_t = NBR(previous clear look) - NBR(t), required to persist into the next
    clear look so a single cloud-shadow artefact does not register as a burn.
    Returns magnitude and the (last-clear-before, first-clear-after) date bracket.
    """
    t = nbr.time.values.astype("datetime64[D]").astype("int64").astype("float64")
    tt = xr.DataArray(np.broadcast_to(t[:, None, None], nbr.shape), dims=nbr.dims, coords=nbr.coords)
    tt = tt.where(nbr.notnull())
    prev = nbr.shift(time=1).ffill("time")
    prev_t = tt.shift(time=1).ffill("time")
    nxt = nbr.shift(time=-1).bfill("time")
    drop = prev - nbr
    persist = xr.where(nxt.notnull(), np.minimum(drop, prev - nxt), drop)
    persist = persist.where(nbr.notnull() & prev.notnull())
    filled = persist.fillna(-9)
    k = filled.argmax("time")
    mag = persist.isel(time=k).where(filled.max("time") > -9)
    hi = tt.isel(time=k).where(mag.notnull())
    lo = prev_t.isel(time=k).where(mag.notnull())
    return mag, lo, hi


# ----------------------------------------------------------------------------- S1
def load_s1(cat, epsg, bb_ll, geobox, start, end):
    t0, t1 = start - pd.Timedelta(days=40), end + pd.Timedelta(days=30)
    items = cat.search(collections=["sentinel-1-rtc"], bbox=bb_ll,
                       datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}").item_collection()
    if len(items) == 0:
        return None, 0, 0
    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=geobox, groupby="solar_day",
                       chunks={}, resampling="average").compute()
    ds = ds.where(ds > 0)
    tm = pd.DatetimeIndex(ds.time.values)
    pre = ds.sel(time=tm < start)
    post = ds.sel(time=tm > end)
    if pre.sizes["time"] == 0 or post.sizes["time"] == 0:
        return None, int(pre.sizes["time"]), int(post.sizes["time"])
    lr = 10 * np.log10(post.median("time") / pre.median("time"))
    return lr, int(pre.sizes["time"]), int(post.sizes["time"])


# ----------------------------------------------------------------------------- TESSERA
_HTTP = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tessera/tessera/npy/v1"
_S3 = "s3://us-west-2.opendata.source.coop/tessera/tessera/npy/v1"
_find_tiles_indexed = tessera.find_tiles


def _npy_header(url):
    import ast
    import requests
    raw = requests.get(url, headers={"Range": "bytes=0-255"}, timeout=60).content
    if raw[:6] != b"\x93NUMPY":
        return None
    hlen = int.from_bytes(raw[8:10], "little") + 10
    return ast.literal_eval(raw[10:hlen].decode("latin1")), hlen


def find_tiles_any_year(bbox, year):
    """Only some years have a published GeoParquet index; for the rest, build the
    same index rows by reading the .npy headers of the 0.1-degree tiles."""
    try:
        return _find_tiles_indexed(bbox, year)
    except Exception:
        pass
    minx, miny, maxx, maxy = bbox
    rows = []
    for lon in np.arange(np.floor(minx * 10) / 10 + 0.05, maxx + 0.05, 0.1):
        for lat in np.arange(np.floor(miny * 10) / 10 + 0.05, maxy + 0.05, 0.1):
            name = f"grid_{lon:.2f}_{lat:.2f}"
            g = _npy_header(f"{_HTTP}/{year}/{name}/{name}.npy")
            s = _npy_header(f"{_HTTP}/{year}/{name}/{name}_scales.npy")
            if g is None or s is None:
                continue
            (gh, go), (_, so) = g, s
            H, W, C = gh["shape"]
            rows.append(dict(height=H, width=W, channels=C, grid_offset=go, scales_offset=so,
                             grid_uri=f"{_S3}/{year}/{name}/{name}.npy",
                             scales_uri=f"{_S3}/{year}/{name}/{name}_scales.npy",
                             transform=[lon - 0.05, 0.1 / W, 0.0, lat + 0.05, 0.0, -0.1 / H]))
    return pd.DataFrame(rows)


tessera.find_tiles = find_tiles_any_year


def tessera_change(bb_ll, year, like):
    """Cosine distance between annual embeddings (year-1 -> year) on the 20 m grid."""
    pad = 0.002
    bb = (bb_ll[0] - pad, bb_ll[1] - pad, bb_ll[2] + pad, bb_ll[3] + pad)
    a = tessera.read_aoi(bbox=bb, year=year - 1)
    b = tessera.read_aoi(bbox=bb, year=year)
    b = b.reindex_like(a, method="nearest", tolerance=1e-4)
    num = (a * b).sum("band")
    den = np.sqrt((a ** 2).sum("band") * (b ** 2).sum("band"))
    dist = (1 - num / den).astype("float32")
    dist = dist.rio.write_crs(4326).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return dist.rio.reproject_match(like, resampling=5)  # 5 = average


# ----------------------------------------------------------------------------- helpers
def robust_z(da, mask=None):
    v = da.values[np.isfinite(da.values)] if mask is None else da.values[mask & np.isfinite(da.values)]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (da - med) / (mad if mad > 0 else 1.0), float(med), float(mad)


def clean(mask):
    m = ndi.binary_opening(mask, iterations=1)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.where(sizes >= MIN_PATCH_PX)[0] + 1)
    m = ndi.binary_closing(m, iterations=1)
    holes = ndi.binary_fill_holes(m) & ~m
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, range(1, n + 1))
        m = m | np.isin(lab, np.where(sizes <= 50)[0] + 1)  # fill holes <= 2 ha
    return m


def point_guided(mask, pt_rc, radius_px):
    """Keep burn patches connected (within GROUP_GAP_PX) to the area around the point."""
    grouped = ndi.binary_dilation(mask, iterations=GROUP_GAP_PX)
    lab, n = ndi.label(grouped)
    if n == 0:
        return np.zeros_like(mask)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    near = (yy - pt_rc[0]) ** 2 + (xx - pt_rc[1]) ** 2 <= radius_px ** 2
    keep = np.unique(lab[near & mask])
    keep = keep[keep > 0]
    return mask & np.isin(lab, keep)


def to_polygons(mask, transform, epsg, rule):
    geoms = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    if not geoms:
        return gpd.GeoDataFrame({"rule": []}, geometry=[], crs=epsg)
    return gpd.GeoDataFrame({"rule": [rule] * len(geoms)}, geometry=geoms, crs=epsg).dissolve("rule").reset_index()


def viirs_hits(key, f, start, end, bb_ll):
    p = ROOT / "data" / "firms" / f"{key}_viirs.csv"
    if not p.exists() or p.stat().st_size < 10:
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.acq_date)
    m = ((df.date >= start - pd.Timedelta(days=1)) & (df.date <= end + pd.Timedelta(days=2))
         & df.longitude.between(bb_ll[0], bb_ll[2]) & df.latitude.between(bb_ll[1], bb_ll[3]))
    return df[m]


# ----------------------------------------------------------------------------- main
def run(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
    epsg, (px, py), bb_utm, bb_ll = make_window(f)
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    log = {"key": key, "epsg": epsg}

    # --- Sentinel-2
    s2 = load_s2(cat, f, epsg, bb_ll, start - pd.Timedelta(days=PRE_DAYS), end + pd.Timedelta(days=POST_DAYS))
    tm = pd.DatetimeIndex(s2.time.values)
    pre_mask = tm < start
    post_mask = tm > end
    pre_nbr, n_pre = edge_composite(s2.NBR, tm, pre_mask, from_end=True)
    post_nbr, n_post = edge_composite(s2.NBR, tm, post_mask)
    dnbr = (pre_nbr - post_nbr).rename("dNBR")
    log["s2_scenes_pre"], log["s2_scenes_post"] = n_pre, n_post
    log["s2_dates"] = [str(d.date()) for d in tm]
    log["first_clear_after_days"] = int((tm[post_mask][0] - end).days) if post_mask.any() else None
    log["last_clear_before_days"] = int((start - tm[pre_mask][-1]).days) if pre_mask.any() else None

    mag, lo, hi = break_dates(s2.NBR)
    s0 = start.to_datetime64().astype("datetime64[D]").astype("int64") - DATE_TOL_DAYS
    e0 = end.to_datetime64().astype("datetime64[D]").astype("int64") + DATE_TOL_DAYS
    date_ok = ((lo <= e0) & (hi >= s0)).fillna(False)
    # break date expressed as days relative to the recorded start (midpoint of bracket)
    brk_rel = ((lo + hi) / 2 - start.to_datetime64().astype("datetime64[D]").astype("int64"))

    # --- grid helpers
    transform = dnbr.rio.write_crs(epsg).rio.transform()
    inv = ~transform
    col, row = inv * (px, py)
    pt_rc = (int(row), int(col))
    radius_px = POINT_RADIUS_M / RES

    # --- Sentinel-1
    geobox = s2.odc.geobox
    try:
        s1, n1pre, n1post = load_s1(cat, epsg, bb_ll, geobox, start, end)
    except Exception as e:  # keep going; radar is supplementary
        s1, n1pre, n1post = None, 0, 0
        log["s1_error"] = str(e)
    log["s1_scenes_pre"], log["s1_scenes_post"] = n1pre, n1post

    # --- TESSERA embeddings
    year = start.year
    try:
        tz = tessera_change(bb_ll, year, dnbr.rio.write_crs(epsg))
        tz = tz.assign_coords(x=dnbr.x, y=dnbr.y)
    except Exception as e:
        tz = None
        log["tessera_error"] = str(e)

    # --- VIIRS
    vi = viirs_hits(key, f, start, end, bb_ll)
    log["viirs_hits_in_window"] = int(len(vi))
    if len(vi):
        t = Transformer.from_crs(4326, epsg, always_xy=True)
        vx, vy = t.transform(vi.longitude.values, vi.latitude.values)
        log["viirs_min_dist_to_point_m"] = float(np.min(np.hypot(vx - px, vy - py)))

    # --- evidence + rules
    z_dnbr, med, mad = robust_z(dnbr)
    log["dnbr_background_median"], log["dnbr_background_mad"] = med, mad
    d = dnbr.values
    z = z_dnbr.values
    dok = date_ok.values
    zt = robust_z(tz)[0].values if tz is not None else None

    rules = {}
    rules["A_wall_to_wall"] = clean(np.nan_to_num(d) > 0.10)
    cand_b = clean(np.nan_to_num(z) >= 2.0)
    rules["B_point_guided"] = point_guided(cand_b, pt_rc, radius_px)
    cand_c = clean((np.nan_to_num(z) >= 1.5) & dok)
    rules["C_point_date"] = point_guided(cand_c, pt_rc, radius_px)
    if zt is not None:
        cand_d = clean(((np.nan_to_num(z) >= 1.5) & dok) | ((np.nan_to_num(z) >= 0.75) & dok & (np.nan_to_num(zt) >= 1.5)))
        rules["D_point_date_embed"] = point_guided(cand_d, pt_rc, radius_px)

    polys = []
    for name, m in rules.items():
        polys.append(to_polygons(m, transform, epsg, name))
    polys = pd.concat(polys, ignore_index=True)
    polys = gpd.GeoDataFrame(polys, crs=epsg)
    polys["ha"] = polys.area / 1e4
    polys.to_crs(4326).to_file(out / "polygons.geojson", driver="GeoJSON")

    # --- confidence / QA for the main product (rule C)
    main = rules["C_point_date"]
    qa = {}
    qa["detected"] = bool(main.sum() >= MIN_PATCH_PX)
    qa["mapped_ha"] = float(main.sum() * RES * RES / 1e4)
    qa["reported_ha"] = f["reported_acres"] / 2.47105
    qa["area_ratio_mapped_to_reported"] = qa["mapped_ha"] / qa["reported_ha"]
    qa["median_dnbr_in_polygon"] = float(np.nanmedian(d[main])) if main.any() else None
    qa["viirs_confirmed"] = bool(log["viirs_hits_in_window"] > 0 and log.get("viirs_min_dist_to_point_m", 1e9) < 3000)
    if s1 is not None and main.any():
        bg = ~ndi.binary_dilation(main, iterations=10)
        qa["s1_vh_change_in_minus_out_db"] = float(np.nanmedian(s1.vh.values[main]) - np.nanmedian(s1.vh.values[bg]))
    if zt is not None and main.any():
        qa["embedding_change_z_in_polygon"] = float(np.nanmedian(zt[main]))
    flags = []
    if (log["first_clear_after_days"] or 99) > 10:
        flags.append(f"first clear Sentinel-2 look {log['first_clear_after_days']} days after burn")
    if not qa["detected"]:
        flags.append("NOT DETECTED: no date-matched burn signal near the point")
    elif qa["area_ratio_mapped_to_reported"] < 0.33:
        flags.append("mapped area < 1/3 of reported (patchy/under-canopy burn or partial detection)")
    elif qa["area_ratio_mapped_to_reported"] > 2:
        flags.append("mapped area > 2x reported (possible commission or adjacent burns)")
    if not qa["viirs_confirmed"]:
        flags.append("no VIIRS active-fire detection (common for small/cool Rx burns)")
    score = sum([qa["detected"],
                 0.33 <= qa["area_ratio_mapped_to_reported"] <= 2 if qa["detected"] else False,
                 qa["viirs_confirmed"],
                 (log["first_clear_after_days"] or 99) <= 10,
                 (qa.get("embedding_change_z_in_polygon") or 0) > 1])
    qa["confidence"] = "high" if score >= 4 else "medium" if score >= 2 else "low"
    qa["flags"] = flags
    log["qa_rule_C"] = qa

    # --- save rasters for figures/validation
    layers = xr.Dataset({"dNBR": dnbr, "z_dNBR": z_dnbr, "break_mag": mag, "break_rel_days": brk_rel,
                         "date_ok": date_ok.astype("uint8"), "pre_NBR": pre_nbr, "post_NBR": post_nbr})
    for name, m in rules.items():
        layers[f"rule_{name}"] = (("y", "x"), m.astype("uint8"))
    if s1 is not None:
        layers["s1_vh_lr"] = s1.vh
        layers["s1_vv_lr"] = s1.vv
    if tz is not None:
        layers["tessera_dist"] = tz
    # RGB composites for figures
    for tag, msk, fe in [("pre", pre_mask, True), ("post", post_mask, False)]:
        for b in ["B04", "B03", "B02", "B12", "B8A"]:
            layers[f"{tag}_{b}"] = edge_composite(s2[b], tm, msk, from_end=fe)[0]
    layers = layers.rio.write_crs(epsg)
    layers.attrs = {}
    layers.to_netcdf(out / "layers.nc")
    log["point_xy"] = [px, py]
    log["window_utm"] = list(bb_utm)
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "s2_dates"}, indent=1, default=str))


if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k)
