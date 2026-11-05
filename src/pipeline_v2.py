"""Round 2: point + date -> burn polygon, free data only, no hand work.

Adds three things to round 1 (pipeline.py), all with settings fixed in advance:

1. More clear looks: Landsat 8/9 is merged with Sentinel-2.
2. Green-up / dry-down correction with look-alike controls: pixels are grouped by
   their *pre-fire-year* TESSERA embedding; on every image date each pixel's NBR is
   compared with the median NBR of its group, using only pixels well away from the
   reported point. Seasonal change (and Landsat-vs-Sentinel differences) cancel out;
   a burn stands out.
3. Control-line snapping (run with and without): roads, tracks, trails, streams and
   power lines from Overture Maps (OpenStreetMap-derived) plus ridgelines from the
   Copernicus 30 m DEM cut the landscape into candidate "units". A unit that is mostly mapped as burned is
   filled in; burned pixels that spill thinly across a control line are trimmed.

Variants written for every fire:
  std      Sentinel-2 dNBR > 0.1 anywhere in the window (round-1 "standard")
  v1       round-1 main method: Sentinel-2 only, point + date match
  v1_snap  v1 snapped to control lines
  v2       Sentinel-2 + Landsat, look-alike-adjusted, point + date match
  v2_snap  v2 snapped to control lines
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
import xarray as xr
from rasterio import features
from scipy import ndimage as ndi
from skimage.morphology import h_minima
from skimage.segmentation import watershed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402  (also patches the TESSERA tile finder)
import tessera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference_v2"
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs_v2"
RES = P.RES

N_STRATA = 30
MIN_STRATUM_PX = 30
DEM_RIDGE_DEPTH_M = 20      # basins shallower than this are merged
SNAP_FILL_FRAC = 0.4        # unit >= 40% mapped burned -> fill whole unit
SNAP_TRIM_FRAC = 0.1        # unit < 10% mapped burned -> treat as spill-over, drop
SNAP_MAX_UNIT_X = 3         # units > 3x reported size are not used for snapping
OVERTURE = "s3://overturemaps-us-west-2/release/2026-09-23.1"


# ----------------------------------------------------------------------------- Landsat
def load_landsat(cat, bb_ll, geobox, t0, t1):
    items = cat.search(collections=["landsat-c2-l2"], bbox=bb_ll,
                       datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                       query={"platform": {"in": ["landsat-8", "landsat-9"]},
                              "eo:cloud_cover": {"lt": 95}}).item_collection()
    if len(items) == 0:
        return None
    ds = odc.stac.load(items, bands=["nir08", "swir22", "qa_pixel"], geobox=geobox,
                       groupby="solar_day", chunks={},
                       resampling={"qa_pixel": "nearest", "*": "bilinear"}).compute()
    ds["time"] = pd.DatetimeIndex(ds.time.values).normalize()
    qa = ds.qa_pixel.values.astype("uint16")
    bit = lambda b: (qa >> b) & 1 == 1  # noqa: E731
    cloud = bit(1) | bit(2) | bit(3)
    near_cloud = np.stack([ndi.binary_dilation(c, iterations=17) for c in cloud])
    invalid = bit(0) | cloud | bit(5) | (bit(4) & near_cloud)
    invalid |= (ds.nir08.values == 0) | (ds.swir22.values == 0)
    nir = ds.nir08.astype("float32") * 2.75e-5 - 0.2
    sw2 = ds.swir22.astype("float32") * 2.75e-5 - 0.2
    nbr = ((nir - sw2) / (nir + sw2)).where(~invalid)
    keep = nbr.notnull().mean(("x", "y")).values > 0.2
    return nbr.sel(time=keep)


# ----------------------------------------------------------------------------- look-alike controls
def embedding_strata(bb_ll, year, like, epsg):
    pad = 0.002
    emb = tessera.read_aoi(bbox=(bb_ll[0] - pad, bb_ll[1] - pad, bb_ll[2] + pad, bb_ll[3] + pad), year=year - 1)
    emb = emb.transpose("band", "y", "x").rio.write_crs(4326).rio.set_spatial_dims(x_dim="x", y_dim="y")
    emb = emb.rio.reproject_match(like.rio.write_crs(epsg), resampling=5)
    X = emb.values.reshape(emb.shape[0], -1).T
    ok = np.isfinite(X).all(1)
    rng = np.random.default_rng(0)
    sample = rng.choice(np.where(ok)[0], size=min(50_000, ok.sum()), replace=False)
    pca = PCA(16, random_state=0).fit(X[sample])
    km = MiniBatchKMeans(N_STRATA, random_state=0, n_init=3).fit(pca.transform(X[sample]))
    lab = np.full(X.shape[0], -1, dtype=int)
    lab[ok] = km.predict(pca.transform(X[ok]))
    return lab.reshape(like.shape)


def adjust_series(nbr, lab, pool):
    """NBR minus the median NBR of look-alike control pixels on the same date."""
    vals = nbr.values
    out = np.full_like(vals, np.nan)
    labf, poolf = lab.ravel(), pool.ravel()
    for i in range(vals.shape[0]):
        v = vals[i].ravel()
        m = poolf & np.isfinite(v)
        if m.sum() < 200:
            continue
        g = pd.DataFrame({"l": labf[m], "v": v[m]}).groupby("l").v.agg(["median", "count"])
        g = g[(g["count"] >= MIN_STRATUM_PX) & (g.index >= 0)]["median"]
        exp = np.full(v.shape, np.median(v[m]))
        has = np.isin(labf, g.index.values)
        exp[has] = g.reindex(labf[has]).values
        out[i] = (v - exp).reshape(vals.shape[1:])
    return xr.DataArray(out, dims=nbr.dims, coords=nbr.coords)


# ----------------------------------------------------------------------------- control lines
def control_line_features(key, bb_ll):
    """Roads, tracks, trails, rail, streams/rivers/canals (incl. water-body edges) and
    power lines from Overture Maps (largely OpenStreetMap-derived), read straight from
    its public cloud files. Unlike the public OpenStreetMap query servers there is no
    rate limit, so this scales to hundreds of points. The release is pinned."""
    cache = CACHE / f"lines_{key}.geojson"
    if cache.exists():
        return gpd.read_file(cache)
    import duckdb
    import shapely
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    w, s, e, n = bb_ll
    box = f"bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s}"
    src = lambda theme, typ: (f"read_parquet('{OVERTURE}/theme={theme}/type={typ}/*', "  # noqa: E731
                              "hive_partitioning=1)")
    queries = {
        "transport": f"SELECT class AS kind, ST_AsWKB(geometry) g FROM {src('transportation', 'segment')} WHERE {box}",
        "water": (f"SELECT subtype AS kind, ST_AsWKB(ST_Boundary(geometry)) g FROM {src('base', 'water')} "
                  f"WHERE {box} AND ST_GeometryType(geometry) IN ('POLYGON','MULTIPOLYGON') "
                  f"UNION ALL SELECT subtype, ST_AsWKB(geometry) FROM {src('base', 'water')} "
                  f"WHERE {box} AND ST_GeometryType(geometry) IN ('LINESTRING','MULTILINESTRING')"),
        "power": (f"SELECT class AS kind, ST_AsWKB(geometry) g FROM {src('base', 'infrastructure')} "
                  f"WHERE {box} AND subtype = 'power' AND ST_GeometryType(geometry) IN ('LINESTRING','MULTILINESTRING')"),
    }
    frames = []
    for name, q in queries.items():
        df = con.execute(q).df()
        if len(df):
            frames.append(gpd.GeoDataFrame({"source": name, "kind": df["kind"]},
                                           geometry=shapely.from_wkb([bytes(b) for b in df["g"]]), crs=4326))
    gdf = (gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326) if frames
           else gpd.GeoDataFrame({"source": [], "kind": []}, geometry=[], crs=4326))
    CACHE.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def load_dem(cat, bb_ll, geobox):
    items = cat.search(collections=["cop-dem-glo-30"], bbox=bb_ll).item_collection()
    dem = odc.stac.load(items, bands=["data"], geobox=geobox, resampling="bilinear").data
    return dem.max("time").values.astype("float64")


def ridges_from_dem(dem):
    """Drainage divides between basins at least DEM_RIDGE_DEPTH_M deep. Flat land
    (e.g. the Everglades) has no such basins and therefore contributes no ridges."""
    dem_s = ndi.gaussian_filter(dem, 2)
    markers, n = ndi.label(h_minima(dem_s, DEM_RIDGE_DEPTH_M))
    if n < 2:
        return np.zeros(dem.shape, bool)
    ws = watershed(dem_s, markers, watershed_line=True)
    return ws == 0


def control_lines_and_snap(key, f, ds, log, epsg, bb_ll, transform, rules):
    """Build candidate units from OSM lines + DEM ridges, then snap v1 and v2."""
    like = ds["dnbr_s2"]
    for k in ("osm_lines", "osm_error"):
        log.pop(k, None)
    try:
        lines_gdf = control_line_features(key, bb_ll)
        log["control_line_features"] = int(len(lines_gdf))
        log["control_line_source"] = OVERTURE
        log.pop("control_line_error", None)
    except Exception as e:
        lines_gdf = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=4326)
        log["control_line_error"] = str(e)
    ridges = ridges_from_dem(ds["dem"].values.astype("float64"))
    cells, lines = unit_cells(lines_gdf, ridges, like, transform, epsg)
    log["candidate_units"] = int(cells.max())
    reported_px = f["reported_acres"] * 4046.86 / (RES * RES)
    rules["v1_snap"] = snap(rules["v1"], cells, lines, reported_px)
    rules["v2_snap"] = snap(rules["v2"], cells, lines, reported_px)
    return cells, lines


def unit_cells(lines_gdf, ridges, like, transform, epsg):
    lines = ridges.copy()
    geoms = [g for g in lines_gdf.to_crs(epsg).geometry if g is not None and not g.is_empty]
    if geoms:
        lines |= features.rasterize([(g, 1) for g in geoms], out_shape=like.shape,
                                    transform=transform, fill=0, all_touched=True).astype(bool)
    cells, _ = ndi.label(~lines)  # 4-connected: units cannot leak through diagonal lines
    return cells, lines


def snap(base, cells, lines, reported_px):
    if not base.any():
        return base.copy()
    n = np.bincount(cells.ravel())
    burned = np.bincount(cells.ravel(), weights=base.ravel().astype(float), minlength=len(n))
    frac = np.divide(burned, n, out=np.zeros_like(burned), where=n > 0)
    usable = (n <= SNAP_MAX_UNIT_X * reported_px) & (n * RES * RES >= 1e4)
    usable[0] = False
    fill = usable & (frac >= SNAP_FILL_FRAC)
    trim = usable & (frac < SNAP_TRIM_FRAC)
    out = fill[cells] | (base & ~trim[cells] & (cells > 0))
    out |= lines & ndi.binary_closing(out, iterations=2)
    return out


# ----------------------------------------------------------------------------- main
def run(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
    epsg, (px, py), bb_utm, bb_ll = P.make_window(f)
    cat = pystac_client.Client.open(P.STAC, modifier=planetary_computer.sign_inplace)
    log = {"key": key, "epsg": epsg, "point_xy": [px, py]}
    t0, t1 = start - pd.Timedelta(days=P.PRE_DAYS), end + pd.Timedelta(days=P.POST_DAYS)

    s2 = P.load_s2(cat, f, epsg, bb_ll, t0, t1)
    geobox = s2.odc.geobox
    like = s2.NBR.isel(time=0)
    transform = like.rio.write_crs(epsg).rio.transform()
    tm = pd.DatetimeIndex(s2.time.values)
    ls = load_landsat(cat, bb_ll, geobox, t0, t1)
    log["s2_looks"] = int(len(tm))
    log["landsat_looks"] = 0 if ls is None else int(ls.sizes["time"])

    # ---- round-1 style (Sentinel-2 only, unadjusted)
    pre1, _ = P.edge_composite(s2.NBR, tm, tm < start, from_end=True)
    post1, _ = P.edge_composite(s2.NBR, tm, tm > end)
    dnbr1 = (pre1 - post1).values
    mag1, lo1, hi1 = P.break_dates(s2.NBR)
    s_day = start.to_datetime64().astype("datetime64[D]").astype("int64")
    e_day = end.to_datetime64().astype("datetime64[D]").astype("int64")
    date_ok1 = ((lo1 <= e_day + P.DATE_TOL_DAYS) & (hi1 >= s_day - P.DATE_TOL_DAYS)).fillna(False).values
    z1 = P.robust_z(pre1 - post1)[0].values

    # ---- round-2: merged series, look-alike adjusted
    nbr = s2.NBR if ls is None else xr.concat([s2.NBR, ls.assign_coords(x=s2.x, y=s2.y)], "time").sortby("time")
    tm2 = pd.DatetimeIndex(nbr.time.values)
    log["first_clear_after_days_merged"] = int((tm2[tm2 > end][0] - end).days) if (tm2 > end).any() else None
    log["first_clear_after_days_s2"] = int((tm[tm > end][0] - end).days) if (tm > end).any() else None

    area_m2 = f["reported_acres"] * 4046.86
    r_ctrl = P.POINT_RADIUS_M + 1.5 * np.sqrt(area_m2 / np.pi)
    yy, xx = np.meshgrid(s2.y.values, s2.x.values, indexing="ij")
    pool = np.hypot(xx - px, yy - py) > r_ctrl
    try:
        lab = embedding_strata(bb_ll, start.year, like, epsg)
        log["strata"] = "tessera"
    except Exception as e:
        lab = np.zeros(like.shape, dtype=int)
        log["strata"] = f"single group (embeddings unavailable: {e})"
    adj = adjust_series(nbr, lab, pool)
    pre2, _ = P.edge_composite(adj, tm2, tm2 < start, from_end=True)
    post2, _ = P.edge_composite(adj, tm2, tm2 > end)
    dnbr2 = (pre2 - post2)
    mag2, lo2, hi2 = P.break_dates(adj)
    date_ok2 = ((lo2 <= e_day + P.DATE_TOL_DAYS) & (hi2 >= s_day - P.DATE_TOL_DAYS)).fillna(False).values
    z2 = P.robust_z(dnbr2)[0].values

    # ---- rules without snapping (same thresholds as round 1)
    inv = ~transform
    col, row = inv * (px, py)
    pt_rc, rad = (int(row), int(col)), P.POINT_RADIUS_M / RES
    rules = {
        "std": P.clean(np.nan_to_num(dnbr1) > 0.10),
        "v1": P.point_guided(P.clean((np.nan_to_num(z1) >= 1.5) & date_ok1), pt_rc, rad),
        "v2": P.point_guided(P.clean((np.nan_to_num(z2) >= 1.5) & date_ok2), pt_rc, rad),
    }

    dem = load_dem(cat, bb_ll, geobox)
    ds = xr.Dataset({"dnbr_s2": (("y", "x"), dnbr1), "dnbr_adj": dnbr2, "z_adj": (("y", "x"), z2),
                     "date_ok_adj": (("y", "x"), date_ok2.astype("uint8")),
                     "strata": (("y", "x"), lab.astype("int16")),
                     "dem": (("y", "x"), dem.astype("float32"))},
                    coords={"y": s2.y, "x": s2.x})
    for b in ["B12", "B8A", "B04", "B03", "B02"]:
        ds[f"post_{b}"] = P.edge_composite(s2[b], tm, tm > end)[0]
        ds[f"pre_{b}"] = P.edge_composite(s2[b], tm, tm < start, from_end=True)[0]
    finish(key, f, ds, log, epsg, bb_ll, transform, rules)


def finish(key, f, ds, log, epsg, bb_ll, transform, rules):
    """Control lines + snapping, then write polygons, layers and log."""
    out = OUT / key
    cells, lines = control_lines_and_snap(key, f, ds, log, epsg, bb_ll, transform, rules)
    polys = [P.to_polygons(m, transform, epsg, k) for k, m in rules.items()]
    polys = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=epsg)
    polys["ha"] = polys.area / 1e4
    polys.to_crs(4326).to_file(out / "polygons.geojson", driver="GeoJSON")
    log["mapped_ha"] = {k: round(float(m.sum() * RES * RES / 1e4), 2) for k, m in rules.items()}
    ds["lines"] = (("y", "x"), lines.astype("uint8"))
    ds["cells"] = (("y", "x"), cells.astype("int32"))
    for k, m in rules.items():
        ds[f"rule_{k}"] = (("y", "x"), m.astype("uint8"))
    ds = ds.rio.write_crs(epsg)
    ds.attrs = {}
    tmp = out / "layers.tmp.nc"
    ds.to_netcdf(tmp)
    tmp.replace(out / "layers.nc")
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(key, json.dumps(log["mapped_ha"]), log.get("control_line_features", log.get("control_line_error")),
          log["candidate_units"], log.get("s2_looks"), log.get("landsat_looks"), flush=True)


def relines(key):
    """Redo only control lines + snapping from saved layers (no imagery re-download)."""
    f = json.loads((REF / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    with xr.open_dataset(OUT / key / "layers.nc", decode_coords="all") as src:
        ds = src.load()
    epsg = log["epsg"]
    _, _, _, bb_ll = P.make_window(f)
    transform = ds.rio.transform()
    rules = {k: ds[f"rule_{k}"].values.astype(bool) for k in ("std", "v1", "v2")}
    ds = ds.drop_vars([v for v in ds.data_vars if v.startswith("rule_") or v in ("lines", "cells")])
    finish(key, f, ds, log, epsg, bb_ll, transform, rules)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--relines":
        keys = args[1:] or [k for k in json.loads((REF / "fires.json").read_text())
                            if (OUT / k / "run_log.json").exists()]
        for k in keys:
            relines(k)
        sys.exit(0)
    keys = args or list(json.loads((REF / "fires.json").read_text()))
    for k in keys:
        if (OUT / k / "run_log.json").exists():
            continue
        try:
            run(k)
        except Exception as e:  # one bad fire should not stop the batch; it is logged
            (OUT / k).mkdir(parents=True, exist_ok=True)
            (OUT / k / "error.txt").write_text(repr(e))
            print(k, "FAILED", repr(e), flush=True)
