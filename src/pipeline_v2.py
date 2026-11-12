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
def load_landsat(cat, bb_ll, geobox, t0, t1, with_thermal=False):
    items = P.stac_search(cat, collections=["landsat-c2-l2"], bbox=bb_ll,
                          datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                          query={"platform": {"in": ["landsat-8", "landsat-9"]},
                                 "eo:cloud_cover": {"lt": 95}})
    if len(items) == 0:
        return (None, None) if with_thermal else None
    zenith, azimuth = {}, {}
    for it in items:
        d = pd.Timestamp(it.datetime).tz_localize(None).normalize()
        zenith[d] = 90.0 - float(it.properties.get("view:sun_elevation", 50.0) or 50.0)
        azimuth[d] = float(it.properties.get("view:sun_azimuth", 155.0) or 155.0)
    bands = ["nir08", "swir22", "lwir11", "qa_pixel"] if with_thermal else ["nir08", "swir22", "qa_pixel"]
    ds = P.retry(lambda: odc.stac.load(items, bands=bands, geobox=geobox,
                                       groupby="solar_day", chunks={},
                                       resampling={"qa_pixel": "nearest", "*": "bilinear"}).compute())
    ds["time"] = pd.DatetimeIndex(ds.time.values).normalize()
    qa = ds.qa_pixel.values.astype("uint16")
    bit = lambda b: (qa >> b) & 1 == 1  # noqa: E731
    cloud = bit(1) | bit(2) | bit(3)
    near_cloud = np.stack([ndi.binary_dilation(c, iterations=17) for c in cloud])
    water = np.stack([ndi.binary_dilation(w, iterations=2) for w in bit(7)])
    invalid = bit(0) | cloud | bit(5) | water | (bit(4) & near_cloud)
    invalid |= (ds.nir08.values == 0) | (ds.swir22.values == 0)
    nir = ds.nir08.astype("float32") * 2.75e-5 - 0.2
    sw2 = ds.swir22.astype("float32") * 2.75e-5 - 0.2
    nbr = ((nir - sw2) / (nir + sw2)).where(~invalid)
    # bit 5 is snow: masked per pixel above, but it must not drop an otherwise clear date.
    cloud_frac = (bit(0) | bit(1) | bit(2) | bit(3)).mean(axis=(1, 2))
    valid_frac = nbr.notnull().mean(("x", "y")).values
    keep = (cloud_frac < 0.80) & (valid_frac > 0.02)
    if not np.any(keep):
        return (None, None) if with_thermal else None
    nbr = nbr.sel(time=keep)
    nbr.attrs["solar_zenith"] = [zenith.get(t, 40.0) for t in pd.DatetimeIndex(nbr.time.values)]
    nbr.attrs["solar_azimuth"] = [azimuth.get(t, 155.0) for t in pd.DatetimeIndex(nbr.time.values)]
    if not with_thermal:
        return nbr
    st_c = (ds.lwir11.astype("float32") * 0.00341802 + 149.0 - 273.15).where(~invalid & (ds.lwir11 > 0))
    return nbr, st_c.sel(time=keep)


def load_tcc(bb_ll, like, epsg):
    """USFS/FIA NLCD Tree Canopy Cover (0..100%) from the MRLC WCS service, reprojected to the
    20 m grid. Falls back to ESA WorldCover 10 m tree cover if MRLC is unreachable."""
    import requests
    from pyproj import Transformer
    from rasterio.io import MemoryFile
    try:
        tr = Transformer.from_crs(4326, 5070, always_xy=True)
        xs, ys = tr.transform([bb_ll[0], bb_ll[2], bb_ll[0], bb_ll[2]],
                              [bb_ll[1], bb_ll[1], bb_ll[3], bb_ll[3]])
        url = ("https://www.mrlc.gov/geoserver/mrlc_download/wcs?service=WCS&version=2.0.1"
               "&request=GetCoverage&coverageId=mrlc_download__nlcd_tcc_conus_2021_v2021-4"
               f"&format=image/tiff&subset=X({min(xs) - 500:.0f},{max(xs) + 500:.0f})"
               f"&subset=Y({min(ys) - 500:.0f},{max(ys) + 500:.0f})")
        r = requests.get(url, timeout=20)
        if r.ok and r.content[:2] in (b"II", b"MM"):
            with MemoryFile(r.content) as mem, xr.open_dataarray(mem, engine="rasterio") as da:
                tcc = da.squeeze(drop=True).astype("float32")
                tcc = tcc.where((tcc >= 0) & (tcc <= 100))
                tcc = tcc.rio.reproject_match(like.rio.write_crs(epsg), resampling=5)
                return np.clip(tcc.values, 0.0, 100.0), "usfs_fia_nlcd_tcc_2021"
    except Exception:
        pass
    try:
        cat = P.open_stac()
        items = [it for it in P.stac_search(cat, collections=["esa-worldcover"], bbox=bb_ll) if "2021" in it.id]
        wc = odc.stac.load(items, bands=["map"], crs=f"EPSG:{epsg}", resolution=10, bbox=bb_ll,
                           resampling="nearest")["map"].max("time").compute()
        tree = (wc == 10).astype("float32") * 100.0
        tree = tree.rio.write_crs(epsg).rio.reproject_match(like.rio.write_crs(epsg), resampling=5)
        return np.clip(tree.values, 0.0, 100.0), "esa_worldcover_10m"
    except Exception:
        return np.zeros(like.shape, dtype="float32"), "unavailable"


def topo_illumination(dem, zenith_deg, azimuth_deg):
    """Cosine of solar incidence angle cos(i) on the 20 m DEM for a given solar zenith & azimuth."""
    dz_dy, dz_dx = np.gradient(ndi.gaussian_filter(dem.astype("float64"), 1.0), RES)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)  # radians clockwise from north
    z = np.radians(zenith_deg)
    a = np.radians(azimuth_deg)
    cos_i = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return cos_i.astype("float32")


# ----------------------------------------------------------------------------- look-alike controls & stand affinity
def embedding_features(bb_ll, year, like, epsg):
    """Load pre-fire TESSERA 128-D embeddings and return:
      - lab:       k-means look-alike strata (H, W) for seasonal detrending
      - emb_unit:  L2-normalized 16-D PCA embedding vectors (H, W, 16) for stand cosine similarity
      - emb_edges: watershed stand-boundary lines (H, W) where vegetation type changes sharply"""
    pad = 0.002
    box = (bb_ll[0] - pad, bb_ll[1] - pad, bb_ll[2] + pad, bb_ll[3] + pad)
    emb = None
    for y in (year - 1, year - 2, 2024):
        try:
            emb = tessera.read_aoi(bbox=box, year=y)
            break
        except Exception:
            continue
    if emb is None:
        raise RuntimeError(f"no TESSERA embeddings found for {year - 1}..2024")
    emb = emb.transpose("band", "y", "x").rio.write_crs(4326).rio.set_spatial_dims(x_dim="x", y_dim="y")
    emb = emb.rio.reproject_match(like.rio.write_crs(epsg), resampling=5)
    X = emb.values.reshape(emb.shape[0], -1).T
    ok = np.isfinite(X).all(1)
    rng = np.random.default_rng(0)
    sample = rng.choice(np.where(ok)[0], size=min(50_000, ok.sum()), replace=False)
    pca = PCA(16, random_state=0).fit(X[sample])
    Z = np.zeros((X.shape[0], 16), dtype="float32")
    Z[ok] = pca.transform(X[ok])
    km = MiniBatchKMeans(N_STRATA, random_state=0, n_init=3).fit(Z[sample])
    lab = np.full(X.shape[0], -1, dtype=int)
    lab[ok] = km.predict(Z[ok])
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-6
    emb_unit = (Z / norms).reshape(like.shape + (16,))
    gx = np.linalg.norm(np.diff(emb_unit, axis=1, prepend=emb_unit[:, :1]), axis=-1)
    gy = np.linalg.norm(np.diff(emb_unit, axis=0, prepend=emb_unit[:1, :]), axis=-1)
    grad = ndi.gaussian_filter(np.hypot(gx, gy), 1.0)
    markers, n_m = ndi.label(h_minima(grad, 0.08))
    emb_edges = (watershed(grad, markers, watershed_line=True) == 0) if n_m >= 2 else np.zeros(like.shape, bool)
    return lab.reshape(like.shape), emb_unit, emb_edges


def embedding_strata(bb_ll, year, like, epsg):
    return embedding_features(bb_ll, year, like, epsg)[0]


def stand_affinity(emb_unit, dist, near_pt, core_seeds=None):
    """Cosine similarity (-1..1) of each pixel's pre-fire TESSERA embedding to the stand around
    the recorded point (within 120 m) and any high-confidence burned seed pixels."""
    if emb_unit is None:
        return None
    protos = []
    pt_zone = dist <= 120.0
    if np.any(pt_zone):
        p = emb_unit[pt_zone].mean(axis=0)
        n = np.linalg.norm(p)
        if n > 1e-5:
            protos.append(p / n)
    if core_seeds is not None and np.any(core_seeds & near_pt):
        p = emb_unit[core_seeds & near_pt].mean(axis=0)
        n = np.linalg.norm(p)
        if n > 1e-5:
            protos.append(p / n)
    if not protos:
        return None
    aff = np.max([emb_unit @ p for p in protos], axis=0).astype("float32")
    # Normalise against the search circle so uniform plantations still have a selective 0.72 cutoff
    if np.any(near_pt):
        q70 = float(np.nanquantile(aff[near_pt], 0.70))
        if q70 > 0.72:
            aff = aff - (q70 - 0.72)
    return aff


def adjust_series(nbr, lab, pool, dem=None, zenith=None, azimuth=None):
    """NBR minus the median NBR of look-alike control pixels on the same date, plus linear
    topographic sun-shadow (cos i) detrending when DEM and solar angles are provided."""
    vals = nbr.values
    out = np.full_like(vals, np.nan)
    labf, poolf = lab.ravel(), pool.ravel()
    for i in range(vals.shape[0]):
        v = vals[i].ravel().copy()
        m = poolf & np.isfinite(v)
        if m.sum() < 200:
            continue
        if dem is not None and zenith is not None and azimuth is not None:
            ci = topo_illumination(dem, float(zenith[i]), float(azimuth[i])).ravel()
            cm = m & np.isfinite(ci)
            if cm.sum() >= 200 and np.std(ci[cm]) > 0.02:
                # Remove shadow-driven slope darkening across the control pool
                slope_k = np.cov(ci[cm], v[cm])[0, 1] / (np.var(ci[cm]) + 1e-6)
                v = v - slope_k * (ci - np.median(ci[cm]))
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
    items = P.stac_search(cat, collections=["cop-dem-glo-30"], bbox=bb_ll)
    dem = P.retry(lambda: odc.stac.load(items, bands=["data"], geobox=geobox, resampling="bilinear").data.compute())
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
    """Build candidate units from OSM lines + DEM ridges + TESSERA stand boundaries, then snap v1 and v2."""
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
    emb_edges = ds["emb_edges"].values.astype(bool) if "emb_edges" in ds else None
    cells, lines = unit_cells(lines_gdf, ridges, like, transform, epsg, emb_edges=emb_edges)
    log["candidate_units"] = int(cells.max())
    reported_px = f["reported_acres"] * 4046.86 / (RES * RES)
    rules["v1_snap"] = snap(rules["v1"], cells, lines, reported_px)
    rules["v2_snap"] = snap(rules["v2"], cells, lines, reported_px)
    return cells, lines


def unit_cells(lines_gdf, ridges, like, transform, epsg, emb_edges=None):
    lines = ridges.copy()
    if emb_edges is not None:
        lines |= emb_edges.astype(bool)
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
    usable = (n <= SNAP_MAX_UNIT_X * reported_px) & (n * RES * RES >= 4000)
    usable[0] = False
    fill = usable & (frac >= SNAP_FILL_FRAC)
    trim = usable & (frac < SNAP_TRIM_FRAC)
    out = fill[cells] | (base & ~trim[cells] & (cells > 0))
    out |= lines & ndi.binary_closing(out, iterations=2)
    return out


def viirs_seed_mask(key, f, like, transform, epsg):
    """375 m footprints around active-fire VIIRS detections within 1 km of the point on the burn dates."""
    for d in (OUT / "firms", REF / "firms", ROOT / "data" / "firms"):
        p = d / f"{key}_viirs.csv"
        if p.exists() and p.stat().st_size >= 10:
            break
    else:
        return None
    df = pd.read_csv(p)
    if df.empty or "acq_date" not in df:
        return None
    dt = pd.to_datetime(df.acq_date)
    s, e = pd.Timestamp(f["start"]) - pd.Timedelta(days=1), pd.Timestamp(f["end"]) + pd.Timedelta(days=2)
    df = df[(dt >= s) & (dt <= e)]
    if df.empty:
        return None
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326).to_crs(epsg)
    pt = gpd.GeoSeries.from_xy([f["lon"]], [f["lat"]], crs=4326).to_crs(epsg).iloc[0]
    pts = pts[pts.distance(pt) <= P.POINT_RADIUS_M + 200]
    if pts.empty:
        return None
    foot = [g.buffer(200) for g in pts.geometry]
    return features.rasterize([(g, 1) for g in foot], out_shape=like.shape, transform=transform, fill=0).astype(bool)


# ----------------------------------------------------------------------------- main
def run(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
    epsg, (px, py), bb_utm, bb_ll = P.make_window(f)
    cat = P.open_stac()
    log = {"key": key, "epsg": epsg, "point_xy": [px, py]}
    t0, t1 = start - pd.Timedelta(days=P.PRE_DAYS), end + pd.Timedelta(days=P.POST_DAYS)

    s2 = P.load_s2(cat, f, epsg, bb_ll, t0, t1)
    geobox = s2.odc.geobox
    like = s2.NBR.isel(time=0, drop=True)
    transform = like.rio.write_crs(epsg).rio.transform()
    tm = pd.DatetimeIndex(s2.time.values)
    ls, ls_st = load_landsat(cat, bb_ll, geobox, t0, t1, with_thermal=True)
    dem = load_dem(cat, bb_ll, geobox)
    tcc, tcc_src = load_tcc(bb_ll, like, epsg)
    r_search = P.search_radius_m(f)
    yy, xx = np.meshgrid(s2.y.values, s2.x.values, indexing="ij")
    near_pt = np.hypot(xx - px, yy - py) <= r_search
    log["s2_looks"] = int(len(tm))
    log["landsat_looks"] = 0 if ls is None else int(ls.sizes["time"])
    log["search_radius_m"] = round(r_search, 0)
    log["tcc_source"] = tcc_src
    log["mean_tcc_pct"] = round(float(np.nanmean(tcc[near_pt])), 1) if np.isfinite(tcc[near_pt]).any() else 0.0

    # ---- round-1 style (Sentinel-2 only, unadjusted, with delayed scorch composite)
    pre1, _ = P.edge_composite(s2.NBR, tm, tm < start, from_end=True)
    post1, _ = P.post_scorch_composite(s2.NBR, tm, tm > end)
    dnbr1 = (pre1 - post1).values
    mag1, lo1, hi1 = P.break_dates(s2.NBR)
    s_day = start.to_datetime64().astype("datetime64[D]").astype("int64")
    e_day = end.to_datetime64().astype("datetime64[D]").astype("int64")
    date_ok1 = ((lo1 <= e_day + P.DATE_TOL_DAYS) & (hi1 >= s_day - P.DATE_TOL_DAYS)).fillna(False).values
    z1 = P.robust_z(pre1 - post1)[0].values

    # ---- round-2: merged series, look-alike + topographic illumination adjusted
    if ls is None:
        nbr = s2.NBR
        zen = s2.solar_zenith.values
        azi = s2.solar_azimuth.values
    else:
        ls_xy = ls.assign_coords(x=s2.x, y=s2.y)
        nbr = xr.concat([s2.NBR, ls_xy], "time")
        zen = np.concatenate([s2.solar_zenith.values, np.asarray(ls.attrs.get("solar_zenith", [40.0] * ls.sizes["time"]))])
        azi = np.concatenate([s2.solar_azimuth.values, np.asarray(ls.attrs.get("solar_azimuth", [155.0] * ls.sizes["time"]))])
        order = np.argsort(nbr.time.values)
        nbr, zen, azi = nbr.isel(time=order), zen[order], azi[order]
    tm2 = pd.DatetimeIndex(nbr.time.values)
    log["first_clear_after_days_merged"] = int((tm2[tm2 > end][0] - end).days) if (tm2 > end).any() else None
    log["first_clear_after_days_s2"] = int((tm[tm > end][0] - end).days) if (tm > end).any() else None

    area_m2 = f["reported_acres"] * 4046.86
    r_ctrl = P.POINT_RADIUS_M + 1.5 * np.sqrt(area_m2 / np.pi)
    dist = np.hypot(xx - px, yy - py)
    pool = dist > r_ctrl
    try:
        lab, emb_unit, emb_edges = embedding_features(bb_ll, start.year, like, epsg)
        log["strata"] = "tessera"
    except Exception as e:
        lab = np.zeros(like.shape, dtype=int)
        emb_unit, emb_edges = None, np.zeros(like.shape, bool)
        log["strata"] = f"single group (embeddings unavailable: {e})"
    adj = adjust_series(nbr, lab, pool, dem=dem, zenith=zen, azimuth=azi)
    pre2, _ = P.edge_composite(adj, tm2, tm2 < start, from_end=True)
    post2, _ = P.post_scorch_composite(adj, tm2, tm2 > end)
    dnbr2 = (pre2 - post2)
    mag2, lo2, hi2 = P.break_dates(adj)
    date_ok2 = ((lo2 <= e_day + P.DATE_TOL_DAYS) & (hi2 >= s_day - P.DATE_TOL_DAYS)).fillna(False).values

    # ---- Sub-pixel Char fraction change (distinguishes black char from autumn leaf-off / grass curing)
    adj_char = adjust_series(s2.CHAR, lab, pool, dem=dem,
                             zenith=s2.solar_zenith.values, azimuth=s2.solar_azimuth.values)
    pre_char, _ = P.edge_composite(adj_char, tm, tm < start, from_end=True)
    post_char, _ = P.edge_composite(adj_char, tm, tm > end)
    dchar = (post_char - pre_char).values
    z_char = P.robust_z(post_char - pre_char)[0].values
    # Senescence / leaf-off veto: if Red and SWIR1 both brightened while Char did not rise, it is leaf/grass curing, not char
    pre_b04 = P.edge_composite(s2.B04, tm, tm < start, from_end=True)[0].values
    post_b04 = P.edge_composite(s2.B04, tm, tm > end)[0].values
    pre_b8a = P.edge_composite(s2.B8A, tm, tm < start, from_end=True)[0].values
    post_b8a = P.edge_composite(s2.B8A, tm, tm > end)[0].values
    senescence_only = (post_b04 > pre_b04 + 0.015) & (post_b8a >= pre_b8a - 0.01) & (np.nan_to_num(dchar) <= 0.01)

    # ---- Landsat thermal anomaly (post-burn surface warming relative to look-alike controls)
    dt_thermal = np.full(like.shape, np.nan, dtype="float32")
    if ls_st is not None and ls_st.sizes["time"]:
        ls_st = ls_st.assign_coords(x=s2.x, y=s2.y)
        tl = pd.DatetimeIndex(ls_st.time.values)
        adj_st = adjust_series(ls_st, lab, pool)
        if (tl > end).any():
            post_st, _ = P.edge_composite(adj_st, tl, tl > end)
            pre_st, n_pre_st = P.edge_composite(adj_st, tl, tl < start, from_end=True)
            dt_thermal = (post_st - pre_st).values if n_pre_st > 0 else post_st.values

    # ---- Canopy-attenuation scaling (USFS/FIA TCC): surface signal under partial canopy (30..80% TCC)
    # is attenuated by tree crowns, but only boosted where sub-pixel char also increased.
    c_frac = np.clip(np.nan_to_num(tcc, nan=0.0) / 100.0, 0.0, 0.75)
    canopy_boost = np.where(np.nan_to_num(dchar) > 0.01, 1.0 / np.maximum(1.0 - 0.5 * c_frac, 0.65), 1.0)
    dnbr_eff = xr.DataArray(np.where(senescence_only, 0.0, dnbr2.values * canopy_boost),
                            dims=("y", "x"), coords={"y": s2.y, "x": s2.x})
    z2 = P.robust_z(dnbr_eff)[0].values
    # Combine dNBR z-score with sub-pixel char z-score and thermal warming confirmation
    char_bonus = np.clip(np.nan_to_num(z_char, nan=0.0) * 0.25, 0.0, 0.5)
    therm_bonus = np.where(np.nan_to_num(dt_thermal, nan=0.0) >= 1.5, 0.25, 0.0)
    cold_veto = np.isfinite(dt_thermal) & (dt_thermal < -2.5)  # cold shadow / snowmelt
    z2_comb = np.where(senescence_only | cold_veto, 0.0, np.nan_to_num(z2, nan=-9.0) + char_bonus + therm_bonus)

    # ---- rules without snapping (size-scaled search radius + stand-affinity seed-and-grow + VIIRS seeds)
    inv = ~transform
    col, row = inv * (px, py)
    pt_rc, rad = (int(row), int(col)), r_search / RES
    vmask = viirs_seed_mask(key, f, like, transform, epsg)
    core_seeds = (z2_comb >= 1.5) & date_ok2 & (dnbr_eff.values >= 0.08)
    aff = stand_affinity(emb_unit, dist, near_pt, core_seeds=core_seeds)
    raw1 = P.seed_and_grow(z1, dnbr1, date_ok1, viirs_mask=vmask) & near_pt
    raw2 = P.seed_and_grow(z2_comb, dnbr_eff.values, date_ok2, viirs_mask=vmask,
                           stand_aff=aff, stand_edges=emb_edges, dchar=dchar) & near_pt
    rules = {
        "std": P.clean(np.nan_to_num(dnbr1) > 0.10),
        "v1": P.point_guided(P.clean(raw1), pt_rc, rad),
        "v2": P.point_guided(P.clean(raw2), pt_rc, rad),
    }

    ds = xr.Dataset({"dnbr_s2": (("y", "x"), dnbr1), "dnbr_adj": dnbr_eff, "z_adj": (("y", "x"), z2_comb),
                     "dchar": (("y", "x"), dchar.astype("float32")),
                     "dt_thermal": (("y", "x"), dt_thermal.astype("float32")),
                     "tcc": (("y", "x"), tcc.astype("float32")),
                     "stand_aff": (("y", "x"), (np.zeros(like.shape, "float32") if aff is None else aff)),
                     "emb_edges": (("y", "x"), emb_edges.astype("uint8")),
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
    ds = ds.drop_vars([c for c in ds.coords if c not in ("y", "x", "spatial_ref")])
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
