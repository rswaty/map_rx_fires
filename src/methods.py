"""The two method layers the round-2 run (pipeline_v2.py) does not save. Normally run via run_csv.py.

    python src/methods.py <output folder>            # all burns with a round-2 run; skips burns done
    python src/methods.py <output folder> <key> ...  # selected burns (recomputed)

Writes <output folder>/<key>/methods.nc on the same 20 m grid as layers.nc:
  rule_landsat  Landsat 8/9 only, point + date match (same thresholds as round 1)
  rule_combo    Sentinel-2 + Landsat, each date compared with the median of pixels
                far from the point (one group), point + date match. This is round 2
                without the embedding look-alike groups.
  dnbr_landsat, dnbr_combo
  ls_pre_*/ls_post_*   Landsat false-colour composites (B12 = SWIR2, B8A = NIR, B04 = red,
                       named like the Sentinel-2 bands)
"""
import json
import sys
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import xarray as xr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
import pipeline_v2 as V  # noqa: E402

OUT = V.ROOT / "outputs_arkansas"


def set_output(out):
    global OUT
    OUT = Path(out)


def load_landsat(cat, bb_ll, geobox, t0, t1):
    """Landsat 8/9 L2 surface reflectance with the same cloud mask as pipeline_v2."""
    items = cat.search(collections=["landsat-c2-l2"], bbox=bb_ll,
                       datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                       query={"platform": {"in": ["landsat-8", "landsat-9"]},
                              "eo:cloud_cover": {"lt": 95}}).item_collection()
    if len(items) == 0:
        return None
    ds = odc.stac.load(items, bands=["red", "nir08", "swir22", "qa_pixel"], geobox=geobox,
                       groupby="solar_day", chunks={},
                       resampling={"qa_pixel": "nearest", "*": "bilinear"}).compute()
    ds["time"] = pd.DatetimeIndex(ds.time.values).normalize()
    qa = ds.qa_pixel.values.astype("uint16")
    bit = lambda b: (qa >> b) & 1 == 1  # noqa: E731
    cloud = bit(1) | bit(2) | bit(3)
    near_cloud = np.stack([ndi.binary_dilation(c, iterations=17) for c in cloud])
    invalid = bit(0) | cloud | bit(5) | (bit(4) & near_cloud)
    invalid |= (ds.nir08.values == 0) | (ds.swir22.values == 0)
    out = xr.Dataset({b: (ds[s].astype("float32") * 2.75e-5 - 0.2).where(~invalid)
                      for b, s in [("B04", "red"), ("B8A", "nir08"), ("B12", "swir22")]})
    out["NBR"] = (out.B8A - out.B12) / (out.B8A + out.B12)
    keep = out.NBR.notnull().mean(("x", "y")).values > 0.2
    return out.sel(time=keep)


def point_date_rule(nbr, tm, start, end, pt_rc):
    pre, _ = P.edge_composite(nbr, tm, tm < start, from_end=True)
    post, _ = P.edge_composite(nbr, tm, tm > end)
    dnbr = pre - post
    _, lo, hi = P.break_dates(nbr)
    s_day = start.to_datetime64().astype("datetime64[D]").astype("int64")
    e_day = end.to_datetime64().astype("datetime64[D]").astype("int64")
    date_ok = ((lo <= e_day + P.DATE_TOL_DAYS) & (hi >= s_day - P.DATE_TOL_DAYS)).fillna(False).values
    if not np.isfinite(dnbr.values).any():
        return dnbr.values, np.zeros(dnbr.shape, bool)
    z = P.robust_z(dnbr)[0].values
    rule = P.point_guided(P.clean((np.nan_to_num(z) >= 1.5) & date_ok), pt_rc, P.POINT_RADIUS_M / P.RES)
    return dnbr.values, rule


def run(key):
    f = json.loads((OUT / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    with xr.open_dataset(OUT / key / "layers.nc", decode_coords="all") as src:
        grid = src[["dnbr_s2"]].load()
    start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
    epsg, (px, py), _, bb_ll = P.make_window(f)
    cat = pystac_client.Client.open(P.STAC, modifier=planetary_computer.sign_inplace)
    t0, t1 = start - pd.Timedelta(days=P.PRE_DAYS), end + pd.Timedelta(days=P.POST_DAYS)

    s2 = P.load_s2(cat, f, epsg, bb_ll, t0, t1)
    s2 = s2.assign_coords(x=grid.x, y=grid.y) if s2.sizes["x"] == grid.sizes["x"] else s2.reindex_like(grid, method="nearest")
    geobox = s2.odc.geobox
    ls = load_landsat(cat, bb_ll, geobox, t0, t1)
    transform = grid.rio.transform()
    col, row = ~transform * (px, py)
    pt_rc = (int(row), int(col))
    ds = xr.Dataset(coords={"y": grid.y, "x": grid.x})
    info = {"epsg": epsg}

    if ls is not None and ls.sizes["time"]:
        ls = ls.assign_coords(x=grid.x, y=grid.y)
        tl = pd.DatetimeIndex(ls.time.values)
        dn, rule = point_date_rule(ls.NBR, tl, start, end, pt_rc)
        for b in ["B12", "B8A", "B04"]:
            ds[f"ls_pre_{b}"] = P.edge_composite(ls[b], tl, tl < start, from_end=True)[0]
            ds[f"ls_post_{b}"] = P.edge_composite(ls[b], tl, tl > end)[0]
        info["landsat_dates_before"] = [str(d.date()) for d in tl[tl < start]]
        info["landsat_dates_after"] = [str(d.date()) for d in tl[tl > end]]
    else:
        dn, rule = np.full(grid.dnbr_s2.shape, np.nan), np.zeros(grid.dnbr_s2.shape, bool)
        info["landsat_dates_before"], info["landsat_dates_after"] = [], []
    ds["dnbr_landsat"] = (("y", "x"), dn)
    ds["rule_landsat"] = (("y", "x"), rule.astype("uint8"))

    nbr = s2.NBR if ls is None else xr.concat([s2.NBR, ls.NBR], "time").sortby("time")
    tm2 = pd.DatetimeIndex(nbr.time.values)
    area_m2 = f["reported_acres"] * 4046.86
    yy, xx = np.meshgrid(grid.y.values, grid.x.values, indexing="ij")
    pool = np.hypot(xx - px, yy - py) > P.POINT_RADIUS_M + 1.5 * np.sqrt(area_m2 / np.pi)
    adj = V.adjust_series(nbr, np.zeros(pool.shape, int), pool)
    dn, rule = point_date_rule(adj, tm2, start, end, pt_rc)
    ds["dnbr_combo"] = (("y", "x"), dn)
    ds["rule_combo"] = (("y", "x"), rule.astype("uint8"))
    s2t = pd.DatetimeIndex(s2.time.values)
    info["s2_dates_before"] = [str(d.date()) for d in s2t[s2t < start]]
    info["s2_dates_after"] = [str(d.date()) for d in s2t[s2t > end]]
    info["mapped_ha"] = {k: round(float(ds[f"rule_{k}"].values.sum() * P.RES ** 2 / 1e4), 2)
                         for k in ("landsat", "combo")}

    ds = ds.rio.write_crs(log["epsg"])
    ds.attrs = {"info": json.dumps(info)}
    tmp = OUT / key / "methods.tmp.nc"
    ds.to_netcdf(tmp)
    tmp.replace(OUT / key / "methods.nc")
    print(key, info["mapped_ha"], len(info["landsat_dates_before"]), len(info["landsat_dates_after"]), flush=True)


if __name__ == "__main__":
    set_output(sys.argv[1])
    keys = sys.argv[2:] or [k for k in json.loads((OUT / "fires.json").read_text())
                            if (OUT / k / "run_log.json").exists() and not (OUT / k / "methods.nc").exists()]
    for k in keys:
        try:
            run(k)
        except Exception as e:  # one bad burn should not stop the batch
            import traceback
            traceback.print_exc()
            print(k, "FAILED", repr(e), flush=True)
