"""The five reviewer maps per burn, from the saved layers. Normally run via run_csv.py.

    python src/maps.py <output folder> ["Map title"]   # FIRMS_MAP_KEY=<key> to fetch VIIRS

For each burn (<output folder>/review/):
  <key>.pdf  five pages, also saved as PNGs:
    <key>_1_landsat.png      Landsat 8/9 before and after, Landsat-only outline
    <key>_2_sentinel.png     Sentinel-2 before and after, Sentinel-2-only outline (round 1)
    <key>_3_combined.png     Sentinel-2 + Landsat combined, no embeddings
    <key>_4_viirs.png        VIIRS 375 m active-fire detections (NASA FIRMS)
    <key>_5_best_effort.png  Sentinel-2 + Landsat with embedding look-alike correction (round 2),
                             burn severity inside
All burns:
  burns.kmz, burns.gpkg  one layer per method (best effort shown by default)
  summary.csv, summary.pdf

Severity = green-up-corrected dNBR inside the best-effort outline, Key & Benson (2006)
breaks. Not field-calibrated.
Confidence rule for the best effort (fixed before results were seen):
  low    not detected, mapped < 0.2 x reported, or first clear look > 21 days after burn
  high   mapped 0.5-2 x reported and first clear look <= 10 days after burn
  medium otherwise
"""
import json
import os
import sys
import warnings
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import rioxarray  # noqa: E402,F401
import simplekml  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib_scalebar.scalebar import ScaleBar  # noqa: E402
from rasterio import features  # noqa: E402
from shapely.geometry import Point, box, shape  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "data" / "basemap" / "conus_states.gpkg"
OUT = ROOT / "outputs_arkansas"
REV = OUT / "review"
TITLE = "Prescribed burns"
AC = 4046.86
RES = 20

SEV = [  # (upper dNBR bound, label, colour)
    (0.10, "Unburned / very low (dNBR < 0.10)", "#9ecae1"),
    (0.27, "Low (0.10 - 0.27)", "#ffe066"),
    (0.44, "Moderate-low (0.27 - 0.44)", "#fd8d3c"),
    (0.66, "Moderate-high (0.44 - 0.66)", "#e31a1c"),
    (np.inf, "High (> 0.66)", "#67000d"),
]
METHODS = {  # key: (layer, short name, how it works)
    "landsat": ("rule_landsat", "Landsat only",
                "Landsat 8/9 (30 m) burn index drop that is next to the recorded point and dated to the burn."),
    "sentinel": ("rule_v1", "Sentinel-2 only",
                 "Sentinel-2 (20 m) burn index drop that is next to the recorded point and dated to the burn."),
    "combined": ("rule_combo", "Sentinel-2 + Landsat combined",
                 "Both satellites in one time series (more clear looks). Each image is compared with land "
                 "far from the point, so seasonal change cancels out. No embeddings."),
    "best": ("rule_v2", "Best effort (Sentinel-2 + Landsat + embeddings)",
             "As the combined method, but each pixel is compared only with look-alike land (same "
             "vegetation type, from TESSERA satellite embeddings of the year before), which removes "
             "more green-up / dry-down noise."),
}
TOPO = ctx.providers.USGS.USTopo  # OSM's tile servers block scripted downloads
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{date}"
FIRMS_ARCHIVE = "https://firms.modaps.eosdis.nasa.gov/data/country/{s}/{y}/{s}_{y}_United_States.csv"
FIRMS_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
VIIRS_DAYS = (-1, 2)  # detections from 1 day before to 2 days after the recorded date


# ----------------------------------------------------------------------------- data
def load(key):
    f = json.loads((OUT / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    ds = xr.open_dataset(OUT / key / "layers.nc", decode_coords="all").load()
    mp = OUT / key / "methods.nc"
    info = {}
    if mp.exists():
        m = xr.open_dataset(mp, decode_coords="all").load()
        info = json.loads(m.attrs.get("info", "{}"))
        ds = ds.merge(m.drop_vars("spatial_ref", errors="ignore"), compat="override")
    return f, log, ds, info


def vectorize(mask, values, transform, epsg):
    geoms, vals = [], []
    for g, v in features.shapes(values.astype("int16"), mask=mask, transform=transform):
        geoms.append(shape(g))
        vals.append(int(v))
    return gpd.GeoDataFrame({"cls": vals}, geometry=geoms, crs=epsg)


def outlines(ds, epsg):
    """One dissolved polygon per method (WGS84) and its acres; None where nothing was mapped."""
    T = ds.rio.transform()
    out = {}
    for k, (layer, name, _) in METHODS.items():
        if layer not in ds:
            out[k] = None
            continue
        m = ds[layer].values.astype(bool)
        if not m.any():
            out[k] = gpd.GeoDataFrame({"method": [], "acres": []}, geometry=[], crs=4326)
            continue
        g = vectorize(m, m, T, epsg).dissolve().reset_index(drop=True)[["geometry"]]
        g["method"], g["acres"] = name, m.sum() * RES * RES / AC
        out[k] = g.to_crs(4326)
    return out


def severity(ds, epsg):
    main = ds["rule_v2"].values.astype(bool)
    dn = np.nan_to_num(ds["dnbr_adj"].values, nan=0.0)
    cls = np.digitize(dn, [b for b, _, _ in SEV[:-1]])
    sev = vectorize(main, cls, ds.rio.transform(), epsg)
    if len(sev):
        sev = sev.dissolve("cls").reset_index()
        sev["severity"] = [SEV[c][1] for c in sev.cls]
        sev["acres"] = [((cls == c) & main).sum() * RES * RES / AC for c in sev.cls]
        sev = sev.to_crs(4326)
    stats = {"mapped_acres": main.sum() * RES * RES / AC,
             "dnbr_median": float(np.median(dn[main])) if main.any() else None,
             "sev_acres": {SEV[c][1]: ((cls == c) & main).sum() * RES * RES / AC for c in range(len(SEV))}}
    return sev, stats


def confidence(f, log, stats):
    ratio = stats["mapped_acres"] / f["reported_acres"]
    gap = log.get("first_clear_after_days_merged")
    flags = []
    if stats["mapped_acres"] < 1 / AC * 1e4:
        flags.append("not detected (< 1 ha mapped)")
    elif ratio < 1 / 3:
        flags.append("mapped area < 1/3 of reported")
    elif ratio > 2:
        flags.append("mapped area > 2x reported")
    if gap is None:
        flags.append("no clear image after the burn")
    elif gap > 14:
        flags.append(f"first clear image {gap} days after burn")
    if not str(log.get("strata", "")).startswith("tessera"):
        flags.append("green-up correction without embeddings")
    if stats["mapped_acres"] < 1 / AC * 1e4 or ratio < 0.2 or gap is None or gap > 21:
        level = "low"
    elif 0.5 <= ratio <= 2 and gap <= 10:
        level = "high"
    else:
        level = "medium"
    return level, ratio, gap, flags


def viirs_archive(fires):
    """Burns from past years without a FIRMS key: one pass over each NASA FIRMS yearly US archive
    (no key needed) keeps detections within ~30 km of every burn from that year."""
    if os.environ.get("FIRMS_MAP_KEY"):
        return
    todo = {k: f for k, f in fires.items() if not (OUT / "firms" / f"{k}_viirs.csv").exists()
            and int(f["start"][:4]) < pd.Timestamp.now().year}
    for y in sorted({f["start"][:4] for f in todo.values()}):
        keep = {k: [] for k, f in todo.items() if f["start"][:4] == y}
        ok = False
        for s in ("viirs-snpp", "viirs-noaa20", "viirs-noaa21"):
            try:
                for c in pd.read_csv(FIRMS_ARCHIVE.format(s=s, y=y), chunksize=500_000):
                    ok = True
                    for k in keep:
                        f = fires[k]
                        m = ((c.latitude - f["lat"]).abs() < 0.3) & ((c.longitude - f["lon"]).abs() < 0.3)
                        if m.any():
                            keep[k].append(c[m].assign(source=s))
            except Exception as e:
                print(f"VIIRS archive {s} {y} unavailable: {e.__class__.__name__}", flush=True)
        if not ok:
            continue
        (OUT / "firms").mkdir(parents=True, exist_ok=True)
        for k, parts in keep.items():
            df = pd.concat(parts) if parts else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
            df.to_csv(OUT / "firms" / f"{k}_viirs.csv", index=False)


def viirs(key, f):
    """VIIRS 375 m active-fire detections near the point, from the FIRMS area API (needs a free
    MAP_KEY in FIRMS_MAP_KEY) or the yearly archive (see viirs_archive), cached in
    <output folder>/firms/<key>_viirs.csv.
    Returns (points in WGS84 or None if no data source, note)."""
    firms_dir = OUT / "firms"
    p = firms_dir / f"{key}_viirs.csv"
    mk = os.environ.get("FIRMS_MAP_KEY")
    if not p.exists() and mk:
        bbox = f"{f['lon'] - 0.3:.4f},{f['lat'] - 0.3:.4f},{f['lon'] + 0.3:.4f},{f['lat'] + 0.3:.4f}"
        d0 = pd.Timestamp(f["start"]) + pd.Timedelta(days=VIIRS_DAYS[0])
        days = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days + VIIRS_DAYS[1] - VIIRS_DAYS[0] + 1
        parts = []
        for src in FIRMS_SOURCES:
            r = requests.get(FIRMS_API.format(key=mk, src=src, bbox=bbox, days=min(days, 10), date=f"{d0:%Y-%m-%d}"),
                             timeout=120)
            if r.ok and r.text.startswith("latitude"):
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                if len(df):
                    df["source"] = src
                    parts.append(df)
        df = pd.concat(parts) if parts else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
        if len(df):  # SP replaces NRT for the same satellite and time
            df = df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
        firms_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    if not p.exists():
        return None, (f"{f['start'][:4]} VIIRS detections not loaded yet. NASA\n"
                      "publishes the free yearly archive after the year ends;\n"
                      "until then set FIRMS_MAP_KEY (free key from\n"
                      "firms.modaps.eosdis.nasa.gov/api) and rerun.")
    df = pd.read_csv(p)
    if df.empty:
        return gpd.GeoDataFrame({"acq_date": []}, geometry=[], crs=4326), "no detections"
    d = pd.to_datetime(df.acq_date)
    df = df[(d >= pd.Timestamp(f["start"]) + pd.Timedelta(days=VIIRS_DAYS[0]))
            & (d <= pd.Timestamp(f["end"]) + pd.Timedelta(days=VIIRS_DAYS[1]))]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    return g, f"{len(g)} detections within ~30 km"


# ----------------------------------------------------------------------------- map furniture
def utm_of(f):
    return gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326).estimate_utm_crs()


def extent_ll(f, outs, pts):
    epsg = utm_of(f)
    p = gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326).to_crs(epsg).iloc[0]
    geoms = [p.buffer(1100)]
    for g in outs.values():
        if g is not None and len(g):
            geoms.append(g.to_crs(epsg).geometry.iloc[0].intersection(p.buffer(2500)))
    if pts is not None and len(pts):
        geoms += list(pts.to_crs(epsg).buffer(190).intersection(p.buffer(2500)))
    b = gpd.GeoSeries(geoms, crs=epsg).union_all().bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = max(b[2] - b[0], b[3] - b[1]) / 2 * 1.12
    bx = gpd.GeoSeries.from_xy([cx - half, cx + half], [cy - half, cy + half], crs=epsg).to_crs(4326)
    return bx.x.min(), bx.x.max(), bx.y.min(), bx.y.max()


def setup_ll_axes(ax, ext, lat, north=True, nticks=5):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians(lat)))
    ax.grid(color="k", alpha=0.25, lw=0.4, ls="--")
    ax.tick_params(labelsize=7, top=True, right=True, labeltop=True, labelright=False)
    fmt = matplotlib.ticker.FuncFormatter
    ax.xaxis.set_major_formatter(fmt(lambda v, _: f"{v:.3f}°"))
    ax.yaxis.set_major_formatter(fmt(lambda v, _: f"{v:.3f}°"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.add_artist(ScaleBar(111320 * np.cos(np.radians(lat)), "m", location="lower right",
                           length_fraction=0.25, box_alpha=0.8, font_properties={"size": 8}))
    if north:
        ax.annotate("N", xy=(0.96, 0.97), xytext=(0.96, 0.89), xycoords="axes fraction", ha="center",
                    va="center", fontsize=12, fontweight="bold",
                    arrowprops=dict(facecolor="k", width=4, headwidth=12),
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.7))


def road_basemap(ax):
    try:
        ctx.add_basemap(ax, crs=4326, source=TOPO, attribution="USGS The National Map")
    except Exception as e:  # map is still usable without tiles
        ax.text(0.5, 0.02, f"basemap unavailable: {e}"[:90], transform=ax.transAxes, ha="center", fontsize=6)


def draw_point(ax, f, label=True):
    ax.plot(f["lon"], f["lat"], marker="*", ms=17, mfc="#00e5ff", mec="k", mew=1.2, zorder=8)
    if label:
        ax.annotate(f"Recorded point\n{f['lat']:.4f}, {f['lon']:.4f}", (f["lon"], f["lat"]),
                    xytext=(10, -22), textcoords="offset points", fontsize=7.5, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=0.5, alpha=0.85))


def search_circle(f):
    pt = gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326)
    return pt.to_crs(pt.estimate_utm_crs()).buffer(1000).to_crs(4326)


def draw_outline(ax, g, colour="k", halo="white"):
    if g is not None and len(g):
        g.boundary.plot(ax=ax, color=halo, lw=3.2, zorder=6)
        g.boundary.plot(ax=ax, color=colour, lw=1.4, zorder=7)


def states_around(lons, lats):
    """States containing the points (lower 48), the map window around them, and a title."""
    st = gpd.read_file(STATES)
    pts = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=4326)
    home = st[st.intersects(pts.union_all().buffer(0.01))]
    if not len(home):  # outside the lower 48: frame the points
        b = [min(lons) - 3, min(lats) - 2, max(lons) + 3, max(lats) + 2]
        return st, home, b, ""
    b = home.total_bounds
    pad = max(b[2] - b[0], b[3] - b[1]) * 0.08
    return st, home, [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], ", ".join(home.name)


def inset(fig, f):
    ax = fig.add_axes([0.73, 0.06, 0.22, 0.20])
    st, home, b, name = states_around([f["lon"]], [f["lat"]])
    st.cx[b[0]:b[2], b[1]:b[3]].plot(ax=ax, fc="#f2f2f2", ec="#999", lw=0.4)
    if len(home):
        home.plot(ax=ax, fc="#dddddd", ec="k", lw=0.8)
    ax.plot(f["lon"], f["lat"], "r*", ms=11, mec="k", mew=0.5)
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect(1 / np.cos(np.radians((b[1] + b[3]) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(name, fontsize=8)


def burn_title(f):
    return " - ".join(x for x in (f.get("managed_area"), f.get("burn_unit")) if x)


def header(fig, f, n, title):
    d = pd.Timestamp(f["start"]).strftime("%-d %b %Y")
    if f["end"] != f["start"]:
        d += " to " + pd.Timestamp(f["end"]).strftime("%-d %b %Y")
    parts = [f"Map {n} of 5: {title}", f"prescribed burn {d}", f.get("burn_type"),
             f"reported {f['reported_acres']:.0f} ac"]
    fig.text(0.06, 0.955, burn_title(f), fontsize=15, fontweight="bold")
    fig.text(0.06, 0.925, "  |  ".join(p for p in parts if p), fontsize=9.5)


def footnote(fig, extra=""):
    fig.text(0.02, 0.015,
             "Satellite estimate from a recorded point, date and acres (free data); not field-verified. "
             "Prescribed burns under canopy often read as unburned even where surface fuels burned.\n"
             + (extra + " " if extra else "") + "Coordinates WGS84 decimal degrees.", fontsize=6.5, color="#333")


def side_text(fig, handles, title, lines, legend_top=0.88):
    ax = fig.add_axes([0.71, 0.30, 0.28, legend_top - 0.30])
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, title=title,
                    title_fontsize=8.5, alignment="left")
    leg._legend_box.align = "left"
    ax.text(0, 0.42, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8)


def acres_lines(f, g):
    if g is None:
        return ["Mapped: not run"]
    a = float(g.acres.sum()) if len(g) else 0.0
    if a < 2.5:
        return [f"Reported: {f['reported_acres']:.0f} ac    Mapped: NOT DETECTED"]
    return [f"Reported: {f['reported_acres']:.0f} ac    Mapped: {a:.0f} ac  ({a / f['reported_acres']:.0%} of reported)"]


def wrap(s, n=54):
    import textwrap
    return textwrap.wrap(s, n)


STD_HANDLES = [Line2D([], [], color="#444", lw=1, ls="--", label="1 km search radius around point"),
               Line2D([], [], marker="*", ls="", ms=13, mfc="#00e5ff", mec="k", label="Recorded lat/long")]


# ----------------------------------------------------------------------------- map pages
def false_colour_ll(ds, prefix, epsg):
    """Before and after SWIR-NIR-red composites reprojected to WGS84, one shared stretch."""
    arrs = {}
    for tag in ("pre", "post"):
        bands = [f"{prefix}{tag}_B12", f"{prefix}{tag}_B8A", f"{prefix}{tag}_B04"]
        if not all(b in ds for b in bands):
            return None
        da = xr.concat([ds[b] for b in bands], "band").astype("float32")
        da = da.rio.write_crs(epsg).rio.write_nodata(np.nan).rio.reproject("EPSG:4326", nodata=np.nan)
        arrs[tag] = da
    stack = np.concatenate([a.values.reshape(3, -1) for a in arrs.values()], axis=1)
    if not np.isfinite(stack).any():
        return {t: (np.full(a.shape[1:] + (3,), 0.6), a) for t, a in arrs.items()}
    lo = np.nanpercentile(stack, 2, axis=1)
    hi = np.nanpercentile(stack, 98, axis=1)
    out = {}
    for t, a in arrs.items():
        img = np.clip((np.moveaxis(a.values, 0, -1) - lo) / (hi - lo), 0, 1)
        img = np.where(np.isnan(img).any(-1, keepdims=True), 0.6, img)
        out[t] = (img, a)
    return out


def date_list(dates, last):
    d = dates[-3:] if last else dates[:3]
    if not d:
        return "no clear image"
    return ", ".join(pd.Timestamp(x).strftime("%-d %b %Y") for x in d)


def page_before_after(f, log, ds, info, outs, ext, n, which):
    sat = "Landsat 8/9" if which == "landsat" else "Sentinel-2"
    prefix = "ls_" if which == "landsat" else ""
    g = outs[which]
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, f, n, f"{sat} before and after the burn")
    imgs = false_colour_ll(ds, prefix, log["epsg"])
    db = info.get("landsat_dates_before" if which == "landsat" else "s2_dates_before", [])
    da = info.get("landsat_dates_after" if which == "landsat" else "s2_dates_after", [])
    for i, (tag, ttl, dates) in enumerate([("pre", "Before", date_list(db, True)), ("post", "After", date_list(da, False))]):
        ax = fig.add_axes([0.07 + i * 0.32, 0.14, 0.31, 0.72])
        setup_ll_axes(ax, ext, f["lat"], north=(i == 1), nticks=3)
        if imgs is not None:
            img, a = imgs[tag]
            x, y = a.x.values, a.y.values
            dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
            ax.imshow(img, extent=[x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy],
                      interpolation="nearest", zorder=1)
        else:
            ax.text(0.5, 0.5, f"no {sat} images", transform=ax.transAxes, ha="center")
        search_circle(f).boundary.plot(ax=ax, color="white", lw=0.9, ls="--", zorder=4)
        draw_outline(ax, g, colour="k" if tag == "pre" else "#00e5ff", halo="white" if tag == "pre" else "k")
        draw_point(ax, f, label=False)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"{ttl} the burn\n{dates}", fontsize=9)
        ax.set_xlabel("Longitude (WGS84)", fontsize=8)
        ax.set_ylabel("Latitude (WGS84)" if i == 0 else "", fontsize=8)
        ax.tick_params(labelleft=(i == 0))
    handles = [Line2D([], [], color="#00e5ff", lw=2, label=f"Burn outline, {METHODS[which][1]} method"),
               Line2D([], [], color="#888", lw=1, ls="--", label="1 km search radius around point"),
               STD_HANDLES[1]]
    lines = acres_lines(f, g) + [""]
    lines += [f"Images: {len(db)} before, {len(da)} after (searched 60 days",
              "before and 45 days after). Each panel is the per-pixel",
              "median of the 3 clearest looks closest to the burn;",
              "dates above are those looks. Flat grey = no clear view.", "", "How the outline is drawn:"]
    lines += wrap(METHODS[which][2])
    lines += ["", "False colour (SWIR-NIR-red): healthy vegetation green,",
              "fresh burn scar dark red / brown / purple."]
    side_text(fig, handles, f"{sat} false colour", lines)
    inset(fig, f)
    footnote(fig, f"{sat} surface reflectance ({'30 m resampled to 20 m' if which == 'landsat' else '20 m'}).")
    return fig


def page_road(f, n, title, ext, draw, handles, legend_title, lines, note=""):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.06, 0.08, 0.62, 0.80])
    setup_ll_axes(ax, ext, f["lat"])
    road_basemap(ax)
    draw(ax)
    search_circle(f).boundary.plot(ax=ax, color="#444", lw=0.9, ls="--", zorder=4)
    draw_point(ax, f)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Longitude (WGS84)", fontsize=8)
    ax.set_ylabel("Latitude (WGS84)", fontsize=8)
    header(fig, f, n, title)
    side_text(fig, handles, legend_title, lines)
    inset(fig, f)
    footnote(fig, note)
    return fig, ax


def page_combined(f, log, info, outs, ext):
    g = outs["combined"]

    def draw(ax):
        if g is not None and len(g):
            g.plot(ax=ax, color="#fd8d3c", alpha=0.45, lw=0, zorder=3)
        draw_outline(ax, g)

    handles = [mpatches.Patch(fc="#fd8d3c", alpha=0.6, ec="k", lw=1.2, label="Mapped burned area")] + STD_HANDLES
    nb, na = len(info.get("s2_dates_before", [])), len(info.get("s2_dates_after", []))
    lb, la = len(info.get("landsat_dates_before", [])), len(info.get("landsat_dates_after", []))
    lines = acres_lines(f, g) + [f"Images: {nb} + {na} Sentinel-2, {lb} + {la} Landsat (before + after)",
                                 f"First clear image after burn: {log.get('first_clear_after_days_merged')} days",
                                 "", "How the outline is drawn:"] + wrap(METHODS["combined"][2])
    fig, _ = page_road(f, 3, "Sentinel-2 + Landsat combined", ext, draw, handles, "Combined method", lines,
                       "Combined Sentinel-2 + Landsat burn index, point + date match.")
    return fig


def page_viirs(f, pts, note, ext):
    def draw(ax):
        if pts is None:
            ax.text(0.5, 0.88, "VIIRS detections not available yet\n(see note at right)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, zorder=10,
                    bbox=dict(boxstyle="round", fc="white", ec="k", alpha=0.9))
        elif len(pts):
            utm = utm_of(f)
            sq = pts.to_crs(utm).buffer(187.5, cap_style=3).to_crs(4326)
            gpd.GeoDataFrame(pts.drop(columns="geometry"), geometry=sq).plot(
                ax=ax, fc="#ff2200", ec="yellow", lw=0.9, alpha=0.5, zorder=5)
            pts.plot(ax=ax, color="k", markersize=6, zorder=6)
        else:
            ax.text(0.5, 0.06, "No VIIRS detection in this window", transform=ax.transAxes, ha="center",
                    fontsize=10, zorder=10, bbox=dict(boxstyle="round", fc="white", ec="k", alpha=0.9))

    handles = [mpatches.Patch(fc="#ff2200", ec="yellow", alpha=0.6,
                              label="VIIRS fire detection (375 m pixel footprint)")] + STD_HANDLES
    d0 = pd.Timestamp(f["start"]) + pd.Timedelta(days=VIIRS_DAYS[0])
    d1 = pd.Timestamp(f["end"]) + pd.Timedelta(days=VIIRS_DAYS[1])
    lines = [f"Window: {d0:%-d %b} - {d1:%-d %b %Y}"]
    if pts is None:
        lines += note.split("\n")
    else:
        ins = 0
        if len(pts):
            utm = utm_of(f)
            p = gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326).to_crs(utm).iloc[0]
            ins = int((pts.to_crs(utm).distance(p) <= 1375).sum())
        lines += [f"Detections within 1 km of the point (+ half a pixel): {ins}",
                  f"All detections loaded: {note}"]
        if len(pts):
            lines += ["Dates: " + ", ".join(sorted({pd.Timestamp(x).strftime('%-d %b') for x in pts.acq_date}))]
    lines += ["", "VIIRS (Suomi-NPP / NOAA-20 / NOAA-21) sees heat from",
              "active flames when a satellite passes (about twice a day",
              "per satellite). It confirms fire was present, not how far",
              "it spread; small, cool or short burns are often missed."]
    fig, _ = page_road(f, 4, "VIIRS active-fire detections (NASA FIRMS)", ext, draw, handles,
                       "VIIRS active fire", lines, "VIIRS 375 m active-fire product, NASA FIRMS.")
    return fig


def page_best(f, log, outs, sev, stats, conf, ext):
    level, ratio, gap, flags = conf
    g = outs["best"]

    def draw(ax):
        if len(sev):
            sev.plot(ax=ax, color=[SEV[c][2] for c in sev.cls], alpha=0.6, lw=0, zorder=3)
        draw_outline(ax, g)

    handles = [mpatches.Patch(fc=c, ec="k", lw=0.4, label=f"{lab}: {stats['sev_acres'][lab]:.0f} ac")
               for _, lab, c in SEV]
    handles += [Line2D([], [], color="k", lw=2, label="Mapped burn perimeter")] + STD_HANDLES
    lines = acres_lines(f, g)
    lines += [f"Images: {log.get('s2_looks')} Sentinel-2 + {log.get('landsat_looks')} Landsat",
              (f"First clear image after burn: {gap} day{'s' if gap != 1 else ''}" if gap is not None
               else "No clear image after burn"), f"Confidence: {level.upper()}"]
    lines += [f"  - {x}" for x in flags] + ["", "How the outline is drawn:"] + wrap(METHODS["best"][2])
    fig, _ = page_road(f, 5, "best effort (adds embeddings)", ext, draw, handles,
                       "Burn severity (satellite index)", lines,
                       "Severity = green-up-corrected dNBR, Key & Benson (2006) classes, not calibrated to field "
                       "(CBI) plots.")
    return fig


# ----------------------------------------------------------------------------- exports
def kml_poly(folder, geom, name, desc, fill, line="ff000000", width=2, visible=True):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        k = folder.newpolygon(name=name, description=desc,
                              outerboundaryis=list(p.exterior.coords),
                              innerboundaryis=[list(i.coords) for i in p.interiors])
        k.style.polystyle.color = fill
        k.style.linestyle.color = line
        k.style.linestyle.width = width
        k.visibility = int(visible)


def hex_kml(c, alpha):
    c = c.lstrip("#")
    return f"{alpha:02x}{c[4:6]}{c[2:4]}{c[0:2]}"


KML_COLOURS = {"landsat": "#2ca02c", "sentinel": "#ff00ff", "combined": "#fd8d3c", "best": "#ffff00"}
PAGES = ["1_landsat", "2_sentinel", "3_combined", "4_viirs", "5_best_effort"]


def main(out=None, title=None):
    global OUT, REV, TITLE
    if out is not None:
        OUT, REV = Path(out), Path(out) / "review"
    if title:
        TITLE = title
    REV.mkdir(parents=True, exist_ok=True)
    for p in [*REV.glob("*.png"), *REV.glob("*.pdf"), *REV.glob("*.kmz"), *REV.glob("*.gpkg")]:
        p.unlink()
    fires = json.loads((OUT / "fires.json").read_text())
    viirs_archive(fires)
    kml = simplekml.Kml(name=f"{TITLE} - satellite burn maps")
    pts_rec, layers, sevs, vpts, rows = [], {k: [] for k in METHODS}, [], [], []
    for key, f in fires.items():
        if not (OUT / key / "run_log.json").exists():
            err = OUT / key / "error.txt"
            why = err.read_text()[:200] if err.exists() else "not run"
            print(key, "no maps:", why)
            rows.append({"key": key, "managed_area": f.get("managed_area", ""), "burn_unit": f.get("burn_unit", ""),
                         "burn_date": f["start"], "reported_acres": f["reported_acres"], "lat": f["lat"],
                         "lon": f["lon"], "status": f"no maps: {why}"})
            continue
        f, log, ds, info = load(key)
        epsg = log["epsg"]
        outs = outlines(ds, epsg)
        sev, stats = severity(ds, epsg)
        conf = confidence(f, log, stats)
        level, ratio, gap, flags = conf
        vp, vnote = viirs(key, f)
        ext = extent_ll(f, outs, vp)
        figs = [page_before_after(f, log, ds, info, outs, ext, 1, "landsat"),
                page_before_after(f, log, ds, info, outs, ext, 2, "sentinel"),
                page_combined(f, log, info, outs, ext),
                page_viirs(f, vp, vnote, ext),
                page_best(f, log, outs, sev, stats, conf, ext)]
        with PdfPages(REV / f"{key}.pdf") as pdf:
            for name, fig in zip(PAGES, figs):
                fig.savefig(REV / f"{key}_{name}.png", dpi=170)
                pdf.savefig(fig)
                plt.close(fig)

        ac = {k: (None if g is None else round(float(g.acres.sum()) if len(g) else 0.0, 1)) for k, g in outs.items()}
        info_row = {"key": key, "status": "ok", "managed_area": f.get("managed_area", ""),
                    "burn_unit": f.get("burn_unit", ""), "burn_date": f["start"],
                    "reported_acres": f["reported_acres"],
                    "ac_landsat": ac["landsat"], "ac_sentinel": ac["sentinel"], "ac_combined": ac["combined"],
                    "ac_best_effort": ac["best"],
                    "viirs_detections": None if vp is None else len(vp),
                    "best_effort_confidence": level, "best_effort_flags": "; ".join(flags),
                    "first_clear_image_days_after": gap, "s2_images": log.get("s2_looks"),
                    "landsat_images": log.get("landsat_looks"),
                    "median_dnbr_in_best_effort": None if stats["dnbr_median"] is None else round(stats["dnbr_median"], 3),
                    **{f"best_ac_{lab.split(' (')[0].lower().replace(' / ', '_').replace('-', '_').replace(' ', '_')}":
                       round(a, 1) for lab, a in stats["sev_acres"].items()},
                    "lat": f["lat"], "lon": f["lon"], "burn_type": f.get("burn_type", "")}
        rows.append(info_row)
        pts_rec.append({**{k: info_row[k] for k in ("key", "managed_area", "burn_unit", "burn_date", "reported_acres")},
                        "geometry": Point(f["lon"], f["lat"])})
        for k, g in outs.items():
            if g is not None and len(g):
                g = g.copy()
                g["key"], g["burn_date"] = key, f["start"]
                if k == "best":
                    g["confidence"], g["reported_acres"] = level, f["reported_acres"]
                layers[k].append(g)
        if len(sev):
            s = sev.copy()
            s["key"] = key
            sevs.append(s)
        if vp is not None and len(vp):
            v = vp.copy()
            v["key"] = key
            vpts.append(v)

        fol = kml.newfolder(name=f"{burn_title(f)} ({f['start']})")
        desc = (f"Burn date: {f['start']}<br>Reported: {f['reported_acres']:.0f} ac<br>"
                f"Best effort: {stats['mapped_acres']:.0f} ac<br>Confidence: {level}<br>"
                + "<br>".join(flags) + f"<br>Recorded point: {f['lat']:.4f}, {f['lon']:.4f}")
        p = fol.newpoint(name=f"{burn_title(f)} (recorded point)", description=desc,
                         coords=[(f["lon"], f["lat"])])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"
        if len(sev):
            fs = fol.newfolder(name="Best effort: burn severity (satellite index)")
            for _, r in sev.iterrows():
                kml_poly(fs, r.geometry, r.severity, f"{r.acres:.1f} ac", hex_kml(SEV[r.cls][2], 0xa0),
                         line="00000000", width=0)
        for k in ["best", "combined", "sentinel", "landsat"]:
            g = outs[k]
            if g is None or not len(g):
                continue
            vis = k == "best"
            fk = fol.newfolder(name=f"{METHODS[k][1]} outline" + ("" if vis else " (off by default)"))
            fk.visibility = int(vis)
            kml_poly(fk, g.geometry.iloc[0], METHODS[k][1], f"{g.acres.iloc[0]:.0f} ac", "00000000",
                     line=hex_kml(KML_COLOURS[k], 0xff), width=3 if vis else 2, visible=vis)
        if vp is not None and len(vp):
            fv = fol.newfolder(name="VIIRS detections (off by default)")
            fv.visibility = 0
            for _, r in vp.iterrows():
                q = fv.newpoint(name=f"VIIRS {r.acq_date}", coords=[(r.longitude, r.latitude)])
                q.visibility = 0
        print(key, level, ac, "viirs:", vnote.split("\n")[0][:40], flush=True)

    kml.savekmz(str(REV / "burns.kmz"))
    gpkg = REV / "burns.gpkg"
    if pts_rec:
        gpd.GeoDataFrame(pts_rec, crs=4326).to_file(gpkg, layer="recorded_points", driver="GPKG")
    names = {"landsat": "landsat_only", "sentinel": "sentinel2_only", "combined": "combined_s2_landsat",
             "best": "best_effort"}
    for k, parts in layers.items():
        if parts:
            gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326).to_file(gpkg, layer=names[k], driver="GPKG")
    if sevs:
        gpd.GeoDataFrame(pd.concat(sevs, ignore_index=True), crs=4326).to_file(gpkg, layer="best_effort_severity",
                                                                               driver="GPKG")
    if vpts:
        gpd.GeoDataFrame(pd.concat(vpts, ignore_index=True), crs=4326).to_file(gpkg, layer="viirs_detections",
                                                                               driver="GPKG")
    df = pd.DataFrame(rows)
    df.to_csv(REV / "summary.csv", index=False)
    summary_pdf(df)


def summary_table(fig, df, rect):
    cols = ["#", "Site - unit", "Date", "Reported ac", "Landsat ac", "Sentinel-2 ac", "Combined ac",
            "Best effort ac", "VIIRS hits", "Confidence"]
    fmt = lambda v: "-" if v is None or pd.isna(v) else f"{v:.0f}"  # noqa: E731
    cells = []
    for i, r in df.iterrows():
        name = burn_title({"managed_area": str(r.managed_area or "")[:30], "burn_unit": str(r.burn_unit or "")[:16]})
        if r.status != "ok":
            cells.append([i + 1, name or r.key, r.burn_date, fmt(r.reported_acres), "-", "-", "-", "-", "-", "FAILED"])
            continue
        cells.append([i + 1, name or r.key, r.burn_date, fmt(r.reported_acres),
                      fmt(r.ac_landsat), fmt(r.ac_sentinel), fmt(r.ac_combined), fmt(r.ac_best_effort),
                      "n/a" if pd.isna(r.viirs_detections) else f"{r.viirs_detections:.0f}",
                      r.best_effort_confidence])
    tb = fig.add_axes(rect)
    tb.axis("off")
    t = tb.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left",
                 colWidths=[0.04, 0.32, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.08])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)


def summary_pdf(df):
    df = df.reset_index(drop=True)
    if "status" not in df:
        df["status"] = "ok"
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.40, 0.42, 0.50])
    st, home, b, _ = states_around(list(df.lon), list(df.lat))
    st.cx[b[0]:b[2], b[1]:b[3]].plot(ax=ax, fc="#f7f7f7", ec="#bbb", lw=0.4)
    if len(home):
        home.plot(ax=ax, fc="#eeeeee", ec="k", lw=0.8)
    col = {"high": "#1a9850", "medium": "#fdae61", "low": "#d73027"}
    for i, r in df.iterrows():
        c = col.get(r.get("best_effort_confidence"), "#999999")
        ax.plot(r.lon, r.lat, "o", ms=10 if len(df) <= 30 else 6, mfc=c, mec="k")
        if len(df) <= 30:
            ax.annotate(str(i + 1), (r.lon, r.lat), xytext=(7, 4), textcoords="offset points", fontsize=10,
                        fontweight="bold")
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect(1 / np.cos(np.radians((b[1] + b[3]) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=f"{k} confidence (best effort)")
                       for k, c in col.items()], loc="lower left", fontsize=8, frameon=False)
    fig.text(0.03, 0.95, f"{TITLE}: satellite burn maps (no perimeters used)", fontsize=14, fontweight="bold")
    first, per_page = 12, 34
    summary_table(fig, df.iloc[:first], [0.03, 0.05, 0.94, 0.30])
    notes = ("How to read these maps\n\n"
             "- Each burn has a 5-page PDF (also as PNGs):\n"
             "  1 Landsat before/after, 2 Sentinel-2 before/after,\n"
             "  3 Sentinel-2 + Landsat combined, 4 VIIRS active fire,\n"
             "  5 best effort (adds embeddings) with burn severity.\n"
             "- Every map starts from the recorded lat/long (star) and\n  burn date; no burn perimeters were used.\n"
             "- Mapped acres can differ from reported acres because\n  patchy or light under-canopy fire is hard to see.\n"
             "- Google Earth: open burns.kmz. GIS: burns.gpkg.\n"
             "- summary.csv has every number in this table and more.\n\n"
             "What would help most: your unit or burn perimeters, and\nnotes on where the maps are right or wrong.")
    fig.text(0.50, 0.88, notes, fontsize=9.5, va="top")
    with PdfPages(REV / "summary.pdf") as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        for s in range(first, len(df), per_page):
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.03, 0.95, f"{TITLE}: summary (continued)", fontsize=12, fontweight="bold")
            summary_table(fig, df.iloc[s:s + per_page], [0.03, 0.05, 0.94, 0.86])
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else None, a[1] if len(a) > 1 else None)
