"""Arkansas test: reviewer-ready maps and files from the round-2 outputs.

For each burn (outputs_arkansas/review/):
  <key>.pdf          p1 road map, p2 aerial imagery, p3 satellite evidence + alternative outlines
  <key>_roads.png, <key>_imagery.png
All burns:
  arkansas_burns.kmz   Google Earth: perimeter, severity, recorded point, alternatives (off)
  arkansas_burns.gpkg  GIS layers: recorded_points, perimeters, severity, alternatives
  summary.csv, summary.pdf

Main product = round-2 polygon without snapping ("v2"). Severity = green-up-corrected
dNBR inside that polygon, Key & Benson (2006) breaks. Not field-calibrated.
Confidence rule (fixed before results were seen):
  low    not detected, mapped < 0.2 x reported, or first clear look > 21 days after burn
  high   mapped 0.5-2 x reported and first clear look <= 10 days after burn
  medium otherwise
"""
import json
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
import rioxarray  # noqa: E402,F401
import simplekml  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib_scalebar.scalebar import ScaleBar  # noqa: E402
from rasterio import features  # noqa: E402
from shapely.geometry import Point, shape  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from validate import rgb  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "arkansas"
OUT = ROOT / "outputs_arkansas"
REV = OUT / "review"
AC = 4046.86
RES = 20

SEV = [  # (upper dNBR bound, label, colour)
    (0.10, "Unburned / very low (dNBR < 0.10)", "#9ecae1"),
    (0.27, "Low (0.10 - 0.27)", "#ffe066"),
    (0.44, "Moderate-low (0.27 - 0.44)", "#fd8d3c"),
    (0.66, "Moderate-high (0.44 - 0.66)", "#e31a1c"),
    (np.inf, "High (> 0.66)", "#67000d"),
]
RULES = {  # name shown to reviewer, colour
    "v2": ("Main method (round 2)", "#000000"),
    "v2_snap": ("Round 2 snapped to roads/streams/ridges", "#00bcd4"),
    "v1": ("Round 1 (Sentinel-2 only)", "#ff00ff"),
    "v1_snap": ("Round 1 snapped", "#8e24aa"),
    "std": ("Standard: any darkening, ignores point/date", "#ffeb3b"),
}
TOPO = ctx.providers.USGS.USTopo  # OSM's tile servers block scripted downloads
IMAGERY = ctx.providers.Esri.WorldImagery
ESRI_ROADS = ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
              "World_Transportation/MapServer/tile/{z}/{y}/{x}")
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
               "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")


# ----------------------------------------------------------------------------- data
def load(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    ds = xr.open_dataset(OUT / key / "layers.nc", decode_coords="all").load()
    return f, log, ds


def vectorize(mask, values, transform, epsg):
    geoms, vals = [], []
    for g, v in features.shapes(values.astype("int16"), mask=mask, transform=transform):
        geoms.append(shape(g))
        vals.append(int(v))
    return gpd.GeoDataFrame({"cls": vals}, geometry=geoms, crs=epsg)


def products(key, f, log, ds):
    epsg, T = log["epsg"], ds.rio.transform()
    alts = []
    for r in RULES:
        m = ds[f"rule_{r}"].values.astype(bool)
        if m.any():
            g = vectorize(m, m, T, epsg).dissolve().reset_index(drop=True)[["geometry"]]
            g["rule"], g["method"], g["acres"] = r, RULES[r][0], m.sum() * RES * RES / AC
            alts.append(g)
    alts = (gpd.GeoDataFrame(pd.concat(alts, ignore_index=True), crs=epsg) if alts
            else gpd.GeoDataFrame({"rule": [], "method": [], "acres": []}, geometry=[], crs=epsg))
    main = ds["rule_v2"].values.astype(bool)
    dn = np.nan_to_num(ds["dnbr_adj"].values, nan=0.0)
    cls = np.digitize(dn, [b for b, _, _ in SEV[:-1]])
    sev = vectorize(main, cls, T, epsg)
    if len(sev):
        sev = sev.dissolve("cls").reset_index()
        sev["severity"] = [SEV[c][1] for c in sev.cls]
        sev["acres"] = [((cls == c) & main).sum() * RES * RES / AC for c in sev.cls]
    stats = {"mapped_acres": main.sum() * RES * RES / AC,
             "dnbr_median": float(np.median(dn[main])) if main.any() else None,
             "sev_acres": {SEV[c][1]: ((cls == c) & main).sum() * RES * RES / AC for c in range(len(SEV))}}
    return alts, sev, stats


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


# ----------------------------------------------------------------------------- map pages
def extent_ll(f, alts):
    pt = gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326)
    epsg = pt.estimate_utm_crs()
    p = pt.to_crs(epsg).iloc[0]
    geoms = [p.buffer(1100)]
    main = alts[alts.rule == "v2"]
    if len(main):
        geoms.append(main.to_crs(epsg).geometry.iloc[0])
    b = gpd.GeoSeries(geoms, crs=epsg).union_all().bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = max(b[2] - b[0], b[3] - b[1]) / 2 * 1.12
    box = gpd.GeoSeries.from_xy([cx - half, cx + half], [cy - half, cy + half], crs=epsg).to_crs(4326)
    return box.x.min(), box.x.max(), box.y.min(), box.y.max()


def setup_ll_axes(ax, ext, lat):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians(lat)))
    ax.grid(color="k", alpha=0.25, lw=0.4, ls="--")
    ax.tick_params(labelsize=7, top=True, right=True, labeltop=True, labelright=False)
    fmt = matplotlib.ticker.FuncFormatter
    ax.xaxis.set_major_formatter(fmt(lambda v, _: f"{v:.3f}°"))
    ax.yaxis.set_major_formatter(fmt(lambda v, _: f"{v:.3f}°"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.add_artist(ScaleBar(111320 * np.cos(np.radians(lat)), "m", location="lower right",
                           length_fraction=0.25, box_alpha=0.8, font_properties={"size": 8}))
    ax.annotate("N", xy=(0.96, 0.97), xytext=(0.96, 0.89), xycoords="axes fraction", ha="center",
                va="center", fontsize=12, fontweight="bold",
                arrowprops=dict(facecolor="k", width=4, headwidth=12),
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.7))


def basemap(ax, which):
    try:
        if which == "roads":
            ctx.add_basemap(ax, crs=4326, source=TOPO, attribution="USGS The National Map")
        else:
            ctx.add_basemap(ax, crs=4326, source=IMAGERY, attribution="Esri, Maxar, Earthstar Geographics")
            ctx.add_basemap(ax, crs=4326, source=ESRI_ROADS, attribution=False)
            ctx.add_basemap(ax, crs=4326, source=ESRI_LABELS, attribution=False)
    except Exception as e:  # map is still usable without tiles
        ax.text(0.5, 0.02, f"basemap unavailable: {e}"[:90], transform=ax.transAxes, ha="center", fontsize=6)


def draw_point(ax, f, dark_bg):
    ax.plot(f["lon"], f["lat"], marker="*", ms=17, mfc="#00e5ff", mec="k", mew=1.2, zorder=6)
    ax.annotate(f"Recorded point\n{f['lat']:.4f}, {f['lon']:.4f}", (f["lon"], f["lat"]),
                xytext=(10, -22), textcoords="offset points", fontsize=7.5, zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=0.5, alpha=0.85))


def search_circle(f):
    pt = gpd.GeoSeries([Point(f["lon"], f["lat"])], crs=4326)
    utm = pt.estimate_utm_crs()
    return pt.to_crs(utm).buffer(1000).to_crs(4326)


def side_panel(fig, f, log, stats, level, ratio, gap, flags, sev):
    ax = fig.add_axes([0.71, 0.30, 0.28, 0.58])
    ax.axis("off")
    handles = [mpatches.Patch(fc=c, ec="k", lw=0.4, label=f"{lab}: {stats['sev_acres'][lab]:.0f} ac")
               for _, lab, c in SEV]
    handles += [Line2D([], [], color="k", lw=2, label="Mapped burn perimeter"),
                Line2D([], [], color="#444", lw=1, ls="--", label="1 km search radius around point"),
                Line2D([], [], marker="*", ls="", ms=13, mfc="#00e5ff", mec="k", label="Recorded lat/long")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False,
                    title="Burn severity (satellite index)", title_fontsize=8.5, alignment="left")
    leg._legend_box.align = "left"
    nd = "NOT DETECTED" if stats["mapped_acres"] < 2.5 else f"{stats['mapped_acres']:.0f} ac"
    txt = [f"Reported: {f['reported_acres']:.0f} ac    Mapped: {nd}"
           + (f"  ({ratio:.0%} of reported)" if stats["mapped_acres"] >= 2.5 else ""),
           f"Images: {log.get('s2_looks')} Sentinel-2 + {log.get('landsat_looks')} Landsat",
           (f"First clear image after burn: {gap} day{'s' if gap != 1 else ''}" if gap is not None
            else "No clear image after burn"),
           f"Confidence: {level.upper()}"]
    txt += [f"  - {x}" for x in flags]
    ax.text(0, 0.30, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8, family="DejaVu Sans")


def footnote(fig):
    fig.text(0.02, 0.015,
             "Satellite estimate from a recorded point, date and acres (Sentinel-2 + Landsat, free data); not field-verified. "
             "Severity = green-up-corrected dNBR, Key & Benson (2006) classes,\nnot calibrated to field (CBI) plots; "
             "prescribed burns under canopy often read as low/unburned even where surface fuels burned. "
             "Coordinates WGS84 decimal degrees.", fontsize=6.5, color="#333")


def inset(fig, f):
    ax = fig.add_axes([0.73, 0.06, 0.22, 0.20])
    st = gpd.read_file(ROOT / "data" / "basemap" / "conus_states.gpkg")
    ar = st[st.postal == "AR"]
    st.cx[-96:-88, 31.5:38].plot(ax=ax, fc="#f2f2f2", ec="#999", lw=0.4)
    ar.plot(ax=ax, fc="#dddddd", ec="k", lw=0.8)
    ax.plot(f["lon"], f["lat"], "r*", ms=11, mec="k", mew=0.5)
    ax.set_xlim(-95.0, -89.4)
    ax.set_ylim(32.8, 36.7)
    ax.set_aspect(1 / np.cos(np.radians(34.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Arkansas", fontsize=8)


def page_map(f, log, alts, sev, stats, conf, which):
    level, ratio, gap, flags = conf
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.06, 0.08, 0.62, 0.80])
    ext = extent_ll(f, alts)
    setup_ll_axes(ax, ext, f["lat"])
    basemap(ax, which)
    if len(sev):
        s = sev.to_crs(4326)
        s.plot(ax=ax, color=[SEV[c][2] for c in s.cls], alpha=0.6, lw=0, zorder=3)
    main = alts[alts.rule == "v2"]
    if len(main):
        m = main.to_crs(4326)
        halo = "white" if which == "imagery" else "#ffffff"
        m.boundary.plot(ax=ax, color=halo, lw=3.2, zorder=4)
        m.boundary.plot(ax=ax, color="k", lw=1.4, zorder=5)
    search_circle(f).boundary.plot(ax=ax, color="#ffffff" if which == "imagery" else "#444", lw=0.9, ls="--", zorder=4)
    draw_point(ax, f, which == "imagery")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Longitude (WGS84)", fontsize=8)
    ax.set_ylabel("Latitude (WGS84)", fontsize=8)
    d = pd.Timestamp(f["start"]).strftime("%-d %b %Y")
    fig.text(0.06, 0.955, f"{f['managed_area']} - {f['burn_unit']}", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.925, f"Prescribed burn {d}  |  {f['burn_type']}  |  reported {f['reported_acres']:.0f} ac  |  "
             f"burn boss {f['burn_boss']}  |  " + ("road map" if which == "roads" else "aerial imagery with roads"),
             fontsize=9.5)
    side_panel(fig, f, log, stats, level, ratio, gap, flags, sev)
    inset(fig, f)
    footnote(fig)
    return fig


def page_evidence(f, log, ds, alts):
    """Before/after Sentinel-2 (SWIR false colour: fresh burns show dark red/brown) + all outlines."""
    epsg = log["epsg"]
    x, y = ds.x.values, ds.y.values
    full = [x.min() - RES / 2, x.max() + RES / 2, y.min() - RES / 2, y.max() + RES / 2]
    px, py = log["point_xy"]
    main = alts[alts.rule == "v2"]
    b = main.total_bounds if len(main) else [px, py, px, py]
    half = max(1300, (b[2] - b[0]) / 2 * 1.3, (b[3] - b[1]) / 2 * 1.3)
    cx, cy = (b[0] + b[2]) / 2 if len(main) else px, (b[1] + b[3]) / 2 if len(main) else py
    win = [cx - half, cx + half, cy - half, cy + half]
    fig, axs = plt.subplots(1, 3, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.30, wspace=0.04)
    d = pd.Timestamp(f["start"])
    panels = [(rgb(ds, "pre", swir=True), "Sentinel-2 before burn"),
              (rgb(ds, "post", swir=True), "Sentinel-2 after burn"),
              (rgb(ds, "post", swir=True), "All method outlines (after-burn image)")]
    for ax, (img, title) in zip(axs, panels):
        ax.imshow(np.where(np.isnan(img), 0.6, img), extent=full)
        ax.plot(px, py, marker="*", ms=14, mfc="#00e5ff", mec="k", zorder=6)
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        ax.add_artist(ScaleBar(1, "m", location="lower left", length_fraction=0.3, box_alpha=0.8,
                               font_properties={"size": 7}))
    if len(main):
        main.boundary.plot(ax=axs[1], color="white", lw=1.4)
    for r in ["std", "v1", "v1_snap", "v2_snap", "v2"]:
        g = alts[alts.rule == r]
        if len(g):
            g.boundary.plot(ax=axs[2], color=RULES[r][1], lw=2.2 if r == "v2" else 1.2,
                            ls="-" if r in ("v2", "v2_snap") else "--", zorder=5)
    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
    handles = [Line2D([], [], color=RULES[r][1], lw=2, ls="-" if r in ("v2", "v2_snap") else "--",
                      label=f"{RULES[r][0]}: " + (f"{alts[alts.rule == r].acres.iloc[0]:.0f} ac"
                                                 if (alts.rule == r).any() else "nothing mapped"))
               for r in ["v2", "v2_snap", "v1", "v1_snap", "std"]]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.12), fontsize=8.5, frameon=False,
               title="Outlines from each method (for comparison only; the black outline is the main product)",
               title_fontsize=9)
    fig.text(0.02, 0.95, f"{f['managed_area']} - {f['burn_unit']}: satellite evidence and alternative outlines",
             fontsize=14, fontweight="bold")
    fig.text(0.02, 0.93, f"Burn date {d:%-d %b %Y}. False colour (SWIR-NIR-red): healthy vegetation green, "
             "fresh burn scar dark red/brown/purple. Images are medians of the 3 clearest looks before / after.\n"
             "Flat grey = no cloud-free Sentinel-2 view in the 45 days after the burn (Landsat was used there).",
             fontsize=8.5, va="top")
    fig.text(0.02, 0.04, f"Projection: UTM (EPSG:{epsg}); 20 m pixels. Black/white outline = main method. "
             "Star = recorded lat/long.", fontsize=7.5, color="#333")
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


def main():
    REV.mkdir(parents=True, exist_ok=True)
    fires = json.loads((REF / "fires.json").read_text())
    kml = simplekml.Kml(name="TNC Arkansas prescribed burns - satellite test maps")
    pts, perims, sevs, alts_all, rows = [], [], [], [], []
    for key, f in fires.items():
        if not (OUT / key / "run_log.json").exists():
            print(key, "no run output; skipped")
            continue
        f, log, ds = load(key)
        alts, sev, stats = products(key, f, log, ds)
        conf = confidence(f, log, stats)
        level, ratio, gap, flags = conf
        with PdfPages(REV / f"{key}.pdf") as pdf:
            for which in ("roads", "imagery"):
                fig = page_map(f, log, alts, sev, stats, conf, which)
                fig.savefig(REV / f"{key}_{which}.png", dpi=170)
                pdf.savefig(fig)
                plt.close(fig)
            fig = page_evidence(f, log, ds, alts)
            pdf.savefig(fig)
            plt.close(fig)

        info = {"key": key, "managed_area": f["managed_area"], "burn_unit": f["burn_unit"],
                "burn_date": f["start"], "reported_acres": f["reported_acres"],
                "mapped_acres": round(stats["mapped_acres"], 1), "mapped_vs_reported": round(ratio, 2),
                "confidence": level, "flags": "; ".join(flags),
                "first_clear_image_days_after": gap, "s2_images": log.get("s2_looks"),
                "landsat_images": log.get("landsat_looks"),
                "median_dnbr_in_perimeter": None if stats["dnbr_median"] is None else round(stats["dnbr_median"], 3),
                **{f"ac_{lab.split(' (')[0].lower().replace(' / ', '_').replace('-', '_').replace(' ', '_')}": round(a, 1)
                   for lab, a in stats["sev_acres"].items()},
                **{f"ac_{r}": round(float(alts[alts.rule == r].acres.sum()), 1) for r in RULES},
                "lat": f["lat"], "lon": f["lon"], "burn_boss": f["burn_boss"], "burn_type": f["burn_type"]}
        rows.append(info)
        pts.append({**{k: info[k] for k in ("key", "managed_area", "burn_unit", "burn_date", "reported_acres")},
                    "geometry": Point(f["lon"], f["lat"])})
        a4 = alts.to_crs(4326)
        a4["key"], a4["burn_date"] = key, f["start"]
        alts_all.append(a4)
        m = a4[a4.rule == "v2"].copy()
        if len(m):
            m["confidence"], m["reported_acres"] = level, f["reported_acres"]
            perims.append(m)
        if len(sev):
            s4 = sev.to_crs(4326)
            s4["key"] = key
            sevs.append(s4)

        fol = kml.newfolder(name=f"{f['managed_area']} - {f['burn_unit']} ({f['start']})")
        desc = (f"Burn date: {f['start']}<br>Reported: {f['reported_acres']:.0f} ac<br>"
                f"Mapped: {stats['mapped_acres']:.0f} ac<br>Confidence: {level}<br>"
                + "<br>".join(flags) + f"<br>Recorded point: {f['lat']:.4f}, {f['lon']:.4f}")
        p = fol.newpoint(name=f"{f['managed_area']} (recorded point)", description=desc,
                         coords=[(f["lon"], f["lat"])])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"
        if len(sev):
            fs = fol.newfolder(name="Burn severity (satellite index)")
            for _, r in s4.iterrows():
                kml_poly(fs, r.geometry, r.severity, f"{r.acres:.1f} ac", hex_kml(SEV[r.cls][2], 0xa0),
                         line="00000000", width=0)
        if len(m):
            kml_poly(fol, m.geometry.iloc[0], "Mapped burn perimeter (main method)", desc, "00000000",
                     line="ff00ffff", width=3)
        fa = fol.newfolder(name="Alternative outlines (off by default)")
        fa.visibility = 0
        for _, r in a4[a4.rule != "v2"].iterrows():
            kml_poly(fa, r.geometry, r.method, f"{r.acres:.0f} ac", "00000000",
                     line=hex_kml(RULES[r.rule][1], 0xff), width=2, visible=False)
        print(key, level, f"{stats['mapped_acres']:.0f}/{f['reported_acres']:.0f} ac", flags, flush=True)

    kml.savekmz(str(REV / "arkansas_burns.kmz"))
    gpkg = REV / "arkansas_burns.gpkg"
    gpkg.unlink(missing_ok=True)
    gpd.GeoDataFrame(pts, crs=4326).to_file(gpkg, layer="recorded_points", driver="GPKG")
    if perims:
        gpd.GeoDataFrame(pd.concat(perims, ignore_index=True), crs=4326).to_file(gpkg, layer="perimeters", driver="GPKG")
    if sevs:
        gpd.GeoDataFrame(pd.concat(sevs, ignore_index=True), crs=4326).to_file(gpkg, layer="severity", driver="GPKG")
    if alts_all:
        gpd.GeoDataFrame(pd.concat(alts_all, ignore_index=True), crs=4326).to_file(gpkg, layer="alternative_outlines",
                                                                                   driver="GPKG")
    df = pd.DataFrame(rows)
    df.to_csv(REV / "summary.csv", index=False)
    summary_pdf(df)


def summary_pdf(df):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.40, 0.42, 0.50])
    st = gpd.read_file(ROOT / "data" / "basemap" / "conus_states.gpkg")
    st[st.postal == "AR"].plot(ax=ax, fc="#eeeeee", ec="k", lw=0.8)
    col = {"high": "#1a9850", "medium": "#fdae61", "low": "#d73027"}
    for i, r in df.reset_index(drop=True).iterrows():
        ax.plot(r.lon, r.lat, "o", ms=10, mfc=col[r.confidence], mec="k")
        ax.annotate(str(i + 1), (r.lon, r.lat), xytext=(7, 4), textcoords="offset points", fontsize=10,
                    fontweight="bold")
    ax.set_aspect(1 / np.cos(np.radians(34.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=f"{k} confidence")
                       for k, c in col.items()], loc="lower left", fontsize=8, frameon=False)
    fig.text(0.03, 0.95, "TNC Arkansas prescribed burns: satellite mapping test (blind; no perimeters used)",
             fontsize=14, fontweight="bold")
    cols = ["#", "Preserve - unit", "Date", "Reported ac", "Mapped ac", "% of reported", "Low/very low ac",
            "Mod-high ac", "1st clear img (d)", "Confidence"]
    cells = []
    for i, r in df.reset_index(drop=True).iterrows():
        mod = r.get("ac_moderate_low", 0) + r.get("ac_moderate_high", 0) + r.get("ac_high", 0)
        low = r.get("ac_unburned_very_low", 0) + r.get("ac_low", 0)
        cells.append([i + 1, f"{r.managed_area[:30]} - {r.burn_unit[:16]}", r.burn_date, f"{r.reported_acres:.0f}",
                      f"{r.mapped_acres:.0f}", f"{r.mapped_vs_reported:.0%}", f"{low:.0f}", f"{mod:.0f}",
                      r.first_clear_image_days_after, r.confidence])
    tb = fig.add_axes([0.03, 0.05, 0.94, 0.30])
    tb.axis("off")
    t = tb.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left",
                 colWidths=[0.03, 0.33, 0.09, 0.07, 0.07, 0.08, 0.09, 0.08, 0.09, 0.08])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)
    notes = ("How to read these maps\n\n"
             "- Each burn has a 3-page PDF: road map, aerial imagery,\n  and satellite before/after with alternative outlines.\n"
             "- Black outline = burned area mapped from satellite, starting\n  from the recorded lat/long (star) and burn date.\n"
             "- Colours inside = burn severity index (dNBR), not field-\n  calibrated. Low values are expected under canopy.\n"
             "- Mapped acres can differ from reported acres because\n  patchy or light under-canopy fire is hard to see.\n"
             "- Google Earth: open arkansas_burns.kmz. GIS: arkansas_burns.gpkg.\n\n"
             "What would help most: your unit or burn perimeters, and\nnotes on where the maps are right or wrong.")
    fig.text(0.50, 0.88, notes, fontsize=9.5, va="top")
    fig.savefig(REV / "summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
