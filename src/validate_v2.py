"""Score round-2 variants against the reference perimeters, summarise, and draw figures.

Same metrics as round 1 (see validate.py). Each unit is also described automatically
by ESA WorldCover 2020 (free, 10 m): share of the unit that is tree cover, so results
can be grouped into open (<30% trees), mixed (30-70%) and forest (>70%).
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import xarray as xr
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib_scalebar.scalebar import ScaleBar  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from validate import ALBERS, rasterize, rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference_v2"
OUT = ROOT / "outputs_v2"
RES = 20
VARIANTS = ["std", "v1", "v1_snap", "v2", "v2_snap"]
LABEL = {"std": "Standard (any darkening, dNBR > 0.1)",
         "v1": "Round 1 (Sentinel-2, point + date)",
         "v1_snap": "Round 1 + snapping to control lines",
         "v2": "Round 2 (+ Landsat + green-up correction)",
         "v2_snap": "Round 2 + snapping to control lines"}
SHORT = {"std": "Standard", "v1": "Round 1", "v1_snap": "Round 1\n+ snap", "v2": "Round 2",
         "v2_snap": "Round 2\n+ snap"}
COLOR = {"std": "0.55", "v1": "tab:orange", "v1_snap": "tab:brown", "v2": "tab:red", "v2_snap": "tab:purple"}
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def tree_share(cat, ds, T, epsg):
    bb = ds.rio.transform_bounds(4326)
    items = cat.search(collections=["esa-worldcover"], bbox=bb, datetime="2020-01-01/2020-12-31").item_collection()
    wc = odc.stac.load(items, bands=["map"], geobox=ds.odc.geobox, resampling="mode")["map"].max("time").values
    classes = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
               80: "water", 90: "wetland", 95: "mangrove"}
    v = wc[T]
    shares = {name: round(100 * float((v == c).mean()), 1) for c, name in classes.items()}
    return shares


def score(key, cat):
    f = json.loads((REF / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    ds = xr.open_dataset(OUT / key / "layers.nc", decode_coords="all")
    epsg, transform = log["epsg"], ds.rio.transform()
    like = ds["dnbr_s2"]
    tgt = gpd.read_file(REF / f"{key}_target.geojson").to_crs(epsg)
    oth = gpd.read_file(REF / f"{key}_other_burns.geojson").to_crs(epsg)
    T = rasterize(tgt, like, transform)
    O = rasterize(oth, like, transform) & ~T
    edge = ndi.binary_dilation(T, iterations=1) & ~T
    unit_ha = T.sum() * RES * RES / 1e4
    try:
        cover = tree_share(cat, ds, T, epsg)
    except Exception as e:
        print(key, "land cover lookup failed:", repr(e))
        cover = {}
    tree = cover.get("tree", np.nan)
    cover_class = "open" if tree < 30 else "mixed" if tree <= 70 else "forest" if tree == tree else "unknown"
    rows = []
    for v in VARIANTS:
        M = ds[f"rule_{v}"].values.astype(bool)
        inside = (M & T).sum() * RES * RES / 1e4
        total = M.sum() * RES * RES / 1e4
        rows.append(dict(fire=key, region=f["region"], cover_class=cover_class, tree_pct=tree,
                         wetland_pct=cover.get("wetland", np.nan), unit_ha=round(unit_ha, 1),
                         variant=v, mapped_ha=round(total, 1), detected=bool(inside >= 1),
                         pct_unit_mapped=round(100 * inside / unit_ha, 1),
                         false_outside_ha=round((M & ~T & ~O & ~edge).sum() * RES * RES / 1e4, 1),
                         other_burns_ha=round((M & O).sum() * RES * RES / 1e4, 1),
                         iou=round((M & T).sum() / max((M | T).sum(), 1), 3),
                         first_clear_after_s2=log.get("first_clear_after_days_s2"),
                         first_clear_after_merged=log.get("first_clear_after_days_merged"),
                         landsat_looks=log.get("landsat_looks"), osm_lines=log.get("osm_lines")))
    figure(key, f, log, ds, tgt, oth, epsg, cover_class, tree)
    return rows


def figure(key, f, log, ds, tgt, oth, epsg, cover_class, tree):
    ext = [float(ds.x.min()) - RES / 2, float(ds.x.max()) + RES / 2,
           float(ds.y.min()) - RES / 2, float(ds.y.max()) + RES / 2]
    px, py = log["point_xy"]
    fig, axs = plt.subplots(1, 4, figsize=(22, 6.3))

    def finish(ax):
        tgt.boundary.plot(ax=ax, color="cyan", lw=1.4)
        ax.plot(px, py, marker="*", color="yellow", ms=13, mec="k")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        ax.add_artist(ScaleBar(1, "m", location="lower left", length_fraction=0.3, box_alpha=0.75,
                               font_properties={"size": 9}))

    axs[0].imshow(rgb(ds, "post", swir=True), extent=ext)
    axs[0].set_title("After the burn (SWIR-NIR-red)")
    im = axs[1].imshow(ds["dnbr_s2"].values, extent=ext, cmap="RdYlBu_r", vmin=-0.2, vmax=0.4)
    axs[1].set_title("Round 1 burn signal (Sentinel-2 dNBR)\nred = darkened")
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    im = axs[2].imshow(ds["dnbr_adj"].values, extent=ext, cmap="RdYlBu_r", vmin=-0.2, vmax=0.4)
    axs[2].set_title("Round 2 burn signal: darkening compared with\nlook-alike unburned pixels (green-up removed)")
    plt.colorbar(im, ax=axs[2], fraction=0.046)
    axs[3].imshow(rgb(ds, "post"), extent=ext)
    lines = ds["lines"].values.astype(bool)
    axs[3].imshow(np.where(lines, 1, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6)
    for v, ls in [("v1", "-"), ("v2", "-"), ("v2_snap", "--")]:
        m = ds[f"rule_{v}"].values
        if m.any():
            axs[3].contour(ds.x.values, ds.y.values, m, levels=[0.5], colors=[COLOR[v]],
                           linewidths=1.8, linestyles=ls)
    axs[3].set_title("Mapped burns. Light grey = control lines\n(roads, trails, streams, ridges)")
    axs[3].legend(handles=[Line2D([], [], color="cyan", lw=1.4, label="Reference unit"),
                           Line2D([], [], color=COLOR["v1"], lw=1.8, label=LABEL["v1"]),
                           Line2D([], [], color=COLOR["v2"], lw=1.8, label=LABEL["v2"]),
                           Line2D([], [], color=COLOR["v2_snap"], lw=1.8, ls="--", label=LABEL["v2_snap"])],
                  loc="upper right", fontsize=7.5, framealpha=0.85)
    oth.boundary.plot(ax=axs[3], color="white", lw=0.6, ls=":")
    for ax in axs:
        finish(ax)
    fig.suptitle(f"{f['label']}  ({f['lat']:.3f}\u00b0N, {abs(f['lon']):.3f}\u00b0W)   recorded {f['start']} to "
                 f"{f['end']}, {f['reported_acres']:.0f} ac.   Unit is {tree:.0f}% tree cover ({cover_class}).   "
                 "Yellow star = simulated input point", fontsize=11.5)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    loc = fig.add_axes([0.0, 0.78, 0.06, 0.16])
    states = gpd.read_file(ROOT / "data" / "basemap" / "conus_states.gpkg").to_crs(ALBERS)
    states.plot(ax=loc, color="0.92", edgecolor="0.6", lw=0.3)
    p = gpd.GeoSeries.from_xy([f["lon"]], [f["lat"]], crs=4326).to_crs(ALBERS).iloc[0]
    loc.plot(p.x, p.y, marker="*", color="red", ms=12, mec="k")
    loc.set_axis_off()
    fig.savefig(OUT / key / f"{key}_panels.png", dpi=100)
    plt.close(fig)


def summary(df):
    df["solid"] = df.pct_unit_mapped >= 10
    g = df.groupby("variant").agg(fires=("fire", "nunique"), detected=("detected", "sum"),
                                  solid_detection_ge10pct=("solid", "sum"),
                                  median_pct_unit=("pct_unit_mapped", "median"),
                                  median_iou=("iou", "median"),
                                  total_false_outside_ha=("false_outside_ha", "sum"),
                                  fires_with_false_gt_5ha=("false_outside_ha", lambda s: int((s > 5).sum())),
                                  total_on_other_burns_ha=("other_burns_ha", "sum")).reindex(VARIANTS)
    by_cover = df.pivot_table(index="cover_class", columns="variant", values="pct_unit_mapped",
                              aggfunc="median").reindex(columns=VARIANTS)
    iou_cover = df.pivot_table(index="cover_class", columns="variant", values="iou",
                               aggfunc="median").reindex(columns=VARIANTS)
    n_cover = df[df.variant == "v1"].groupby("cover_class").fire.count()
    return g, by_cover, iou_cover, n_cover


def summary_figure(df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fires = df[df.variant == "v1"].sort_values(["cover_class", "tree_pct"]).fire.tolist()
    xs = np.arange(len(VARIANTS))
    for ax, col, title in [(axs[0], "pct_unit_mapped", "% of reference unit mapped as burned"),
                           (axs[1], "iou", "Overlap with reference unit (IoU, 0-1)"),
                           (axs[2], "false_outside_ha", "False detections outside the unit (ha)")]:
        for fire in fires:
            sub = df[df.fire == fire].set_index("variant").reindex(VARIANTS)
            cc = sub.cover_class.iloc[0]
            c = {"open": "goldenrod", "mixed": "olivedrab", "forest": "darkgreen"}.get(cc, "grey")
            ax.plot(xs, sub[col].values, "-o", color=c, alpha=0.45, ms=4)
        med = df.groupby("variant")[col].median().reindex(VARIANTS)
        ax.plot(xs, med.values, "-s", color="k", lw=2.5, ms=8, label="median")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[v] for v in VARIANTS], fontsize=9)
        ax.set_title(title)
    axs[2].set_yscale("symlog", linthresh=1)
    axs[2].set_ylim(0, None)
    fig.text(0.5, 0.01, "Standard = any darkening (dNBR > 0.1) anywhere.   Round 1 = Sentinel-2, near the point, "
             "darkened on the recorded date.   Round 2 = Round 1 + Landsat + green-up correction with look-alike "
             "pixels.   + snap = boundaries snapped to roads, trails, streams, ridges.", ha="center", fontsize=9)
    axs[0].legend(handles=[Line2D([], [], color=c, marker="o", label=l) for c, l in
                           [("goldenrod", "open (<30% trees)"), ("olivedrab", "mixed"),
                            ("darkgreen", "forest (>70% trees)"), ("k", "median")]], fontsize=8)
    fig.suptitle(f"Round 2: {df.fire.nunique()} prescribed fires. Each line is one fire.", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "summary_round2.png", dpi=110)


def site_map():
    fires = json.loads((REF / "fires.json").read_text())
    states = gpd.read_file(ROOT / "data" / "basemap" / "conus_states.gpkg").to_crs(ALBERS)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    states.plot(ax=ax, color="0.93", edgecolor="0.6", lw=0.4)
    for k, f in fires.items():
        p = gpd.GeoSeries.from_xy([f["lon"]], [f["lat"]], crs=4326).to_crs(ALBERS).iloc[0]
        ax.plot(p.x, p.y, marker="*", color="red", ms=13, mec="k", mew=0.5)
        name = f["label"].split(" - ")[0] if k.startswith("nps_") else f["label"].split(" - ")[0].split(",")[0]
        ax.annotate(name[:28], (p.x, p.y), xytext=(5, 4), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax.set_axis_off()
    ax.set_title(f"Round 2 test fires (n = {len(fires)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "site_map_round2.png", dpi=110)


if __name__ == "__main__":
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    keys = [k for k in json.loads((REF / "fires.json").read_text()) if (OUT / k / "run_log.json").exists()]
    rows = []
    for k in keys:
        try:
            rows += score(k, cat)
        except Exception as e:
            print(k, "scoring failed", repr(e))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_round2.csv", index=False)
    g, by_cover, iou_cover, n_cover = summary(df)
    pd.set_option("display.width", 200)
    print(g.to_string())
    print("\nmedian % of unit mapped, by cover class\n", by_cover.round(1).to_string())
    print("\nmedian IoU, by cover class\n", iou_cover.round(3).to_string())
    print("\nfires per cover class\n", n_cover.to_string())
    g.to_csv(OUT / "summary_round2.csv")
    summary_figure(df)
    site_map()
