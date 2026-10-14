"""Score each mapping rule against the reference perimeters and make figures.

Metrics are chosen for imperfect references (unit boundaries, patchy burns):
  * detected           : >= 1 ha of mapped burn inside the reference unit
  * pct_unit_mapped    : share of the unit mapped as burned (an estimate, NOT a miss rate)
  * false_outside_ha   : mapped burn outside the unit AND outside every other recorded
                         burn of that year (1-pixel edge tolerance) -> the most trustworthy
                         error number, since Rx fires rarely escape
  * other_burns_ha     : mapped burn that falls on OTHER recorded burns (real fire, wrong event)
  * precision_vs_target: in-unit mapped / all mapped
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib_scalebar.scalebar import ScaleBar  # noqa: E402

ALBERS = 5070
SHORT_NAME = {"konza_k1b": "Konza Prairie, KS",
              "grsm_short_bunk": "Great Smoky Mtns, NC",
              "seki_big_stump": "Kings Canyon NP, CA"}


def draw_locator(ax, fires, highlight=None):
    states = gpd.read_file(ROOT / "data" / "basemap" / "conus_states.gpkg").to_crs(ALBERS)
    states.plot(ax=ax, color="0.92", edgecolor="0.6", lw=0.4)
    for k, f in fires.items():
        p = gpd.GeoSeries.from_xy([f["lon"]], [f["lat"]], crs=4326).to_crs(ALBERS).iloc[0]
        on = highlight is None or k == highlight
        ax.plot(p.x, p.y, marker="*", ms=16 if on else 8, color="red" if on else "0.5", mec="k", mew=0.6)
        if highlight is None:
            ax.annotate(SHORT_NAME[k], (p.x, p.y), xytext=(8, 8), textcoords="offset points", fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))
    ax.set_axis_off()

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
OUT = ROOT / "outputs"
RES = 20
RULES = ["A_wall_to_wall", "B_point_guided", "C_point_date", "D_point_date_embed"]
RULE_LABEL = {"A_wall_to_wall": "Standard: any darkening anywhere (dNBR > 0.1)",
              "B_point_guided": "Near the point: unusual darkening touching the point",
              "C_point_date": "Near the point + darkened on the recorded date",
              "D_point_date_embed": "Same + embedding boost for weak signal"}


def other_dates(key, other):
    if key == "konza_k1b":
        return pd.to_datetime(other["DATE"].str.strip().replace("", None), format="%m/%d/%Y", errors="coerce")
    if key == "grsm_short_bunk":
        return pd.to_datetime(other["FireDiscoveryDateTime"], unit="ms", errors="coerce")
    return pd.to_datetime(other["START_DATE"], unit="ms", errors="coerce")


def rasterize(gdf, like, transform):
    if gdf is None or len(gdf) == 0:
        return np.zeros(like.shape, bool)
    return features.rasterize([(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                              out_shape=like.shape, transform=transform, fill=0).astype(bool)


def auc(score, pos, neg):
    s = np.concatenate([score[pos], score[neg]])
    y = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    ok = np.isfinite(s)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], s[ok]))


def rgb(ds, tag, swir=False):
    bands = [f"{tag}_B12", f"{tag}_B8A", f"{tag}_B04"] if swir else [f"{tag}_B04", f"{tag}_B03", f"{tag}_B02"]
    a = np.dstack([ds[b].values for b in bands])
    lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def run(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    ds = xr.open_dataset(OUT / key / "layers.nc", decode_coords="all")
    epsg = log["epsg"]
    transform = ds.rio.transform()
    like = ds["dNBR"]

    tgt = gpd.read_file(REF / f"{key}_target.geojson").to_crs(epsg)
    oth = gpd.read_file(REF / f"{key}_other_burns.geojson").to_crs(epsg)
    oth["date"] = other_dates(key, oth)
    T = rasterize(tgt, like, transform)
    O = rasterize(oth, like, transform) & ~T
    edge = ndi.binary_dilation(T, iterations=1) & ~T
    bg = ~ndi.binary_dilation(T | O, iterations=5)  # clean background, 100 m from any recorded burn

    rows = []
    for r in RULES:
        if f"rule_{r}" not in ds:
            continue
        M = ds[f"rule_{r}"].values.astype(bool)
        inside = (M & T).sum() * RES * RES / 1e4
        outside_false = (M & ~T & ~O & ~edge).sum() * RES * RES / 1e4
        on_other = (M & O).sum() * RES * RES / 1e4
        total = M.sum() * RES * RES / 1e4
        union = (M | T).sum()
        rows.append(dict(fire=key, rule=r, mapped_ha=round(total, 1), detected=inside >= 1,
                         pct_unit_mapped=round(100 * inside / (T.sum() * RES * RES / 1e4), 1),
                         false_outside_ha=round(outside_false, 1),
                         other_burns_ha=round(on_other, 1),
                         precision_vs_target=round(100 * inside / total, 1) if total else None,
                         iou=round((M & T).sum() / union, 3) if union else None))
    metrics = pd.DataFrame(rows)

    # how well does each evidence layer separate the unit from clean background?
    ev = {}
    for name, sign in [("dNBR", 1), ("break_mag", 1), ("tessera_dist", 1), ("s1_vh_lr", 1), ("s1_vv_lr", 1)]:
        if name in ds:
            ev[name] = auc(sign * ds[name].values, T, bg)
    # timing: do pixels inside the unit break when the record says?
    dok = ds["date_ok"].values.astype(bool)
    rel = ds["break_rel_days"].values
    strong = ds["z_dNBR"].values >= 1.5
    timing = {
        "pct_unit_pixels_date_matched": round(100 * float(dok[T].mean()), 1),
        "pct_unit_pixels_strong_signal": round(100 * float(strong[T].mean()), 1),
        "pct_strong_unit_pixels_date_matched": round(100 * float(dok[T & strong].mean()), 1) if (T & strong).any() else None,
        "pct_strong_background_pixels_date_matched": round(100 * float(dok[bg & strong].mean()), 1) if (bg & strong).any() else None,
        "median_break_rel_days_strong_unit": float(np.nanmedian(rel[T & strong])) if (T & strong).any() else None,
    }
    # other recorded burns: is their change dated to *their* date or to ours?
    start = pd.Timestamp(f["start"])
    other_rows = []
    for _, o in oth.iterrows():
        m = rasterize(gpd.GeoDataFrame(geometry=[o.geometry], crs=epsg), like, transform) & ~T
        if m.sum() < 10 or pd.isna(o["date"]):
            continue
        dd = (o["date"].normalize() - start).days
        if abs(dd) > 60:
            continue
        other_rows.append(dict(days_from_target=dd, pixels=int(m.sum()),
                               pct_strong=round(100 * float(strong[m].mean()), 1),
                               pct_flagged_as_target_date=round(100 * float((dok & strong)[m].mean()), 1),
                               median_break_rel_days=float(np.nanmedian(rel[m & strong])) if (m & strong).any() else None))
    other_df = pd.DataFrame(other_rows)

    summary = dict(fire=key, label=f["label"], evidence_auc=ev, timing=timing,
                   other_burns_in_window=other_df.to_dict("records"))
    (OUT / key / "validation.json").write_text(json.dumps(summary, indent=2, default=str))
    metrics.to_csv(OUT / key / "metrics.csv", index=False)

    # ------------------------------------------------------------------ figure
    ext = [ds.x.min() - RES / 2, ds.x.max() + RES / 2, ds.y.min() - RES / 2, ds.y.max() + RES / 2]
    ext = [float(e) for e in ext]
    px, py = log["point_xy"]

    def overlay(ax, show_other=True):
        tgt.boundary.plot(ax=ax, color="cyan", lw=1.2)
        if show_other and len(oth):
            oth.boundary.plot(ax=ax, color="white", lw=0.6, ls="--")
        ax.plot(px, py, marker="*", color="yellow", ms=12, mec="k")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.add_artist(ScaleBar(1, "m", location="lower left", length_fraction=0.3,
                               box_alpha=0.75, font_properties={"size": 9}))

    fig, axs = plt.subplots(2, 4, figsize=(20, 10.5))
    axs = axs.ravel()
    axs[0].imshow(rgb(ds, "pre", swir=True), extent=ext); axs[0].set_title("Before (SWIR-NIR-red)")
    axs[1].imshow(rgb(ds, "post", swir=True), extent=ext); axs[1].set_title("After (SWIR-NIR-red)")
    im = axs[2].imshow(ds["dNBR"].values, extent=ext, cmap="RdYlBu_r", vmin=-0.2, vmax=0.4)
    axs[2].set_title("Burn signal (dNBR = before minus after)\nred = darkened / likely burned")
    plt.colorbar(im, ax=axs[2], fraction=0.046)
    br = np.where(strong, rel, np.nan)
    im = axs[3].imshow(br, extent=ext, cmap="twilight_shifted", vmin=-40, vmax=40)
    axs[3].set_title("When did it darken? days vs recorded date\n(only pixels with strong dNBR)")
    plt.colorbar(im, ax=axs[3], fraction=0.046)
    if "s1_vh_lr" in ds:
        im = axs[4].imshow(ds["s1_vh_lr"].values, extent=ext, cmap="PuOr", vmin=-3, vmax=3)
        axs[4].set_title("Sentinel-1 radar VH change (dB)"); plt.colorbar(im, ax=axs[4], fraction=0.046)
    else:
        axs[4].text(0.5, 0.5, "no Sentinel-1 pair", ha="center"); axs[4].set_title("Sentinel-1")
    if "tessera_dist" in ds:
        v = ds["tessera_dist"].values
        im = axs[5].imshow(v, extent=ext, cmap="magma", vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 99))
        axs[5].set_title(f"TESSERA embedding change {start.year - 1}->{start.year}"); plt.colorbar(im, ax=axs[5], fraction=0.046)
    A = ds["rule_A_wall_to_wall"].values.astype(bool)
    axs[6].imshow(rgb(ds, "post"), extent=ext)
    axs[6].imshow(np.where(A, 1, np.nan), extent=ext, cmap=ListedColormap(["red"]), alpha=0.55)
    axs[6].set_title("STANDARD METHOD (red): every pixel in the window\n"
                     "that darkened (dNBR > 0.1), ignoring point and date")
    C = ds["rule_C_point_date"].values.astype(bool)
    axs[7].imshow(rgb(ds, "post"), extent=ext)
    axs[7].imshow(np.where(C, 1, np.nan), extent=ext, cmap=ListedColormap(["orangered"]), alpha=0.65)
    if "rule_D_point_date_embed" in ds:
        D = ds["rule_D_point_date_embed"].values.astype(bool) & ~C
        axs[7].imshow(np.where(D, 1, np.nan), extent=ext, cmap=ListedColormap(["gold"]), alpha=0.65)
    axs[7].set_title("OUR METHOD (orange): unusual darkening, connected to the\n"
                     "reported point, AND darkened on the recorded burn date\n"
                     "(gold = extra pixels added by embeddings)")
    vi_p = ROOT / "data" / "firms" / f"{key}_viirs.csv"
    vi = pd.read_csv(vi_p) if vi_p.exists() and vi_p.stat().st_size > 10 else pd.DataFrame()
    if len(vi):
        from pyproj import Transformer
        vi["date"] = pd.to_datetime(vi.acq_date)
        vi = vi[(vi.date >= start - pd.Timedelta(days=1)) & (vi.date <= pd.Timestamp(f["end"]) + pd.Timedelta(days=2))]
        if len(vi):
            vx, vy = Transformer.from_crs(4326, epsg, always_xy=True).transform(vi.longitude.values, vi.latitude.values)
            for ax in (axs[2], axs[7]):
                ax.scatter(vx, vy, s=60, marker="^", facecolor="none", edgecolor="lime", lw=1.5)
    for i, ax in enumerate(axs):
        overlay(ax, show_other=i not in (0, 1))
    fig.suptitle(f"{f['label']}   ({f['lat']:.3f}\u00b0N, {abs(f['lon']):.3f}\u00b0W)\n"
                 f"recorded {f['start']} to {f['end']}, {f['reported_acres']:.0f} ac.  "
                 "Cyan = reference unit, dashed white = other recorded burns, yellow star = simulated input point, "
                 "green triangles = VIIRS detections", fontsize=12)
    fig.tight_layout(rect=(0.07, 0, 1, 1))
    loc = fig.add_axes([0.0, 0.86, 0.085, 0.12])
    draw_locator(loc, json.loads((REF / "fires.json").read_text()), highlight=key)
    loc.set_title("Location", fontsize=9)
    fig.savefig(OUT / key / f"{key}_panels.png", dpi=110)
    plt.close(fig)
    print(metrics.to_string(index=False))
    print(json.dumps({"evidence_auc": ev, "timing": timing}, indent=1))
    if len(other_df):
        print(other_df.sort_values("days_from_target").to_string(index=False))
    return metrics


def summary_plot(all_m):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    fires = list(dict.fromkeys(all_m.fire))
    w = 0.2
    for j, r in enumerate(RULES):
        sub = all_m[all_m.rule == r].set_index("fire").reindex(fires)
        x = np.arange(len(fires)) + (j - 1.5) * w
        axs[0].bar(x, sub.pct_unit_mapped, w, label=RULE_LABEL[r])
        axs[1].bar(x, sub.false_outside_ha, w)
        axs[2].bar(x, sub.other_burns_ha, w)
    titles = ["% of reference unit mapped as burned\n(higher = less omission; units are patchy)",
              "False detections outside unit (ha)\n(not on any recorded burn; lower = better)",
              "Mapped area on OTHER recorded burns (ha)\n(real fire, wrong event)"]
    for ax, t in zip(axs, titles):
        ax.set_xticks(np.arange(len(fires)))
        ax.set_xticklabels([s.replace("_", "\n", 1) for s in fires])
        ax.set_title(t)
    axs[1].set_yscale("symlog", linthresh=10)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "summary_tradeoff.png", dpi=110)


if __name__ == "__main__":
    keys = sys.argv[1:] or ["konza_k1b", "grsm_short_bunk", "seki_big_stump"]
    ms = [run(k) for k in keys]
    allm = pd.concat(ms)
    allm.to_csv(OUT / "metrics_all.csv", index=False)
    summary_plot(allm)
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_locator(ax, json.loads((REF / "fires.json").read_text()))
    ax.set_title("Pilot test fires", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "site_map.png", dpi=110)
