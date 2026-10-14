"""Can same-year TESSERA embeddings recognise a burned unit they were not trained on?

Year-over-year embedding change fails for units burned every year (nothing changes).
Here we instead train a classifier on the burn-year embedding itself:
  positives = OTHER recorded burns of that year in the window (never the target)
  negatives = pixels >= 100 m from any recorded burn, outside a 1.5 km exclusion zone
              around the target
and test on the target unit vs. unburned pixels inside that exclusion zone.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import pipeline  # noqa: E402  (patched TESSERA tile finder)
import tessera  # noqa: E402
from validate import other_dates, rasterize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF, OUT = ROOT / "data" / "reference", ROOT / "outputs"


def run(key):
    f = json.loads((REF / "fires.json").read_text())[key]
    log = json.loads((OUT / key / "run_log.json").read_text())
    ds = xr.open_dataset(OUT / key / "layers.nc", decode_coords="all")
    epsg, transform, like = log["epsg"], ds.rio.transform(), ds["dNBR"]
    year = pd.Timestamp(f["start"]).year

    tgt = gpd.read_file(REF / f"{key}_target.geojson").to_crs(epsg)
    oth = gpd.read_file(REF / f"{key}_other_burns.geojson").to_crs(epsg)
    oth = oth[other_dates(key, oth).dt.year == year]
    T = rasterize(tgt, like, transform)
    O = rasterize(oth, like, transform) & ~T
    zone = ndi.binary_dilation(T, iterations=int(1500 / 20))
    clean_bg = ~ndi.binary_dilation(T | O, iterations=5)

    pos_tr, neg_tr = O & ~zone, clean_bg & ~zone
    pos_te, neg_te = T, clean_bg & zone
    res = dict(fire=key, year=year, train_pos_px=int(pos_tr.sum()), test_pos_px=int(pos_te.sum()))
    if pos_tr.sum() < 200:
        res["note"] = "too few other recorded burns in the window to train on"
        return res

    _, _, _, bb_ll = pipeline.make_window(f)
    pad = 0.002
    emb = tessera.read_aoi(bbox=(bb_ll[0] - pad, bb_ll[1] - pad, bb_ll[2] + pad, bb_ll[3] + pad), year=year)
    emb = emb.transpose("band", "y", "x").rio.write_crs(4326).rio.set_spatial_dims(x_dim="x", y_dim="y")
    emb = emb.rio.reproject_match(like.rio.write_crs(epsg), resampling=5)
    X = emb.values.reshape(emb.shape[0], -1).T
    ok = np.isfinite(X).all(1)

    def take(m, n, rng):
        idx = np.where(m.ravel() & ok)[0]
        return rng.choice(idx, size=min(n, len(idx)), replace=False)

    rng = np.random.default_rng(0)
    ip, ineg = take(pos_tr, 4000, rng), take(neg_tr, 8000, rng)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    clf.fit(np.vstack([X[ip], X[ineg]]), np.r_[np.ones(len(ip)), np.zeros(len(ineg))])
    tp, tn = np.where(pos_te.ravel() & ok)[0], np.where(neg_te.ravel() & ok)[0]
    p = clf.predict_proba(np.vstack([X[tp], X[tn]]))[:, 1]
    y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    res["auc_target_vs_nearby_unburned"] = round(float(roc_auc_score(y, p)), 3)
    res["pct_target_px_prob_gt_0.5"] = round(100 * float((p[: len(tp)] > 0.5).mean()), 1)
    res["pct_nearby_unburned_px_prob_gt_0.5"] = round(100 * float((p[len(tp):] > 0.5).mean()), 1)
    prob = np.full(X.shape[0], np.nan)
    prob[ok] = clf.predict_proba(X[ok])[:, 1]
    ds_out = xr.Dataset({"embed_burn_prob": (("y", "x"), prob.reshape(like.shape))},
                        coords={"y": like.y, "x": like.x})
    ds_out.to_netcdf(OUT / key / "embedding_prob.nc")
    return res


if __name__ == "__main__":
    keys = sys.argv[1:] or ["konza_k1b", "grsm_short_bunk", "seki_big_stump"]
    rows = [run(k) for k in keys]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "embedding_experiment.csv", index=False)
