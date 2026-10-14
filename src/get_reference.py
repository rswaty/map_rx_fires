"""Download the three reference prescribed-fire perimeters, the other recorded
burns near each one (same year), and build a simulated "TNC-style" input record
(approximate point + recorded dates + reported acres) for each fire.

The pipeline only ever sees the simulated input record. The perimeters are used
afterwards for validation.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
REF.mkdir(parents=True, exist_ok=True)

KONZA = "https://services3.arcgis.com/NLHBHJt7gvEcsduu/arcgis/rest/services/GIS097/FeatureServer/0"
NPS = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
       "NPS_National_Wildland_Fire_Perimeters_FLAT_view/FeatureServer/0")
CALRX = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/"
         "California_Prescribed_Fire_Perimeters_Public/FeatureServer/0")

# Max offset applied to the simulated point, mimicking approximate locations
# in burn databases.
POINT_JITTER_M = 400
SEED = 42


def query(url, where, extra=None):
    params = {"where": where, "outFields": "*", "outSR": 4326, "f": "geojson"}
    if extra:
        params.update(extra)
    r = requests.get(f"{url}/query", params=params, timeout=120)
    r.raise_for_status()
    gj = r.json()
    if not gj.get("features"):
        raise RuntimeError(f"no features for {where}")
    return gpd.GeoDataFrame.from_features(gj["features"], crs=4326)


def envelope(gdf, buffer_m=12000):
    utm = gdf.estimate_utm_crs()
    b = gdf.to_crs(utm).buffer(buffer_m).to_crs(4326).total_bounds
    return {"geometry": ",".join(map(str, b)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"}


def simulated_point(target, rng):
    utm = target.estimate_utm_crs()
    c = target.to_crs(utm).geometry.union_all().centroid
    ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0) * POINT_JITTER_M
    p = Point(c.x + dist * np.cos(ang), c.y + dist * np.sin(ang))
    p_ll = gpd.GeoSeries([p], crs=utm).to_crs(4326).iloc[0]
    return p_ll.y, p_ll.x, float(dist)


def get_states_basemap():
    """Lower-48 state outlines (Natural Earth) for the locator maps."""
    out = ROOT / "data" / "basemap" / "conus_states.gpkg"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip")
    g = g[(g.iso_a2 == "US") & ~g.postal.isin(["AK", "HI"])][["name", "postal", "geometry"]]
    g.to_file(out)


def main():
    get_states_basemap()
    rng = np.random.default_rng(SEED)
    fires = {}

    # 1) Konza Prairie watershed K1B, tallgrass prairie, burned 2019-03-22.
    kz = query(KONZA, "1=1")
    tgt = kz[kz["NAME_1"] == "K1B"].copy()
    other = kz[kz["NAME_1"] != "K1B"].copy()
    fires["konza_k1b"] = dict(
        label="Konza Prairie K1B (KS) - tallgrass prairie",
        source="Konza Prairie LTER GIS097 (watersheds burned 2019)",
        start="2019-03-22", end="2019-03-22",
        reported_acres=float(tgt["HECTARES_1"].sum() * 2.47105),
        ecosystem="grassland", target=tgt, other=other)

    # 2) Great Smoky Mountains, Cataloochee Short Bunk Rx, 2019-04-17, DGPS-mapped.
    nps = query(NPS, "FeatureCategory='Treatment' AND FireCalendarYear=2019 AND UNITCODE='GRSM'")
    tgt = nps[nps["IncidentName"] == "Cataloochee Short Bunk RX"].copy()
    other = nps[nps["IncidentName"] != "Cataloochee Short Bunk RX"].copy()
    fires["grsm_short_bunk"] = dict(
        label="Great Smoky Mtns - Cataloochee Short Bunk Rx (NC) - oak/pine forest underburn",
        source="NPS National Wildland Fire Perimeters (MAPMETHOD=Differential GPS)",
        start="2019-04-17", end="2019-04-17",
        reported_acres=float(tgt["GISAcres"].sum()),
        ecosystem="eastern hardwood-pine forest", target=tgt, other=other)

    # 3) Kings Canyon NP, Big Stump East Rx, 2025-06-23..28, closed-canopy mixed conifer.
    ca = query(CALRX, "OBJECTID=274")
    tgt = ca.copy()
    other = query(CALRX, "YEAR_>=2024 AND OBJECTID<>274", envelope(tgt))
    fires["seki_big_stump"] = dict(
        label="Kings Canyon NP - Big Stump East Rx (CA) - mixed conifer / sequoia underburn",
        source="CAL FIRE FRAP rxburn (NPS record)",
        start="2025-06-23", end="2025-06-28",
        reported_acres=float(tgt["TREATED_AC"].sum()),
        ecosystem="western mixed-conifer forest", target=tgt, other=other)

    configs = {}
    for key, f in fires.items():
        f["target"].to_file(REF / f"{key}_target.geojson", driver="GeoJSON")
        f["other"].to_file(REF / f"{key}_other_burns.geojson", driver="GeoJSON")
        lat, lon, off = simulated_point(f["target"], rng)
        configs[key] = dict(
            key=key, label=f["label"], source=f["source"], ecosystem=f["ecosystem"],
            lat=lat, lon=lon, point_offset_m=round(off),
            start=f["start"], end=f["end"], reported_acres=round(f["reported_acres"], 1))
        print(key, configs[key])
    (REF / "fires.json").write_text(json.dumps(configs, indent=2))


if __name__ == "__main__":
    main()
