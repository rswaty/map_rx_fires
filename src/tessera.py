"""tessera — cloud-native reader for the Tessera v1 embeddings on source.coop.

The embeddings are int8 `[H,W,128]` tiles + per-pixel float32 `scales` (dequantize:
`embedding * scales`). This module turns the published GeoParquet tile index into
range-readable, GeoZarr-compliant xarray datasets — slice an AOI without
downloading whole tiles, with a spatial query to find which tiles you need.

    import tessera
    da = tessera.read_aoi(bbox=(9.9, 9.9, 10.2, 10.2), year=2024)  # dequantized xarray
    da.sel(band=7).plot()

Requires: s3fs fsspec xarray "zarr<3" duckdb numpy. All reads are anonymous.
"""
import ast
import base64
import json
import struct

import duckdb
import fsspec
import numpy as np
import s3fs
import xarray as xr

BUCKET = "us-west-2.opendata.source.coop"
ROOT = f"s3://{BUCKET}/tessera/tessera/npy/v1"
# path-style HTTPS so DuckDB reads the index anonymously (dotted bucket breaks vhost SSL)
INDEX = lambda year: f"https://s3.us-west-2.amazonaws.com/{BUCKET}/tessera/tessera/npy/v1/index/tessera_{year}.parquet"
RPC = 1  # one row per chunk: always divides H (npy has no edge-chunk padding)
_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')


def find_tiles(bbox, year):
    """Return index rows for tiles intersecting bbox=(minx,miny,maxx,maxy)."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    minx, miny, maxx, maxy = bbox
    return con.execute(
        f"""SELECT * FROM read_parquet('{INDEX(year)}')
            WHERE ST_Intersects(geometry, ST_MakeEnvelope({minx},{miny},{maxx},{maxy}))""").df()


def _inline(a):
    return "base64:" + base64.b64encode(a.tobytes()).decode()


def _za(shape, chunks, dtype, fill=0):
    return json.dumps({"zarr_format": 2, "shape": list(shape), "chunks": list(chunks),
                       "dtype": dtype, "compressor": None, "filters": None, "order": "C",
                       "fill_value": fill})


def _tile_refs(row):
    H, W, C = int(row["height"]), int(row["width"]), int(row["channels"])
    go, so = int(row["grid_offset"]), int(row["scales_offset"])
    gx, sx = row["grid_uri"], row["scales_uri"]
    c, a, _, f, _, e = row["transform"]
    x = (c + (np.arange(W) + 0.5) * a).astype("f8")
    y = (f + (np.arange(H) + 0.5) * e).astype("f8")
    refs = {
        ".zgroup": json.dumps({"zarr_format": 2}),
        ".zattrs": json.dumps({"Conventions": "CF-1.8"}),
        "x/.zarray": _za([W], [W], "<f8"), "x/0": _inline(x),
        "x/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["x"], "standard_name": "longitude",
                                 "units": "degrees_east", "axis": "X"}),
        "y/.zarray": _za([H], [H], "<f8"), "y/0": _inline(y),
        "y/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["y"], "standard_name": "latitude",
                                 "units": "degrees_north", "axis": "Y"}),
        "band/.zarray": _za([C], [C], "<i4"), "band/0": _inline(np.arange(C, dtype="i4")),
        "band/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["band"]}),
        "spatial_ref/.zarray": _za([], [], "<i4"), "spatial_ref/0": _inline(np.array(0, "<i4")),
        "spatial_ref/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": [],
            "grid_mapping_name": "latitude_longitude", "crs_wkt": _WKT,
            "spatial_ref": _WKT, "epsg_code": "EPSG:4326"}),
        "embedding/.zarray": _za([H, W, C], [RPC, W, C], "|i1"),
        "embedding/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["y", "x", "band"],
            "grid_mapping": "spatial_ref"}),
        "scales/.zarray": _za([H, W], [RPC, W], "<f4"),
        "scales/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref"}),
    }
    for ci in range((H + RPC - 1) // RPC):
        r = min(RPC, H - ci * RPC)
        refs[f"embedding/{ci}.0.0"] = [gx, go + ci * RPC * W * C, r * W * C]
        refs[f"scales/{ci}.0"] = [sx, so + ci * RPC * W * 4, r * W * 4]
    return refs


def open_tile(row):
    """Open a single tile (an index row) as a GeoZarr-compliant xarray Dataset."""
    m = fsspec.get_mapper("reference://", fo={"version": 1, "refs": _tile_refs(row)},
                          remote_protocol="s3", remote_options={"anon": True})
    return xr.open_zarr(m, consolidated=False, mask_and_scale=False)


def dequantize(ds):
    """int8 codes * per-pixel scale -> float32 embedding DataArray."""
    return (ds["embedding"].astype("float32") * ds["scales"]).rename("embedding")


def read_aoi(bbox, year, bands=None, resolution=None):
    """Spatial-query the index, range-read each intersecting tile's AOI window,
    dequantize, and mosaic onto one regular EPSG:4326 grid (dims y, x, band).

    Tiles don't share a global pixel grid (W varies with latitude), so windows are
    regridded with NEAREST (no interpolation of embeddings) onto a common target
    grid. `resolution` (degrees/pixel) defaults to the finest tile in the AOI.
    """
    import numpy as np
    minx, miny, maxx, maxy = bbox
    tiles = find_tiles(bbox, year)
    if len(tiles) == 0:
        raise ValueError("no tiles intersect AOI")
    if resolution is None:
        resolution = float(min(0.10 / tiles["width"].max(), 0.10 / tiles["height"].max()))
    tx = np.arange(minx + resolution / 2, maxx, resolution)
    ty = np.arange(maxy - resolution / 2, miny, -resolution)
    tol = float(0.10 / tiles["width"].min())  # snap within coarsest tile pixel; NaN outside

    mosaic = None
    for _, row in tiles.iterrows():
        ds = open_tile(row)
        sub = ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))
        if sub.sizes["x"] == 0 or sub.sizes["y"] == 0:
            continue
        da = dequantize(sub).compute()
        if bands is not None:
            da = da.sel(band=bands)
        # nearest-regrid onto the common target grid (no scipy); beyond tol -> NaN
        g = da.reindex(x=tx, y=ty, method="nearest", tolerance=tol)
        mosaic = g if mosaic is None else mosaic.combine_first(g)
    if mosaic is None:
        raise ValueError("no data in AOI")
    return mosaic
