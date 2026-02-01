import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
import numpy as np
import geopandas as gpd
import torch
import cv2
import os
import tqdm

def read_window(src, geom, out_shape=None, resampling=rasterio.enums.Resampling.nearest):
    """
    Read raster window overlapping a polygon geometry.
    Uses rasterio.windows.from_bounds.
    If out_shape is given, resamples to that shape.
    Returns masked data (float32) and transform.
    """
    left, bottom, right, top = geom.bounds
    window = from_bounds(left, bottom, right, top, transform=src.transform)

    if window.width <= 0 or window.height <= 0:
        raise RuntimeError(f"Polygon outside raster extent for {src.name}")

    patch = src.read(
        1,
        window=window,
        out_shape=out_shape if out_shape else None,
        resampling=resampling,
    )
    transform = src.window_transform(window)
    mask = geometry_mask([geom], transform=transform, invert=True, out_shape=patch.shape)
    return np.where(mask, patch, np.nan).astype(np.float32), transform


def compute_past_deforestation_distances(defo_by_year, target_year):
    """Compute 1-, 5-, and 10-year distance transforms of cumulative deforestation."""
    horizons = [1, 5, 10]
    if target_year not in defo_by_year:
        raise RuntimeError(f"Missing deforestation raster for {target_year}")
    current = defo_by_year[target_year]
    aggregated = (current == 1)
    dist_features = []
    for i in range(1, 11):
        if i in horizons:
            if np.count_nonzero(aggregated) == 0:
                dist = np.ones_like(current, dtype=np.float32) * current.shape[0]
            else:
                dist = cv2.distanceTransform(
                    (~aggregated).astype(np.uint8),
                    cv2.DIST_L2,
                    cv2.DIST_MASK_PRECISE,
                ).astype(np.float32)
            dist_features.append(dist)
            if i == horizons[-1]:
                break
        prev_year = target_year - i
        if prev_year in defo_by_year:
            aggregated |= (defo_by_year[prev_year] == 1)
    return dist_features


def extract_features_for_polygon_and_year(geom, srcs, target_year, patch_size):
    """Compute feature stack for one polygon and one target year."""
    out_shape = (patch_size, patch_size)

    defo_arrays = {}
    for y, ds in srcs["deforestation"].items():
        try:
            arr, _ = read_window(ds, geom, out_shape)
            defo_arrays[y] = np.nan_to_num(arr, nan=-1).astype(np.int8)
        except RuntimeError:
            continue

    if target_year not in defo_arrays:
        raise RuntimeError(f"Polygon outside deforestation raster for year {target_year}")

    current_deforestation = defo_arrays[target_year]
    dist_features = compute_past_deforestation_distances(defo_arrays, target_year)

    # land use
    lu_arr, _ = read_window(srcs["landuse"][target_year], geom, out_shape)
    landuse = np.nan_to_num(lu_arr, nan=0).astype(np.float32)

    # urban distance
    if np.count_nonzero(landuse == 4) == 0:
        urban_distance = np.ones_like(landuse, dtype=np.float32) * landuse.shape[0]
    else:
        urban_distance = cv2.distanceTransform(
            (landuse != 4).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        ).astype(np.float32)

    # slope
    slope_arr, _ = read_window(srcs["slope"], geom, out_shape)
    slope = np.nan_to_num(slope_arr, nan=0).astype(np.float32)

    # pasture
    pa_arr, _ = read_window(srcs["pasture"][target_year], geom, out_shape)
    pasture = np.nan_to_num(pa_arr, nan=0).astype(np.float32)

    # current + future deforestation
    current_layer = current_deforestation.astype(np.float32)
    fy = target_year + 1
    future_layer = defo_arrays.get(fy, np.zeros_like(current_layer, dtype=np.float32))

    channels = dist_features + [
        urban_distance,
        slope,
        landuse,
        pasture,
        current_layer,
        future_layer,
    ]
    return np.stack(channels, axis=0).astype(np.float32)  # (C, H, W)


def extract_all_features(gdf, srcs, output_dir, patch_size=50):
    """
    Uses gdf['split'] for train/val/test.
    Saves one .pt tensor per split-year:
        shape = (N_polygons, C, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)
    split_years = {
        "train": [2014],
        "val": [2014],
        "test": [2015],
    }

    for split, years in split_years.items():
        subset = gdf[gdf["split"] == split]
        if subset.empty:
            continue
        for year in years:
            features = []
            for idx, row in tqdm.tqdm(subset.iterrows()):
                geom = row.geometry
                feat = extract_features_for_polygon_and_year(geom, srcs, year, patch_size)
                features.append(torch.from_numpy(feat))
            tensor = torch.stack(features)  # (N, C, H, W)
            out_path = os.path.join(output_dir, f"{split}_layers_{year}.pt")
            torch.save(tensor, out_path)
            print(f"Saved {out_path}: {tensor.shape}")


def count_positive_years(df, years):
    """
    Counts, for each row, how many of the given years have n_pixels_YEAR > 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing columns like 'n_pixels_1987', etc.
    years : list[int]
        List of years to check, e.g. [1987, 1990, 1995].

    Returns
    -------
    pandas.Series
        Series of counts for each row.
    """
    cols = [f"n_pixels_{year}" for year in years if f"n_pixels_{year}" in df.columns]
    return (df[cols] > 0).sum(axis=1)


if __name__ == "__main__":
    # load polygons with a 'split' column that is either 'train', 'val', or 'test'
    import pandas as pd
    import geopandas as gpd


    df = pd.read_csv('data/original_annotation/dataset_17_3.csv')

    df_train = df.loc[df.split == 'train'].reset_index(drop=True)
    df_train["n_positive_years_train"] = count_positive_years(df_train,  [2011,2012, 2013, 2014])
    df_train = df_train.loc[df_train.n_positive_years_train >= 3].reset_index(drop=True)
    print(df_train.shape)
    df_val = df.loc[df.split == 'val'].reset_index(drop=True)
    df_val["n_positive_years_val"] = count_positive_years(df_val, [2011,2012, 2013, 2014])
    df_val = df_val.loc[df_val.n_positive_years_val >= 1].reset_index(drop=True)
    print(df_val.shape)

    df_test = pd.read_csv('data/original_annotation/15k_test.csv')
    df_test["n_positive_years_test"] = count_positive_years(df_test, [2012, 2013, 2014, 2015])

    df_test = df_test.loc[df_test.n_positive_years_test >= 1].reset_index(drop=True)
    df_test['split'] = 'test'
    print(df_test.shape)
    df = pd.concat([df_val,df_test], ignore_index=True)


    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    gdf = gpd.GeoDataFrame(df)
    gdf.crs = 4326

    # open rasters once
    srcs = {
        "deforestation": {},
        "landuse": {},
        "pasture": {},
        "slope": None
    }

    # deforestation rasters:
    # need 2004..2022 (2022 used only as future year for 2021, and 2004 gives 10-year lookback for 2014)
    for y in range(2003, 2023):
        srcs["deforestation"][y] = rasterio.open(f"data/inputs/deforestation_maps/primary_deforestation_{y}.tif")

    # yearly landuse and pasture for 2014..2021
    for y in range(2013, 2022):
        srcs["landuse"][y] = rasterio.open(f"data/inputs/lulc/lulc_{y}.tif")
        srcs["pasture"][y] = rasterio.open(f"data/engelman/pasture_quality/pasture_quality_{y}.tif")

    # slope is static
    srcs["slope"] = rasterio.open("data/inputs/elevation.tif")

    extract_all_features(gdf, srcs, output_dir="data/engelman_features")

    # cleanup
    for group in srcs.values():
        if isinstance(group, dict):
            for ds in group.values():
                ds.close()
        else:
            group.close()


