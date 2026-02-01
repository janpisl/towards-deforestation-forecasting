import pdb
import os

import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
from random import randrange
from shapely.geometry import Polygon
import concurrent.futures
import cv2
from rasterio.windows import from_bounds

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import get_location

#ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin






def check_rasters_are_identical(paths):
    
    transforms = []
    heights = []
    widths = []

    for raster_path in paths:
        with rasterio.open(raster_path) as src:
            transforms.append(src.transform)
            heights.append(src.height)
            widths.append(src.width)
    
    assert len(list(set(heights))) == 1
    assert len(list(set(widths))) == 1
    assert len(list(set(transforms))) == 1

    return True


def windowed_read(file_path, col, row, patch_size):
    with rasterio.open(file_path) as src:
        window = rasterio.windows.Window(col, row, patch_size, patch_size)
        patch = src.read(window=window)
    
    return patch


def extract_temporal_stack(col, row, patch_size, raster_paths):
    patches = []
    for file_path in raster_paths:
        patches.append(windowed_read(file_path, col, row, patch_size))
    patches = np.concatenate(patches) 
    
    return patches


def process_candidate(args):

    col, row, patch_size, raster_paths, target_values, min_positive_pixels, raster_transform = args
    patches = extract_temporal_stack(col, row, patch_size, raster_paths)
    positive_pixels = np.isin(patches, target_values).sum()
    if positive_pixels >= min_positive_pixels:
        patch_dict = {
            'col': col,
            'row': row,
            'geometry': get_location([row, col], patch_size, raster_transform),
            'positive_pixels': positive_pixels
        }
        positive_pixels_per_year = np.isin(patches, target_values).sum(axis=(1,2))
        for i in range(patches.shape[0]):
            year = i + 1986  # first year of deforestation layers
            #year = i + 2010  # first year of deforestation layers
            patch_dict[f'n_pixels_{year}'] = positive_pixels_per_year[i]

    else:
        patch_dict = None
    
    return patch_dict


def main(rasters_folder, output_gdf_path, patch_size, target_values, min_positive_pixels, n_examples, workers):

    raster_paths = [os.path.join(rasters_folder, f) for f in os.listdir(rasters_folder) if f.endswith(".tif")]
    assert check_rasters_are_identical(raster_paths)

    with rasterio.open(raster_paths[0]) as src:
        raster_transform = src.transform
        height = src.height
        width = src.width

    edge_margin_px = 100
    height, width = height - edge_margin_px*2, width - edge_margin_px*2
    extracted_patches = []
    #all_engelman_features = []
    batch_size = workers*10  # Adjust based on workload and system resources

    save_every = 50000

    while len(extracted_patches) < n_examples:
        # Prepare a batch of candidate patch coordinates
        candidates = []
        for _ in range(batch_size):
            random_col = randrange(0, width - patch_size)
            random_row = randrange(0, height - patch_size)
            candidates.append((
                random_col,
                random_row,
                patch_size,
                raster_paths,
                target_values,
                min_positive_pixels,
                raster_transform
            ))

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(process_candidate, candidates)
        
        for res in results:
            if res is not None:
                #all_engelman_features.append(res.pop('engelman_features'))
                extracted_patches.append(res)
                if len(extracted_patches) % save_every == 0 and len(extracted_patches) > 0:
                    df = pd.DataFrame(extracted_patches)
                    gdf = gpd.GeoDataFrame(df, geometry='geometry')
                    gdf.to_file(output_gdf_path + f"{len(extracted_patches)}.gpkg")
                    print(f"written file with {len(extracted_patches)} patches")
                    
                """if len(all_engelman_features) % save_every == 0 and len(all_engelman_features) > 0:
                    stacked_engelman_feats = np.stack(all_engelman_features)
                    np.save((output_gdf_path + f"{stacked_engelman_feats.shape[0]}.npy"),  stacked_engelman_feats)"""
            if len(extracted_patches) >= n_examples:
                break
        


if __name__ == "__main__":

    patch_size = 50
    n_examples = 20000000
    min_positive_pixels = 0

    #input_folder = '/data/deforestation_forecasting/mapbiomas_deforestation/clipped_masked_cog'
    #input_folder = 'data/collection_9/cog/clipped_masked'
    input_folder = 'data/collection_9/recomputed_primary_deforestation'
    
    #output_gdf_path = 'data/tmp/dataset_14_3_primary_deforestation/patches_'
    #output_gdf_path = 'data/tmp/collection_9_dataset_from_2010/patches_'
    #output_gdf_path = 'data/tmp/collection_9_dataset_from_2010_deforestation_only/patches_'
    #output_gdf_path = 'data/tmp/collection_9_no_filter/patches_'
    #output_gdf_path = 'data/tmp/coll_9_2010_2019/patches_'
    output_gdf_path = 'data/tmp/patches_26_9'
    
    workers = os.cpu_count()
    #workers = 1

    #target_values = [403]
    target_values = [4]
    #target_values = [1]

    main(input_folder, output_gdf_path, patch_size, target_values=target_values, min_positive_pixels=min_positive_pixels, n_examples=n_examples, workers=workers)
