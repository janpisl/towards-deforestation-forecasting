#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import geometry_window
from rasterio.windows import Window
import torch
from shapely import affinity
from torchmetrics.functional import jaccard_index, precision, recall, f1_score 
from torchmetrics.functional.classification import binary_precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score

import math
import numpy as np
import tqdm 
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.get_dataloaders import get_dataloaders
from src.algorithm.models.get_model import  get_model
from src.algorithm.loss import get_loss_fn
from src.evaluate import evaluate 



def extract_patches_for_area(raster_paths, geom, patch_size):
    """
    - Align `geom` to the raster grid (via geometry_window).
    - Read data from all rasters covering (geom + buffer).
    - The read window is buffered so that height and width are divisible by patch_size.
    - Split into non-overlapping patches of patch_size (h, w).
    - Compute total number of positive values across all rasters.
    - Exclude patches that do not intersect `geom`.

    Assumes: all rasters in `raster_paths` share the same grid as raster_paths[0].

    Returns
    -------
    GeoDataFrame with columns:
      - positive_count
      - geometry (patch polygon, aligned to raster grid)
    """

    h, w = patch_size

    with rasterio.open(raster_paths[0]) as ref_src:
        crs = ref_src.crs

        # 1) Window covering geom, snapped to pixel grid
        base_window = geometry_window(ref_src, [geom], pad_x=0, pad_y=0)

        # 2) Buffer window so its size is divisible by patch_size
        H0 = int(base_window.height)
        W0 = int(base_window.width)

        # smallest non-negative padding to make H0, W0 divisible by h, w
        pad_h = (-H0) % h
        pad_w = (-W0) % w

        # expand downward/right; clip to raster bounds if at image edge
        new_height = min(H0 + pad_h, ref_src.height - base_window.row_off)
        new_width = min(W0 + pad_w, ref_src.width - base_window.col_off)

        window = Window(
            row_off=base_window.row_off,
            col_off=base_window.col_off,
            height=new_height,
            width=new_width,
        )

        H = int(window.height)
        W = int(window.width)

        # Note: if clipping occurred at raster edge, H or W may no longer be
        # exactly divisible by h/w, but loops below still only use full patches.
        transform = ref_src.window_transform(window)

        # 3) Build grid of patch geometries once; keep only patches intersecting geom
        patch_index_map = {}  # (i, j) -> index
        patch_geoms = []
        idx = 0

        for i in range(0, H - h + 1, h):
            for j in range(0, W - w + 1, w):
                x_min, y_max = rasterio.transform.xy(transform, i,     j,     offset="ul")
                x_max, y_min = rasterio.transform.xy(transform, i + h, j + w, offset="ul")
                patch_geom = box(x_min, y_min, x_max, y_max)

                if not patch_geom.within(geom):
                    continue  # exclude patches outside AOI

                patch_index_map[(i, j)] = idx
                patch_geoms.append(patch_geom)
                idx += 1

        n_patches = len(patch_geoms)
        total_positive = np.zeros(n_patches, dtype=int)

        # 4) Accumulate positive counts across all rasters for kept patches only
        for path in raster_paths:
            with rasterio.open(path) as src:
                data = src.read(window=window, masked=True)
                H_data, W_data = data.shape[1], data.shape[2]

                if H_data != H or W_data != W:
                    raise ValueError(f"Window size mismatch for raster {path}")

                for i in range(0, H - h + 1, h):
                    for j in range(0, W - w + 1, w):
                        key = (i, j)
                        idx = patch_index_map.get(key)
                        if idx is None:
                            continue  # patch does not intersect geom

                        patch = data[:, i:i + h, j:j + w]
                        if patch.mask.all():
                            continue  # all nodata for this patch in this raster

                        total_positive[idx] += np.sum(patch.data > 0)

    gdf_patches = gpd.GeoDataFrame(
        {"positive_count": total_positive, "geometry": patch_geoms},
        crs=crs,
    )

    return gdf_patches

import numpy as np
import rasterio
from rasterio.transform import Affine



def baseline_predict_eval(dataloader):

    model = ExponentialDecayBaseline()

    preds_batches = []
    targets_batches = []
    inputs_batches = []

    for idx, (input_batch, labels_batch) in tqdm.tqdm(enumerate(dataloader)):

        def_map = input_batch['deforestation_map'][:,-1].squeeze()
        y_hat = model.predict(def_map)

        labels_batch = labels_batch.cpu().detach()[:,-1].squeeze()

        preds_batches.append(y_hat)
        targets_batches.append(labels_batch)
        inputs_batches.append(def_map)

    targets = torch.cat(targets_batches, axis=0).cpu().detach()
    preds = torch.cat(preds_batches, axis=0).cpu().detach()
    inputs = torch.cat(inputs_batches, axis=0).cpu().detach()

    metrics = compute_metrics(targets, preds, 'test')

    
    return metrics, preds, targets

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

class ExponentialDecayBaseline:
    """
    Risk decays exponentially with distance to nearest deforestation (value=1).
    Output risk only for forest pixels (value=0). Other values left at 0.
    """
    def __init__(self, decay_distance: float = 3.0): #Value 3 is based on hyperparam search
        self.decay_distance_pixels = float(decay_distance)

    def predict(self, deforestation_map: torch.Tensor) -> torch.Tensor:
        # deforestation_map: (B,H,W) with {1=deforestation, 0=forest, -1=invalid}
        dm = deforestation_map.cpu().numpy()
        risk_maps = []

        for b in range(dm.shape[0]):
            is_deforestation = (dm[b] == 1)
            is_forest        = (dm[b] == 0)

            if is_deforestation.any():
                # distance to nearest deforestation pixel
                distance_to_defor = distance_transform_edt(~is_deforestation)
            else:
                distance_to_defor = np.full(dm[b].shape, np.inf, dtype=float)

            risk = np.exp(-distance_to_defor / self.decay_distance_pixels)

            # keep risk only on forest pixels
            risk[~is_forest] = 0.0

            risk_maps.append(risk.astype(np.float32))

        return torch.from_numpy(np.stack(risk_maps))

    

def compute_metrics(targets, preds, dataset_name):

    preds = preds.to(dtype=torch.float16)
    targets = targets.to(dtype=torch.int16)

    mask = targets != -1
    valid_targets = targets[mask].flatten()
    valid_preds = preds[mask].flatten()
    
    _prec, _rec, _thresh = binary_precision_recall_curve(valid_preds, valid_targets,ignore_index=-1)


    metrics = {
        f'{dataset_name}_iou' : jaccard_index(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_f1' : f1_score(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_precision' : precision(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_recall' : recall(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_average_precision' : average_precision_score(valid_targets.flatten(), valid_preds.flatten()),
        f'{dataset_name}_ROC_AUC' : roc_auc_score(valid_targets.flatten(), valid_preds.flatten()),
    }

    return metrics






NODATA_U16 = 65535  # reserved nodata
SCALE_U16  = 65534  # map [0,1] -> [0..65534]


def open_output_raster(patch_gdf, ref_raster_path, out_raster_path, block=512):
    with rasterio.open(ref_raster_path) as src:
        crs = src.crs
        res_x, res_y = src.res

    minx, miny, maxx, maxy = patch_gdf.total_bounds
    width  = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / abs(res_y)))

    transform = Affine(res_x, 0, minx, 0, -abs(res_y), maxy)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA_U16,
        "tiled": True,
        "blockxsize": int(block),
        "blockysize": int(block),
        "compress": "zstd",     # change to "deflate" if needed
        "predictor": 2,         # integer predictor
        "bigtiff": "YES",
    }

    dst = rasterio.open(out_raster_path, "w", **profile)
    # IMPORTANT: do NOT prefill the raster (no np.full write)
    return dst, (minx, maxy, res_x, abs(res_y), height, width)


def write_chunk_to_raster(dst, patches_chunk_gdf, preds_chunk_float01, patch_size, geo_params, mask_chunk_invalid):
    minx, maxy, res_x, res_y_abs, height, width = geo_params
    ph, pw = patch_size
    # quantize float [0,1] -> uint16 [0..65534], reserve 65535 for nodata
    #q = np.rint(np.clip(preds_chunk_float01, 0.0, 1.0) * SCALE_U16).astype(np.uint16, copy=False)
    #q = torch.clamp(preds_chunk_float01, 0.0, 1.0).mul(SCALE_U16).round().to(torch.uint16).cpu().numpy()
    preds_np = preds_chunk_float01.detach().cpu().numpy().astype(np.float32, copy=False)
    q = np.rint(preds_np * SCALE_U16).astype(np.uint16)
    q[mask_chunk_invalid] = NODATA_U16

    if mask_chunk_invalid is not None:
        # ensure writable before masking
        q = q.copy()
        q[mask_chunk_invalid.detach().cpu().numpy()] = NODATA_U16

    for i, row in enumerate(patches_chunk_gdf.itertuples(index=False)):
        x_min, y_min, x_max, y_max = row.geometry.bounds

        col_start = int(round((x_min - minx) / res_x))
        row_start = int(round((maxy - y_max) / res_y_abs))

        row_end = row_start + ph
        col_end = col_start + pw

        if row_start < 0 or col_start < 0 or row_end > height or col_end > width:
            continue

        dst.write(q[i], 1, window=Window(col_start, row_start, pw, ph))


def main():


    experiments = {
        'efd6ara9': {'name': '1'},
        #'9vl8uyga': {'name': '2'},
        #'mzt06n2n': {'name': '3'},
        
    }


    for wandb_id in experiments.keys():

        model_path = f'models_trained/{wandb_id}/best.pt'
        config_path = f'models_trained/{wandb_id}/config.json'

        with open(config_path) as src:
            config = json.load(src)

        
        config['test_period'] = {"year_min" : 2012, "year_max": 2017, "input_seq_len": 4}
        config['wandb']['log_to_wandb'] = False


        model = get_model(config)

        model.load_state_dict(torch.load(model_path, weights_only=True)['state_dict'])
        _ = model.eval()
        
        experiments[wandb_id]['config'] = config
        experiments[wandb_id]['model'] = model
        experiments[wandb_id]['loss_fn'] = get_loss_fn(config)

    amazon = gpd.read_file('/home/jan/Documents/EPFL/deforestation_forecasting/data/terrabrasilis/amazon_biome_border/amazon_biome_border.shp')
    raster_paths = [   
        f'data/collection_9/recomputed_primary_deforestation/primary_deforestation_{year}.tif' for year in range(2012,2023) 
        ]
    dataloader_type = 'test'

    exp = experiments[wandb_id]
    model = exp['model']
    config = exp['config'] 

    loss_fn = exp['loss_fn'] 

    amazon = amazon.to_crs(4326)

    amazon_largest = max(amazon.geometry.iloc[0].geoms, key=lambda p: p.area)

    amazon_largest = amazon_largest.simplify(tolerance=0.01)

    patch_size = 500
    sample_area = amazon_largest

    sample_area = sample_area.buffer(-0.1)

    border = 30

    test_year = 2016

    input_raster_paths = [
        path for path
            in raster_paths 
        if int(path.split('_')[-1].replace('.tif','')) <= test_year ][-5:]


    eval_area_patches = extract_patches_for_area(
        input_raster_paths[:-1], sample_area, (patch_size, patch_size)
    )


    # original patch size in pixels
    h, w = patch_size, patch_size

    # scale factors for x (cols) and y (rows) based on new patch_size
    xfact = (w + border) / w
    yfact = (h + border) / h

    # scale each polygon around its centroid; keep same count and centers
    input_area_patches = eval_area_patches.copy()
    input_area_patches["geometry"] = input_area_patches.geometry.apply(
        lambda g: affinity.scale(g, xfact=xfact, yfact=yfact, origin="center")
    )

    input_positive_only = input_area_patches.loc[
        (input_area_patches.positive_count > 0)
    ].reset_index(drop=True)

    eval_positive_only = eval_area_patches.loc[
        (eval_area_patches.positive_count > 0)
    ].reset_index(drop=True)

    assert input_positive_only.shape == eval_positive_only.shape


    print("Area covered:", (eval_area_patches.positive_count > 0).sum() / eval_area_patches.shape[0])

    eval_positive_only.to_file('data/tmp/inference_eval_patches.gpkg')
    input_positive_only.to_file('data/tmp/inference_input_patches.gpkg')



    config['test_period'] = {
        "year_min": test_year - 4,
        "year_max": test_year + 1,
        "input_seq_len": 4
    }

    patches_per_run = 16
    batch_size = 4

    config['input_patch_size'] = patch_size + border
    config['evaluated_patch_size'] = patch_size
    config['batch_size'] = batch_size


    n_patches = len(input_positive_only)
    n_splits = math.ceil(n_patches / patches_per_run)

    print("splits: ", n_splits)

    positive_patches = eval_positive_only.copy()

    positive_patches["pos_i"] = np.arange(len(positive_patches))



    out_path_model = f'data/tmp/inference_results_11_12_4/inference_area_{patch_size}_{test_year}_border_px_{border}.tif'
    out_path_base  = f'data/tmp/inference_results_11_12_4/inference_area_baseline_{patch_size}_{test_year}_border_px_{border}.tif'

    print("creating raster 1")
    dst_model, geo_params = open_output_raster(
        patch_gdf=eval_positive_only,  # bounds of all patches (same as your original)
        ref_raster_path=input_raster_paths[-1],
        out_raster_path=out_path_model
    )
    print("creating raster 2")

    dst_base, _ = open_output_raster(
        patch_gdf=eval_positive_only,
        ref_raster_path=input_raster_paths[-1],
        out_raster_path=out_path_base
    )

    try:
        with torch.inference_mode():
            for split_idx in tqdm.tqdm(range(n_splits)):
                print(split_idx)
                start = split_idx * patches_per_run
                end = min(len(positive_patches), (split_idx + 1) * patches_per_run)
                if start >= end:
                    break

                eval_chunk = positive_patches.iloc[start:end].copy()
                input_chunk = input_positive_only.loc[eval_chunk.index].copy()

                # for the dataloader, you can reset_index, but keep mapping stable
                annotation_path = f'data/tmp/inference_area_{patch_size}_part{split_idx}.csv'
                config['annotation_file'] = annotation_path

                df = pd.DataFrame(input_chunk).reset_index(drop=True)
                df['split'] = 'test'
                df.to_csv(annotation_path, index=False)

                dtl = get_dataloaders([f'{dataloader_type}'], config, inference_dataset=True)[f'{dataloader_type}']

                baseline_metrics_chunk, baseline_preds_chunk, _ = baseline_predict_eval(dtl)
                metrics_chunk, preds_chunk, targets_chunk = evaluate(
                    model, loss_fn, dtl, 'cuda', config, 'test', return_preds_and_targets=True
                )

                if border > 0:
                    b2 = int(border / 2)
                    preds_chunk = preds_chunk[:, b2:-b2, b2:-b2]
                    baseline_preds_chunk = baseline_preds_chunk[:, b2:-b2, b2:-b2]
                    targets_chunk = targets_chunk[:, b2:-b2, b2:-b2]

                mask_invalid = (targets_chunk == -1)
                # WRITE THIS CHUNK DIRECTLY INTO THE FINAL RASTER(S)
                write_chunk_to_raster(dst_model, eval_chunk, preds_chunk, (patch_size, patch_size), geo_params, mask_invalid)
                write_chunk_to_raster(dst_base,  eval_chunk, baseline_preds_chunk, (patch_size, patch_size), geo_params, mask_invalid)

                metrics = metrics_chunk
                baseline_metrics = baseline_metrics_chunk
                print(metrics)
                print(baseline_metrics)
                # free chunk arrays ASAP
                del preds_chunk, baseline_preds_chunk, targets_chunk, mask_invalid

    finally:
        dst_model.close()
        dst_base.close()




if __name__ == "__main__":

    main()