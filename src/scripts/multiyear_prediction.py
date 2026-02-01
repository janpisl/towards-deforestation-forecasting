

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, Tuple
from torchmetrics.functional import jaccard_index, precision, recall, f1_score 
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef, MetricCollection




def split_inputs_for_multi_year(
    input_batch: Dict[str, torch.Tensor],
    T: int,
    K: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs and ground truth for a K-year evaluation, using a batch that already
    contains T+K time steps.

    Inputs
    ------
    input_batch : dict
        Must contain:
          - 'deforestation_map': Tensor (B, T+K, 1, H, W)
              Per-year encoding:
                1  -> deforested THIS year (first time)
                0  -> forest
               -1  -> non-forest (after the first deforestation year, all later years remain -1)
          - 'elevation': Tensor (B, 1, H, W) (static; the model repeats it over time internally)
    T : int
        Number of context years fed to the model.
    K : int
        Number of future years to forecast and evaluate.
    device : torch.device
        Device on which tensors will reside.

    Returns
    -------
    init_def_seq : Tensor (B, T, 1, H, W)
        The first T frames; used as the model’s input sequence.
    future_truth_seq : Tensor (K, B, H, W), dtype=bool
        Per-year ground truth for the next K years: True only in the first deforestation year.
        (Kept for convenience; evaluation below uses the full future frames to build masks precisely.)
    elevation : Tensor (B, 1, H, W)
        Static elevation map (unmodified).
    base_year_map : Tensor (B, H, W)
        The map at the base year t0 (last of the first T frames). Used to build evaluation masks.
    """
    def_seq_all = input_batch['deforestation_map'].to(device).to(torch.float32)  # (B, T+K, 1, H, W)
    elevation   = input_batch['elevation'].to(device).to(torch.float32)         # (B, 1, H, W)
    B, X, _, H, W = def_seq_all.shape
    assert X >= T + K, f"Expected at least T+K frames, got X={X} (T={T}, K={K})."

    # First T frames are the model input; next K frames are the future ground truth.
    init_def_seq = def_seq_all[:, :T]                         # (B, T, 1, H, W)
    future_frames = def_seq_all[:, T:T+K, 0]                  # (B, K, H, W)
    future_truth_seq = (future_frames == 1).permute(1, 0, 2, 3)  # (K, B, H, W) bool

    # Base year (t0) is the last frame among the first T.
    base_year_map = init_def_seq[:, -1, 0]                    # (B, H, W)

    return init_def_seq, future_truth_seq, elevation, base_year_map

@torch.no_grad()
def predict_deforestation_within_k_years(
    model: torch.nn.Module,
    init_def_seq: torch.Tensor,
    elevation: torch.Tensor,
    K: int,
    device: torch.device,
    binarize_feedback: bool = True,
    fixed_window: bool = True
) -> torch.Tensor:
    """
    Autoregressively predict deforestation over K years, and return the per-pixel
    MAXIMUM probability predicted in any of the K years.
    """
    def_seq   = init_def_seq.clone().to(device).to(torch.float32)  # (B, T, 1, H, W)
    elevation = elevation.clone().to(device).to(torch.float32)     # (B, 1, H, W)
    B, T, _, H, W = def_seq.shape

    base_year_map = def_seq[:, -1, 0]                               # (B, H, W)
    ineligible_mask = (base_year_map == -1) | (base_year_map == 1)  # bool

    # Initialize with zeros so max works correctly
    p_out = torch.zeros((B, H, W), device=device, dtype=torch.float32)

    for _ in range(K):
        logits = model({'deforestation_map': def_seq, 'elevation': elevation})  # (B, T_current, 1, H, W)
        step_probs = torch.sigmoid(logits[:, -1, 0])                            # (B, H, W)

        # Update per-pixel maximum probability over time
        p_out = torch.maximum(p_out, step_probs)

        forest_mask = ~ineligible_mask
        new_event_mask = (step_probs >= 0.5) & forest_mask

        ineligible_mask |= new_event_mask

        if binarize_feedback:
            next_frame = torch.zeros_like(base_year_map, dtype=torch.float32)
            next_frame[ineligible_mask] = -1.0
            next_frame[new_event_mask]  = 1.0
        else:
            next_frame = step_probs.clone()
            next_frame[ineligible_mask] = -1.0

        next_frame_t = next_frame.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H, W)
        if fixed_window:
            def_seq = torch.cat([def_seq[:, 1:], next_frame_t], dim=1)
        else:
            def_seq = torch.cat([def_seq, next_frame_t], dim=1)

    return p_out




def compute_metrics(targets, preds, dataset_name, regression_metrics=True):

    preds = preds.to(dtype=torch.float16)
    targets = targets.to(dtype=torch.int16)

    mask = targets != -1
    valid_targets = targets[mask].flatten()
    valid_preds = preds[mask].flatten()
    
    
    if regression_metrics:
        binarized_preds = (preds >= 0.5).int() 
        pred_pos = ((binarized_preds == 1) & mask).sum(dim=(1,2)).float()
        targ_pos = ((targets == 1) & mask).sum(dim=(1,2)).float()
        # stack for torchmetrics
        # exclude tiles with no valid pixels
        has_valid_data = mask.sum(dim=(1,2)) > 0
        pred_pos, targ_pos = pred_pos[has_valid_data], targ_pos[has_valid_data]
        pred_pos = pred_pos.unsqueeze(1)
        targ_pos = targ_pos.unsqueeze(1)
        reg_metrics_coll = MetricCollection({
            f'{dataset_name}_MAE': MeanAbsoluteError(),
            f'{dataset_name}_MSE': MeanSquaredError(),
            f'{dataset_name}_RMSE': MeanSquaredError(squared=False),
            f'{dataset_name}_R2': R2Score(),
            f'{dataset_name}_PearsonR': PearsonCorrCoef(),
        })

        reg_metrics = reg_metrics_coll(pred_pos, targ_pos)


    metrics = {
        f'{dataset_name}_iou' : jaccard_index(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_f1' : f1_score(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_precision' : precision(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_recall' : recall(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_average_precision' : average_precision_score(valid_targets.flatten(), valid_preds.flatten()),
        f'{dataset_name}_ROC_AUC' : roc_auc_score(valid_targets.flatten(), valid_preds.flatten()),
    }

    if regression_metrics:
        metrics.update(reg_metrics)


    return metrics


@torch.no_grad()
def evaluate_deforestation_within_k_years(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    T: int,
    K: int,
    binarize_feedback: bool = True,
    fixed_window: bool = True,
    dataset_name: str = 'val',
    log_to_wandb: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate whether each pixel is correctly predicted to be deforested at least once
    within the next K years, using your existing dataloader (no changes to its API).

    Evaluation mask policy (handles -1 correctly across years)
    ----------------------------------------------------------
    A pixel is evaluated if:
      - it is forest in the base year t0 (value 0), AND
      - within the K-year period, it is never marked as -1 BEFORE its first deforestation year.
        (If no deforestation occurs within K, then any -1 in those K years causes the pixel to be ignored.)

    Steps per batch
    ---------------
      1) Split: first T frames -> model input; next K frames -> per-year ground truth (and statuses).
      2) Run K-step autoregressive prediction -> cumulative binary prediction (within K years).
      3) Build cumulative ground truth (deforested in any of the next K years) and the **validity mask** described above.
      4) Apply the validity mask: pixels outside are set to -1 (ignored).
      5) Compute IoU, F1, precision, recall, and average precision using your original metric function.
    """
    model.eval()
    preds_batches, targets_batches = [], []

    for input_batch, _ in dataloader:
        input_batch = {k: v.to(device).to(torch.float32) for k, v in input_batch.items()}

        # 1) Prepare the initial T frames, future per-year truth (K frames), static elevation, and base-year map.
        init_def_seq, future_truth_seq, elevation, base_year_map = split_inputs_for_multi_year(
            input_batch=input_batch, T=T, K=K, device=device
        )

        # 2) Autoregressive prediction: cumulative binary prediction (1 if deforested in any of the next K years).
        pred_within_period = predict_deforestation_within_k_years(
            model=model,
            init_def_seq=init_def_seq,
            elevation=elevation,
            K=K,
            device=device,
            binarize_feedback=binarize_feedback,
            fixed_window=fixed_window,
        )  # (B,H,W) in {0,1}

        # 3) Build cumulative ground truth and precise validity mask using the full future frames.
        #    We need the raw statuses (1/0/-1) for each future year, not just the boolean event maps.
        future_frames = input_batch['deforestation_map'][:, T:T+K, 0]  # (B,K,H,W) in {1,0,-1}

        event_seq     = (future_frames == 1)     # (B,K,H,W) True only in the first deforestation year
        nonforest_seq = (future_frames == -1)    # (B,K,H,W)

        # Has an event happened by (and including) each year? (cumulative over time)
        cum_event = torch.cumsum(event_seq.to(torch.int8), dim=1).clamp(max=1).bool()  # (B,K,H,W)

        # Years strictly before the first event (or all years if no event occurs within K)
        pre_event = torch.cat([torch.zeros_like(cum_event[:, :1]), cum_event[:, :-1]], dim=1) == 0  # (B,K,H,W)

        # Invalid if any -1 appears before the first event (or in any year if no event occurs within K)
        invalid_before_event = (nonforest_seq & pre_event).any(dim=1)  # (B,H,W)

        # Base-year eligibility: evaluate only where t0 is forest
        base_forest_mask = (base_year_map == 0)  # (B,H,W)

        # Final validity mask
        valid_mask = base_forest_mask & (~invalid_before_event)  # (B,H,W)

        # Cumulative ground truth: deforested in any of the next K years
        target_within_period = event_seq.any(dim=1).to(torch.int16)  # (B,H,W) {0,1}

        # Apply ignore (-1) outside the valid region
        targets_masked = torch.where(
            valid_mask,
            target_within_period,
            torch.full_like(target_within_period, -1, dtype=torch.int16)
        )

        preds_batches.append(pred_within_period.detach().cpu())
        targets_batches.append(targets_masked.detach().cpu())

    preds   = torch.cat(preds_batches, dim=0)
    targets = torch.cat(targets_batches, dim=0)

    metrics = compute_metrics(targets=targets, preds=preds, dataset_name=f'{dataset_name}_by_{K}y')


    return metrics, preds, targets



