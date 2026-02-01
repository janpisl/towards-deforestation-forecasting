import torch.nn.functional as F 
from torchmetrics.functional import jaccard_index, precision, recall, f1_score 
from torchmetrics.functional.classification import binary_precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef, MetricCollection
import wandb
import torch
import matplotlib.pyplot as plt

from src.algorithm.loss import CombinedLossCausalMaskBias


def evaluate(model, loss_fn, dataloader, device, config, dataset_name='val', return_preds_and_targets=False, regression_metrics=False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        config: hyperparameters
        return_preds_and_targets (bool): if True, returns a Tensor containing all predictions and another with all targets
    """

    model.eval()

    losses = []
    preds_batches = []
    targets_batches = []

    with torch.no_grad():
        for idx, (input_batch, labels_batch) in enumerate(dataloader):
                
            if isinstance(input_batch, dict):
                labels_batch =  labels_batch.to(device).to(torch.float32)
                for k,v in input_batch.items():
                    input_batch[k] = v.to(device).to(torch.float32)
            else:
                input_batch, labels_batch = input_batch.to(device).to(torch.float32), labels_batch.to(device).to(torch.float32)    
                
            
            y_hat = model(input_batch)

            #All years except for last used as inputs; take last year predictions as output
            y_hat = y_hat[:, -1].squeeze()
            labels_batch = labels_batch[:, -1].squeeze()

            
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                mask = labels_batch != -1
                loss = loss_fn(y_hat[mask], labels_batch[mask])
            elif isinstance(loss_fn, CombinedLossCausalMaskBias):
                loss = loss_fn(y_hat, labels_batch, model.causal_mask.alpha, model.causal_mask.beta)
            else:
                loss = loss_fn(y_hat, labels_batch)

            predictions = F.sigmoid(y_hat).cpu().detach().to(dtype=torch.float16)
            labels_batch = labels_batch.cpu().detach().to(dtype=torch.int16)

            losses.append(loss.item())
            preds_batches.append(predictions)
            targets_batches.append(labels_batch)
    
        try:
            targets = torch.cat(targets_batches, axis=0).cpu().detach()
            preds = torch.cat(preds_batches, axis=0).cpu().detach()
        except RuntimeError:
            targets_batches[-1] = targets_batches[-1].unsqueeze(0)
            targets = torch.cat(targets_batches, axis=0).cpu().detach()
            preds_batches[-1] = preds_batches[-1].unsqueeze(0)
            preds = torch.cat(preds_batches, axis=0).cpu().detach()

        metrics = compute_metrics(targets, preds, dataset_name, regression_metrics=regression_metrics)
        metrics[f"{dataset_name}_loss"] =  torch.tensor(losses).mean().cpu().detach()

        if config['wandb']['log_to_wandb']:
            wandb.log(metrics)

    if return_preds_and_targets:
        return metrics, preds, targets

    return metrics


def compute_metrics(targets, preds, dataset_name, regression_metrics=False):

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


    _prec, _rec, _thresh = binary_precision_recall_curve(valid_preds, valid_targets,ignore_index=-1)

    #This is to remove the precision=1 at recall=0 because I don't think it's accurate
    #There is no guarantee that the highest probabilities of the model are correct
    #The best guess of how much they are correct on average is the precision at the previous threshold value
    _prec[-1] = _prec[-2] 
    
    # Make the plot
    fig, ax = plt.subplots()
    ax.plot(_rec.numpy(), _prec.numpy())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0,1.05)
    ax.set_ymargin(0.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Thin dashed grid lines

    _fig = wandb.Image(fig)
    plt.close(fig)

    metrics = {
        f'{dataset_name}_iou' : jaccard_index(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_f1' : f1_score(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_precision' : precision(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_recall' : recall(preds, targets, task='binary', num_classes=1, ignore_index=-1),
        f'{dataset_name}_average_precision' : average_precision_score(valid_targets.flatten(), valid_preds.flatten()),
        f'{dataset_name}_ROC_AUC' : roc_auc_score(valid_targets.flatten(), valid_preds.flatten()),
        f'{dataset_name}_PRC' : _fig,
    }

    if regression_metrics:
        metrics.update(reg_metrics)


    return metrics