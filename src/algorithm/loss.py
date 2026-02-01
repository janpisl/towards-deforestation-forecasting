import torch
import segmentation_models_pytorch as smp
from torch import nn, tensor


class CombinedLoss(nn.Module):
    def __init__(self, bce_pos_weight, bce_alpha, ignore_index=-1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=tensor(bce_pos_weight))
        self.dice = smp.losses.DiceLoss(mode="binary", ignore_index=ignore_index)
        self.alpha = bce_alpha
        self.ignore_index = ignore_index
        
        assert self.alpha >= 0 and self.alpha <= 1, "alpha must be between 0 and 1"


    def forward(self, pred, target):

        mask = target != self.ignore_index

        bce = self.alpha * self.bce(pred[mask], target[mask])
        dice = (1 - self.alpha) * self.dice(pred, target)
        return bce +  dice


class CombinedLossCausalMaskBias(nn.Module):
    def __init__(self, bce_pos_weight, bce_alpha, ignore_index=-1, reg_lambda=0.001):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=tensor(bce_pos_weight))
        self.dice = smp.losses.DiceLoss(mode="binary", ignore_index=ignore_index)
        self.alpha = bce_alpha
        self.ignore_index = ignore_index
        
        assert self.alpha >= 0 and self.alpha <= 1, "alpha must be between 0 and 1"

        self.reg_lambda = reg_lambda


    def forward(self, pred, target, alpha, beta):

        mask = target != self.ignore_index

        main_task_loss = self.alpha * self.bce(pred[mask], target[mask]) + (1 - self.alpha) * self.dice(pred, target)
        return main_task_loss + self.reg_lambda * (alpha ** 2 + beta ** 2)




def get_loss_fn(config):

    if isinstance(config['loss_fn'], str):
        loss_configuration = {"name" : config['loss_fn'], "params" : {}}
    else:
        loss_configuration = config['loss_fn']
    if loss_configuration['name'] == 'DiceLoss':
        loss_fn =  smp.losses.DiceLoss(mode="binary", ignore_index=-1) #Using the same one as Engelman et al. for reproducibility 
    elif loss_configuration['name'] == 'BCEWithLogitsLoss':
        if 'pos_weight' in loss_configuration['params'].keys():
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=tensor(loss_configuration['params']['pos_weight']))
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    elif loss_configuration['name'] == 'combined':
        loss_fn = CombinedLoss(loss_configuration['params']['bce_pos_weight'], loss_configuration['params']['bce_alpha'])
    return loss_fn



def compute_loss(preds, targets, loss_fn):
    """_summary_

    Args:
        y_hat (BxTxWxH): _description_
        targets (BxTxWxH): _description_
        loss_fn: _description_
    """

    assert preds.shape == targets.shape

    seq_len = preds.size(1)

    losses_per_step = []

    for i in range(seq_len):

        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            mask = targets[:,i] != -1
            timestep_loss = loss_fn(preds[:,i][mask], targets[:,i][mask])
        else:
            timestep_loss = loss_fn(preds[:,i], targets[:,i])
        
        losses_per_step.append(timestep_loss)
    