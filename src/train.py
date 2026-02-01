"""Train the model
python src/train.py -c  config/config.json
python src/train.py -c  config/config.json -l (to log errors to file and continue running)
"""
import json
import sys
import argparse
import os
import itertools
from pathlib import Path
import pprint
from datetime import datetime
import traceback

import wandb
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
                        

# Add the parent directory to sys.path to recognize 'src' as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithm.loss import get_loss_fn
from src.algorithm.models.get_model import get_model
from src.evaluate import evaluate, compute_metrics
from src.data.get_dataloaders import get_dataloaders
from src import utils


def train(model, optimizer, loss_fn, dataloader, device, config, batch_indices_to_compute_metrics=None):
    """Train the model for one epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        config: (dict) configuration duh
        indices_for_computing_metrics (torch.tensor) : if provided, only use batches at given indices to compute metrics (to save time)
    """

    model.train()

    preds_batches = []
    targets_batches = []

    for idx, (input_batch, labels_batch) in enumerate(dataloader):


        if isinstance(input_batch, dict):
            labels_batch = labels_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            for k,v in input_batch.items():
                input_batch[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            input_batch, labels_batch = input_batch.to(device=device, dtype=torch.float32, non_blocking=True), labels_batch.to(device=device, dtype=torch.float32, non_blocking=True)


        y_hat = model(input_batch)

        if not config['train_on_subsequences'] or config['model']['model_name'] == 'engelman':
            if len(labels_batch.shape) == 5:
                labels_batch = labels_batch[:, -1]
            if len(y_hat.shape) == 5:
                y_hat = y_hat[:,-1]
        
        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            mask = labels_batch != -1
            loss = loss_fn(y_hat[mask], labels_batch[mask])
        else:
            loss = loss_fn(y_hat, labels_batch)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if config['wandb']['log_to_wandb'] and idx % 10 == 0: #Logging only every 10th loss because wandb takes forever
            log_dict =  {"train-loss": loss.item()}
            wandb.log(log_dict)

        # Compute metrics only a subset of training data to save memory and time
        if batch_indices_to_compute_metrics is not None:
            if idx in batch_indices_to_compute_metrics:
                preds = F.sigmoid(y_hat.detach().cpu())
                labels_batch = labels_batch.detach().cpu()
                preds_batches.append(preds)
                targets_batches.append(labels_batch)

    targets = torch.cat(targets_batches, axis=0)
    preds = torch.cat(preds_batches, axis=0)

    metrics = compute_metrics(targets[:, -1], preds[:, -1], 'train', regression_metrics=True)

    if config['wandb']['log_to_wandb']:
        wandb.log(metrics)
        try:
            log_by_year = config['wandb']['log_train_metrics_by_year']
        except KeyError:
            log_by_year = False
        if log_by_year:
            seq_len = y_hat.size(1)
            for i in range(seq_len):
                metrics = compute_metrics(targets[:,i], preds[:,i], f'train-{i}')
                wandb.log(metrics)


def is_best_epoch(metric, current_best, metric_direction):

    if metric_direction == 'asc':
        is_best = metric > current_best
    elif metric_direction == 'desc':
        is_best = metric < current_best

    return is_best


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, config, device, model_dir):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """
    # reload weights from restore_file if specified
    try:
        print(f"Restoring parameters from {config['restore_file_path']}")
        utils.load_checkpoint(config['resume_training']['restore_file_path'], model, optimizer)
    except KeyError:
        pass

    current_best = 0 if config['early_stopping']['metric_direction'] == 'asc' else 999999999
    epochs_since_last_best = 0

    batches_total = len(train_dataloader)
    #Computing metrics over 1/X (currently X=50) of the train dataset to save memory and time 
    k = batches_total // 50  # sample 1/50 of total
    batch_indices_to_compute_metrics = torch.randperm(batches_total)[:k]    
    
    for epoch in range(1, config['epochs']+1):

        if epochs_since_last_best > config['early_stopping']['patience']:
            print(f"No improvement for {config['early_stopping']['patience']} epochs. Early stopping at epoch {epoch}")
            break
        
        train(model, optimizer, loss_fn, train_dataloader, device, config, batch_indices_to_compute_metrics)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, device, config)

        is_best = is_best_epoch(
            val_metrics[config['early_stopping']["metric"]], 
            current_best,
            config['early_stopping']["metric_direction"])

        if is_best:
            current_best = val_metrics[config['early_stopping']["metric"]]
            epochs_since_last_best = 0
        else:
            epochs_since_last_best = epochs_since_last_best + 1

        if config['wandb']['log_to_wandb']:
            wandb.log({'epoch' : epoch})

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              out_folder=model_dir)
        

def initialize_wandb(config, model, train_dl, val_dl):

    try:
        name = f"train {config['train_period']['year_min']}-{config['train_period']['year_max']}; seq-len {config['train_period']['input_seq_len']}"
    except:
        name = None
    
    #Resume training if checkpoint is provided
    try:
        wandb_id = config['resume_training']['wandb_id']
        resume = 'must'
        assert False, "there is a bug in resuming training"
    except KeyError:
        wandb_id, resume = None, None

    wandb.init(
        project=config['wandb']['project'], 
        group=config['wandb']['group'],
        reinit=True,
        id=wandb_id,
        resume=resume,
        name=name
    )

    log_config = config
    log_config['n_parameters'] = utils.count_parameters(model)

    log_config['train_dst_size'] = len(train_dl.dataset)
    log_config['val_dst_size'] = len(val_dl.dataset)

    wandb.config.update(log_config, allow_val_change=True)

    return wandb.run.id


def main(config):
    device = config["device"]

    dataloaders = get_dataloaders(
        ['train', 'val'], config)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    model = get_model(config)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


    loss_fn = get_loss_fn(config)

    if config['wandb']['log_to_wandb']:
        run_id = initialize_wandb(config, model, train_dl, val_dl)
        wandb.watch(model, log='all')
    else:
        run_id = 'no_wandb'

    model_dir = os.path.join(config['output_folder'], f'{run_id}')
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(model_dir, 'config.json'), 'w') as sink:
        json.dump(config, sink)
    
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, config, device, model_dir)



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str)
    parser.add_argument("-l", "--log_errors", action='store_true', help='If used, errors are logged to file and execution moves on to next configuration')
    
    args = parser.parse_args()

    log_errors_mode = args.log_errors

    with open(args.config) as src:
        config = json.load(src)

    #The config file contains iterables to enable simple hyperparams tuning
    #The code below generates all combinations and then executes them one fby one
    try:
        keys, values = zip(*config.items())
        permutations_dicts = [dict(zip(keys, value)) for value in itertools.product(*values)]
    except TypeError: #If the config options are not iterable, just run a single experiment with the config as is
        permutations_dicts = [dict(config)]


    print(f"{len(permutations_dicts)} variant(s) of config")

    for config in permutations_dicts:
        utils.seed_everything(config['seed'])
        config = utils.parse_boolean_recursion(config)
        
        try:
            try:
                wandb.finish()
            except:
                pass
            main(config)
        except Exception as e:
            if log_errors_mode:
                text =  pprint.pformat(config) + '\nError message:\n' + repr(e)  + '\n\n\n'
                with open(f"logs_{datetime.today().strftime('%m_%d')}.txt", "a") as sink:
                    sink.write(text)
            else:
                traceback.print_exc()
                exit()




 
