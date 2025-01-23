from pathlib import Path
import os
#---->read yaml
import yaml
from my_utils.utils import *
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(cfg, args):

    log_path = os.path.join(cfg.General.log_path, 
                            args.dataset, args.resolu)
    just_ff(log_path, create_floder=True)

    cfg.callback_dir = os.path.join(log_path, f'fold{args.fold}', 'callback')
    cfg.log_path = Path(log_path) / f'fold{args.fold}' / 'callback'
    print(f'---->Log dir: {log_path}')
    
    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(log_path, name = f'fold{args.fold}', 
                                             version = 'tb_logger',
                                             log_graph = True, default_hp_metric = False)
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(log_path, name = f'fold{args.fold}', 
                                             version = 'tb_logger',)
    
    return [tb_logger, csv_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    just_ff(cfg.callback_dir, create_floder=True)

    early_stop_callback = EarlyStopping(
        #monitor='val_loss',
        monitor='val_MulticlassF1Score',
        min_delta=0.000,
        patience=cfg.General.patience,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_MulticlassF1Score',
                                         dirpath = str(cfg.callback_dir),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i]) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss
