import argparse
from datasets import DataInterface
from models import ModelInterface
from utils.utils import *
# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str, help="'train' or 'infer'")
    parser.add_argument('--config', default='config/model_base_conf.yaml',type=str)
    parser.add_argument('--resolu', default='20X', type=str, 
                        help="'20X', '10X', '10X-A', '5X', '5X-A'")
    parser.add_argument('--dataset', default='glioma_1p19q', type=str,
                        help='"tcga_lung", "camelyon16", "glioma_idh", \
                            "glioma_1p19q", "glioma_tert", "ctranspath_tcga_lung"')
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--f_extractor', default = 'Resnet50', type=str,
                        help='"Resnet50" , "Ctranspath"')

    args = parser.parse_args() 
    return args

def main(cfg, args):
    ignore_warning()
    pl.seed_everything(cfg.General.seed)
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.dataset = args.dataset
    cfg.stage = args.stage

    cfg.load_loggers = load_loggers(cfg, args)
    cfg.callbacks = load_callbacks(cfg)

    dm = DataInterface(args, cfg)

    ModelInterface_dict = {'model': cfg.Model,
                            'General':cfg.General,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'f_extractor':args.f_extractor
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)
    print('resolu:{} \nstage:{} \nfold:{}'.format(args.resolu,args.stage, args.fold))
    main(cfg, args)
 