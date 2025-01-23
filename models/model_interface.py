
import inspect
import importlib
import os
import random
import pandas as pd
import my_utils.file_util as fu
from sklearn.metrics import accuracy_score
#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch
#---->
import torch
import torchmetrics
#---->
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.pre = []
        self.label = []
        self.data= [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->acc
        self.train_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.val_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro',task="multiclass")
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,task="multiclass",
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(num_classes = 2,task="multiclass"
                                                                             ),
                                                     torchmetrics.F1Score(num_classes = 2,task="multiclass",
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',task="multiclass",
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',task="multiclass",
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')
        self.shuffle = kargs['data']['data_shuffle']
        
        self.count = 0
        self.postive_patch_pro = []
        self.negative_patch_pro = []
        self.val_step_outputs = []
        self.output_results = []

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        data, label,_,_ = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(logits, label)

        Y_hat = int(Y_hat)
        Y = int(label)
        self.train_data[Y]["count"] += 1
        self.train_data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 

    def on_train_epoch_end(self):
        for c in range(self.n_classes):
            count = self.train_data[c]["count"]
            correct = self.train_data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('Train class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.train_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch):
        data, label,_,_ = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        Y = int(label)
        self.val_data[Y]["count"] += 1
        self.val_data[Y]["correct"] += (Y_hat.item() == Y)
        self.val_step_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})

    def on_validation_epoch_end(self):
        logits = torch.cat([x['logits'] for x in self.val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in self.val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in self.val_step_outputs])
        target = torch.stack([x['label'] for x in self.val_step_outputs], dim = 0)
        
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        torch.use_deterministic_algorithms(False)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        print(self.valid_metrics(max_probs.squeeze() , target.squeeze()))
        torch.use_deterministic_algorithms(True)
        #---->acc log
        for c in range(self.n_classes):
            count = self.val_data[c]["count"]
            correct = self.val_data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('VAL class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.val_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch):
        data, label, index, feature_path = batch
        index = index.squeeze(0).cpu().detach().numpy()
        results_dict = self.model(data=data, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        self.pre.append(Y_hat.item())
        
        #---->acc log
        Y = int(label)
        self.label.append(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        self.output_results.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})
 
    def on_test_epoch_end(self):
        ACC = accuracy_score(self.label, self.pre)

        print("ACC:{}".format(ACC))
        probs = torch.cat([x['Y_prob'] for x in self.output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in self.output_results])
        target = torch.stack([x['label'] for x in self.output_results], dim = 0)
        patient_results = []
        for i in range(len(probs)):
            patient_results.append([target[i].cpu().numpy()[0],probs[i].cpu().numpy()[0], probs[i].cpu().numpy()[1]])
        fu.write_csv_row(os.path.join(self.log_path, 'best_prob.csv'), patient_results)

        from sklearn import metrics
        from sklearn.metrics import auc
        #---->
        pred_class1 = []
        pred_class0 = []
        for p in probs:
            pred_class0.append(p[0].cpu().squeeze().numpy())
            pred_class1.append(p[1].cpu().squeeze().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().squeeze().numpy(), \
                                        pred_class1, pos_label=1)

        print("sklearn auc of class1:{:.04}".format(auc(fpr, tpr)))
        torch.use_deterministic_algorithms(False)
        auc_score = self.AUROC(probs, target.squeeze())
        torch.use_deterministic_algorithms(True)
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc_score
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()

        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
#---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')

    def load_model(self):
        name = self.hparams.model.name

        try:
            Model = getattr(importlib.import_module(
                f'models.my_trans'), name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)