import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import openslide

import os
import argparse
import my_utils.vis as vis
from torch.utils.data import DataLoader
from my_utils.utils import *
from torch.autograd import Variable
from PIL import Image
from torch import no_grad, optim 
from feature_align import model
import torch.utils.data as data
import my_utils.file_util as fu
def ret_wsi_path(wsi_root):
    ret = {}
    for c in listdir_com_p(wsi_root):
        for p in listdir_com_p(c):
            ret[p.split('/')[-1]] = []
            for w in listdir_com_p(p):
                ret[p.split('/')[-1]].extend(listdir_com_p(w))
    return ret
def get_transform():
    return transforms.Compose(
                        [
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225))
                        ]
                    )
    
class LoadData(data.Dataset):
    WSI_INDEX_BUFF  = []
    WSI_FILE_OPEN = {}
    PATCH_P_LIST = []
    WSI_INFER_BUFF = {}
    def __init__(self, args, model='train'):
        info("Search wsi........")
        self.args = args
        if model=='train':
            print(50*"$", f"Load train patches",50*"$",)
            record_file = find_file(os.path.join(PATH_RECORD, args.dataset), 4, suffix='.csv')#[0:100]
            for _csv_file in record_file:
                self.PATCH_P_LIST.extend([i[0] for i in fu.csv_reader(_csv_file)])
            print(50*"$", f"WSI num:{len(record_file)}. Patches num:{len(self.PATCH_P_LIST)}",50*"$",)

            self.transform = get_transform()

        else:
            wsi_dict = {get_name_from_path(i):i for i in find_file(args.infer_wsi_dir, 
                                                                   4, suffix=args.infer_wsi_suffix)}
            index_dict = {get_name_from_path(i):i for i in find_file(args.infer_wsi_index_dir, 
                                                                        4, suffix='.csv')}
            for wsi_name, wsi_path in tqdm(wsi_dict.items()):
                if index_dict.get(wsi_name) is None:
                    print(f"wsi: {wsi_name} has no index file!")
                    continue
                self.WSI_FILE_OPEN[wsi_name] = openslide.open_slide(wsi_path)
                ind_list = load_csv_index(index_dict[wsi_name])
                self.WSI_INFER_BUFF[wsi_name] = ind_list 
                    
    def __len__(self):
        return len(self.PATCH_P_LIST)

    def __getitem__(self, idx):
        jpg_path = os.path.join(SAVE_DIR, self.PATCH_P_LIST[idx])
        image_data =  Image.open(jpg_path).convert(mode="RGB")
        return self.transform(image_data)


def load_csv_index(data_path):
    ind_list = fu.csv_reader(data_path)
    ind_list = [[i[1], i[3]] for i in ind_list[1:len(ind_list)]]
    return ind_list


class my_train(object):
    def __init__(self, args) -> None:
        self.args = args
        self.mydata = LoadData(args,model=args.mod)

        self.model = model.AlignNetwork(low_size=RESOLU_SIZE[args.resolution]).cuda()
        if args.gpus > 1 and  torch.cuda.device_count()>1:
            tips(f"GPU NUM:{torch.cuda.device_count()}. Activating DataParallel!!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to('cuda')
        
        if args.mod == 'train':
            self.trainloader = DataLoader(self.mydata, 
                                            batch_size=args.train_bs, 
                                            shuffle=True, 
                                            drop_last=True, 
                                            num_workers=16)
            self.optimizer = optim.Adam(filter(lambda p:p.requires_grad, 
                                                self.model.parameters()), 
                                        lr=args.lr)  

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                               lr_lambda=lambda i:0.95**i)                     
        else:
            self.img_transforms = get_transform()
            pretrained_model = torch.load(args.infer_weight) 
            self.model.load_state_dict(pretrained_model, strict=False)
        info('Lr:{}'.format(self.args.lr))
        self.loss_fn = nn.L1Loss().cuda()
        self.mytb = vis.MyTB('./Align_log/feature-align_TCGA-Lung-'+args.mod, clean_log_dir=False)

    def train_epoch(self):
        self.model.train()
        for data in self.trainloader:
            data = data.cuda()
            total_loss = self.model(data, self.loss_fn,mod=self.args.mod)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if random.randint(0,100) < 3:
                print(f"Loss:{total_loss.item():.05}")

            if random.randint(0,10) < 3:
                self.mytb.add_scalar(total_loss, name="Training Loss")

    def infer(self):
        self.model.eval()
        record = [['WSI_name', 'read_time', 'infer_time']]
        record_save_name = os.path.join(self.args.infer_time_record, 
                                        f"AlignInferTime_{self.args.dataset}_{self.args.resolution}.csv")
        just_dir_of_file(record_save_name)
        fu.write_csv_row(record_save_name, record, model='w')

        # for dataset, _data in self.mydata.WSI_FILE_OPEN.items():
        #     print(f"Processing dataset :{dataset}")
        wsi_num = 0
        for wsi_name, wsi_reader in tqdm(self.mydata.WSI_FILE_OPEN.items()) :
            wsi_num += 1
            if len(self.mydata.WSI_INFER_BUFF[wsi_name])==0:
                print(f"Zeros patches! Break!!!")
                continue
            save_npy_name = os.path.join(self.args.infer_save_p, self.args.resolution, wsi_name+'.npy')
            if os.path.isfile(save_npy_name):
                print(f"WSI :'{wsi_name}' has processed!!")
                continue
            just_dir_of_file(save_npy_name)
            print(f"Infer wsi:{wsi_name} {wsi_num}/{len(self.mydata.WSI_FILE_OPEN.keys())}")
            features = {"feature":[], 'index':[]}
            read_time,infer_time = 0,0
            with torch.no_grad():
                for [x, y] in tqdm(self.mydata.WSI_INFER_BUFF[wsi_name]):
                    start = time.time()
                    image_data = wsi_reader.read_region((int(x), int(y)), 0, 
                                                (224,224)).convert('RGB')
                    image_data = image_data.resize((RESOLU_SIZE[self.args.resolution],
                                                    RESOLU_SIZE[self.args.resolution]))
                    read_time += time.time()-start
                    
                    start = time.time()
                    image_data = self.img_transforms(image_data)
                    data_train = Variable(image_data.float().cuda()).unsqueeze(0)
                    l_feature = self.model.l_model(data_train)
                    infer_time += time.time()-start

                    features["feature"].append(l_feature[-1].detach().cpu().numpy())
                    features["index"].append([int(x), int(y)])
            fu.write_csv_row(record_save_name, [[wsi_name, 
                                                str(f"{read_time:.06}"), 
                                                str(f"{infer_time:.06}")]], model='a')       
            np.save(save_npy_name, features)

    def pretraining(self):
        for epoch in range(self.args.epoch):
            print("#"*30, f"Train epoch:{epoch}, learning rate:{self.scheduler.get_lr()}", "#"*30)
            self.train_epoch()
            self.scheduler.step()
            if epoch >=10 and epoch%3==0:
                save_path = os.path.join(self.args.weight_dir,self.args.resolution,f'epoch-{epoch}.pth')
                just_dir_of_file(save_path)
                torch.save(self.model.state_dict(), save_path)

CONF = fu.read_yaml("config/align_conf.yaml")
RESOLU_SIZE = CONF["RESOLU_SIZE"]
SAVE_DIR = CONF["SAVE_DIR"]
PATH_RECORD = CONF["PATH_RECORD"]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',default=50,type=int)
    parser.add_argument('--train_bs',default=128,type=int)
    parser.add_argument('--mod',default='infer',type=str, help='"train" or "infer"')
    parser.add_argument('--dataset',default='TCGA_lung',type=str, 
                        help='TCGA_lung, TCGA_glioma, CAMELYON')

    parser.add_argument('--resolution',default='20Xto10X',type=str, 
                        help="'20Xto2.5X', '20Xto5X', '20Xto10X'")
    parser.add_argument('--infer_time_record',default='./record/time_record',type=str)
    parser.add_argument('--weight_dir',default='weight/feature_align',type=str)
    parser.add_argument('--infer_wsi_suffix',default='.svs',type=str)
    parser.add_argument('--infer_wsi_dir', default="/media/zbc/18T_dtp/南方医院/datasets_glioma/dtp_glioma(20倍)/北大深圳/data", 
                        help='infer dir of wsi',type=str)
    parser.add_argument('--infer_wsi_index_dir',default="/media/zbc/18T_dtp/南方医院/datasets_glioma/dtp_glioma(20倍)/北大深圳/224/index",
                        type=str,help='index dir of infer wsi')
    parser.add_argument('--infer_save_p',default='align_dir',
                        type=str, help='feature save dir of infer wsi')
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--gpus',default=1,type=int)
    parser.add_argument('--infer_weight',\
        default='weight/feature_align/20Xto10X/epoch-1.pth',type=str)

    args = parser.parse_args()
    ignore_warning()
    seed_enviroment(512)
    if args.mod=='train':
        my_train(args).pretraining()
    else:
        my_train(args).infer()
