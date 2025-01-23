import random
import torch
from tqdm import tqdm
from my_utils.utils import *
import my_utils.file_util as fu


def get_data_info(data_path):
    ret = {}
    for dp in find_file(data_path, 2, suffix='.npy'):
        ret[get_name_from_path(dp)] = dp
    return ret

class MyLoader(object):
    def __init__(self, args=None, cfg=None, stage='train'):
        self.args = args
        self.cfg = cfg
        self.stage = stage
        self.max_patch_num = cfg.max_patch_num
        self.dataset = args.dataset
        self.resolu = args.resolu
        
        self.label_path = cfg.label_path[args.dataset]
        self.fold_dir = cfg.fold_dir
        self.fold = args.fold
        if args.dataset.find('glioma')>=0:
            self.data_path = cfg.data_path['glioma'][self.resolu]
        else:
            self.data_path = cfg.data_path[args.dataset][self.resolu]
        
        self.data = {}
        self.indexes = {}
        self.load_data_info = {}
        self.label = {}
        self.load_wsi_name = []
        self.patients = []

        self.data_info = get_data_info(self.data_path)
        fold_dir_name = 'glioma' if args.dataset.find('glioma')>=0 else args.dataset

        if stage=='train':
            self.patients = [i[0] for i in fu.csv_reader(os.path.join(self.fold_dir, fold_dir_name, 
                                                                        f'flod-{self.fold}-train.csv'))]
        elif stage=='val':
            self.patients = [i[0] for i in fu.csv_reader(os.path.join(self.fold_dir, fold_dir_name, 
                                                                        f'flod-{self.fold}-val.csv'))]
        else:
            if args.dataset=='camelyon16':
                for wsi_name, feature_path in self.data_info.items():
                    if wsi_name.find('test')>=0:
                        self.patients.append(wsi_name)
            else:
                self.patients = [i[0] for i in fu.csv_reader(os.path.join(self.fold_dir, fold_dir_name, 
                                                                        f'flod-{self.fold}-test.csv'))]
        self.read_label()

        # wsi in the fold or not.
        for wsi_name, feature_path in self.data_info.items():
            patient_name = '-'.join(wsi_name.split('-')[0:3]) if \
                args.dataset!='camelyon16' else wsi_name
            if patient_name in self.patients:
                self.load_data_info[wsi_name] = feature_path
                self.load_wsi_name.append(wsi_name)
        flow(f"Total wsi num: {len(self.data_info.keys())}. {stage} num: {len(self.load_wsi_name)}")
        # self.load_wsi_name = self.load_wsi_name[0:10]
        # Loading the data at once or not. 
        
        for wsi_name in tqdm(self.load_wsi_name[0:40]):
            _label = self.get_label(wsi_name)
            feature_path = self.load_data_info[wsi_name]
            features, index = self.load_npy(feature_path)
            if len(features)==0: continue
            self.data[wsi_name] = [features, _label]
            self.indexes[wsi_name] = index
            if len(features)>self.max_patch_num:
                    features = features[0:self.max_patch_num]
                    index = index[0:self.max_patch_num]
        self.new_wsi_name = [i for i in self.data.keys()]

    def load_npy(self, feature_path):
        data = np.load(feature_path,allow_pickle=True)
        features = torch.FloatTensor(data['feature'])
        indexes = torch.LongTensor(data['index'])
        return features, indexes

    def read_label(self):
        self.label = {}
        if self.dataset == 'camelyon16':
            pass
        elif self.dataset.find('tcga_lung')>=0:
            for num, (patient, d) in enumerate(fu.csv_reader(self.label_path)) :
                self.label[patient] = d

        elif self.dataset.find('glioma')>=0:
            for num, (patient, d) in enumerate(fu.csv_reader(self.label_path)) :
                # frist row is head
                if num==0: continue
                self.label[patient] = d

    def get_label(self, wsi_name):
        label = None
        if self.dataset == 'camelyon16':
            pass
        elif self.dataset.find('tcga_lung'):
            patient_name = '-'.join(wsi_name.split('-')[0:3])
            label = 1 if self.label[patient_name].find('TCGA_LUAD_ALL')>=0 else 0
        elif self.dataset == 'glioma_idh':
            patient_name = '-'.join(wsi_name.split('-')[0:3])
            if self.label.get(patient_name) is not None:
                _class = self.label[patient_name]
                label = int(_class=='Mutant')
        elif self.dataset == 'glioma_1p19q':
            patient_name = '-'.join(wsi_name.split('-')[0:3])
            if self.label.get(patient_name) is not None:
                _class = self.label[patient_name]
                label = int(_class=='codel')
        elif self.dataset == 'glioma_tert':
            patient_name = '-'.join(wsi_name.split('-')[0:3])
            if self.label.get(patient_name) is not None:
                _class = self.label[patient_name]
                label = int(_class=='Mutant')
        return label
    
    def __len__(self):
        return len(self.new_wsi_name)

    def __getitem__(self, idx):
        wsi_name = self.new_wsi_name[idx]
        if self.data.get(wsi_name) is None:
            feature_path = self.load_data_info[wsi_name]
            features, index = self.load_npy(feature_path)
            label = self.get_label(wsi_name)
            if len(features)>self.max_patch_num:
                features = features[0:self.max_patch_num]
                index = index[0:self.max_patch_num]
        else:
            features, label  = self.data[wsi_name]
            index = self.indexes[wsi_name]

        features = torch.FloatTensor(features)
        index = torch.LongTensor(index)

        if self.stage == 'train':
            _ind = [x for x in range(features.shape[0])]
            random.shuffle(_ind)
            features = features[_ind]
            index = index[_ind]

        return features, label, index, wsi_name,
