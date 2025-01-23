import time
import os
import argparse
import openslide
from my_utils.utils import *
import my_utils.file_util as fu
import my_utils.file_util as fu


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--ps', type=int, default=256, help="patch size")
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--X20_level', type=int, default=0, 
                    help="Level corresponding to 20X magnification image.")
args = parser.parse_args()

CONF = fu.read_yaml("config/align_conf.yaml")

SAVE_DIR = CONF["SAVE_DIR"]
# svs file of dataset
DATA_DIR = CONF["DATA_DIR"]
# The directory where the csv file for the WSI's patch coordinates. 
# The csv files have the same name of the wsi.
INDEX_DIR = CONF["INDEX_DIR"]
# Records the relative path of the cut jpg file in the root directory where it is saved.
# Facilitates subsequent reading of the aligned model.
PATH_RECORD = CONF["PATH_RECORD"]

SELECT_DATASET = ['TCGA_lung'] 
# To prevent too many jpg files in the directory, which makes it difficult to delete. 
# Limit the number of subfiles in each directory.
MAX_FILE_OF_DIR = CONF["MAX_FILE_OF_DIR"]

def read_csv_index(csv_path):
    index_list = []
    ff = fu.csv_reader(csv_path)
    for f_num, f in enumerate(ff):
        if f_num<=0: continue
        index_list.append([int(f[1]),int(f[2]),
                            int(f[3]),int(f[4])])
    return index_list

if __name__ == '__main__':
    for dataset in SELECT_DATASET:
        wsi_dir = DATA_DIR[dataset]
        index_dir = INDEX_DIR[dataset]
        wsi_list = find_file(wsi_dir, 5, suffix=args.slide_ext)
        index_list = find_file(index_dir, 3, suffix='.csv')
        if len(wsi_list) != len(index_list):
            print(f"Dataset:{dataset}. WSI number ({len(wsi_list)}) is not euqal to the index number({len(index_list)}).")
        wsi_dict = {get_name_from_path(i):i for i in wsi_list}
        index_dict = {get_name_from_path(i):i for i in index_list}

        for wsi_num, (wsi_name, wsi_path) in enumerate(wsi_dict.items()):
            wsi_name = get_name_from_path(wsi_path)
            csv_path = index_dict[wsi_name]

            coord_list = read_csv_index(csv_path)
            
            save_wsi_dir = os.path.join(SAVE_DIR, dataset, wsi_name)
            total = len(coord_list)
            print(f"Dataset:{dataset} WSI :{wsi_name}({wsi_num}/{len(wsi_list)-1}) has {total} patches!")

            if total == 0:
                print('**********ERROR***********')
                print(f'{wsi_name} has NOOOO coords!')
            
            time_start = time.time()
            wsi = None 
            dir_num = 0
            path_record_list = []
            for ind in range(total):
                coord = coord_list[ind]
                fold_num = dir_num//MAX_FILE_OF_DIR
                img_name = f"{wsi_name}#{coord[0]}#{coord[2]}.jpg"
                save_name = os.path.join(save_wsi_dir, str(fold_num), img_name)
                path_record = os.path.join(PATH_RECORD, dataset, f"{wsi_name}_path_record.csv")
                if os.path.isfile(save_name): 
                    dir_num +=1
                    continue

                if wsi is None:
                    wsi = openslide.open_slide(wsi_path)
                img = wsi.read_region((coord[2], coord[0]), args.X20_level, 
                                      (args.ps,args.ps)).convert('RGB')
                just_dir_of_file(save_name,info=False)
                img.save(save_name)

                fu.write_csv_row(path_record, [[f"{dataset}/{wsi_name}/{str(fold_num)}/{img_name}"]], model='a')
                dir_num +=1
            time_elapsed = time.time() - time_start
            print(f"Costing time:{time_elapsed:.2f}s")




