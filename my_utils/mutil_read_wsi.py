# !/usr/bin/env python
# coding: utf-8
# python模拟linux的守护进程
from signal import SIGTERM
import math
import sys, os, time
import time
import argparse
import my_utils.multi_process as mp

from my_utils.handle_huge_img import crop_and_save_wsi
from my_utils.utils import *
import my_utils.file_util as fu
__metaclass__ = type
CONFIG_FILE = "infer.yaml"
RECORD_FILE = './record/splite_wsi.csv'
CONFIG = os.path.join(os.getcwd(), CONFIG_FILE)
RECORD_PATH = os.path.join(os.getcwd(),"postprocess_record.txt")
DATA_ROOT = ""
SAVE_PATH = ''
PATCH_SIZE = 256*16
def postprocessor(data_list, child_num):
    pr = progress_predictor(len(data_list))
    for num, svs_path in enumerate(data_list):
        flow("processor:{}.Begin to handle: {}/{}. img:{}".\
                format(child_num, num, len(data_list), svs_path))
        flow("Remaining time:{}. {}(img/H)".format(pr.predict(), pr.rate()))
        handle = crop_and_save_wsi(svs_path, DATA_ROOT, SAVE_PATH, patch_size=PATCH_SIZE)
        if not handle:
            pr(record_time=False)
        pr()
        write_record(svs_path, mod='a')
        info(f'Writed the record: {get_name_from_path(svs_path)}')
    info("Childen {} finish process. return".format(child_num))

def write_record(p, mod='a'):
    p, n = os.path.split(p)
    n, suffix = os.path.splitext(n)
    fu.write_csv_row(RECORD_FILE, [[n]], model=mod)

class Daemon:
    ERROR_FILE = ['TCGA-44-2656-01A-04-TS4.D00DB24E-08D4-4E55-AA28-4C8F707F1308']
    def __init__(self, pidfile="/tmp/Daemon.pid", stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
        self.has_processed_file = self.read_record()
        pass

    def read_record(self):
        if not os.path.isfile(RECORD_FILE):
            return []
        has_processed = fu.csv_reader(RECORD_FILE)
        has_processed = [p[0] for p in has_processed]
        return has_processed
    

    def is_failed_file(self, p):
        if p.find('-DX')>=0:
            return True
        return False

    def screening_path(self, path_list):
        ret = []
        for p in path_list:
            if self.is_failed_file(p) or complete_path_in_list(p, self.ERROR_FILE):
                continue
            ret.append(p)
        flow(f'Ori path_num:{len(path_list)}. After screening:{len(ret)}')
        return ret

    def _run(self, child_num=4, test=False):
        patients_list = find_file(DATA_ROOT, 3, suffix='.svs')
        patients_list = self.screening_path(patients_list)
        new_patients_list = []
        
        for num, p in enumerate(patients_list):
            # if p.find('TCGA_LUSC_ALL')>=0:
            #     print('TCGA_LUSC_ALL image, skip.')
            #     continue
            # if p.find('0-DX')<0:
            #     print('Not DX image, skip.')
            #     continue
            dir_name = fu.split_path(DATA_ROOT, input_path=os.path.split(p)[0])
            floder = os.path.splitext(os.path.split(p)[-1])[0]
            save_path_nX = os.path.join(SAVE_PATH, '40X', dir_name, floder)
            # if p.find('TCGA-43-6647-01A-01-TS1.f8dfefc5-8e89-462b-81cb-53e8b85bf85b')<0:
            #     continue
            if complete_path_in_list(p, self.has_processed_file):
                continue
            if not just_ff(save_path_nX, info=False) or\
                    crop_and_save_wsi(p, DATA_ROOT, SAVE_PATH, just_hanle=True, patch_size=PATCH_SIZE):
                new_patients_list.append(p)
                continue

            write_record(p)
            info("{}/{} has been crop! {}".format(num, len(patients_list),p))

        flow(f'Ori num: {len(patients_list)}. Process num:{len(new_patients_list)}')
        child_process_batch = MyZip(new_patients_list, batch=math.ceil(len(new_patients_list)/child_num))

        if test:
            for p in new_patients_list:

                self.exist_flag = postprocessor([p], 
                                                child_num=0)
        else:
            for child_index, [child_batch] in enumerate(child_process_batch):
                pid = os.fork()
                if pid == 0:
                    info("Child {} start".format(child_index))
                    self.exist_flag = postprocessor(child_batch, 
                                                    child_num=child_index)
        while True:
            time.sleep(10)
            p,status = os.waitpid(0, os.WNOHANG|os.WUNTRACED|os.WCONTINUED)  
            if mp.childCount()<=0:
                info("All children have finish.")
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=5, 
                    help='child num')
    parser.add_argument('--test', type=int, choices=[0,1],default=1, 
                    help='False is mutil process')
    args = parser.parse_args()
    tips(f"Child num: {args.num}")
    tips(f"Model: {'One process' if args.test else 'Mutil process'}")
    daemon = Daemon('/tmp/nuclei_process.pid', stdout='/tmp/watch_stdout.log')
    # daemon.stop()
    daemon._run(child_num=args.num, test=args.test)

    sys.exit(0)