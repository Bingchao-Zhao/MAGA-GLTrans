import time
import numpy as np
from math import ceil
import openslide
import tifffile
# import pyvips
from tqdm import tqdm
import cv2
import my_utils.file_util as fu
from my_utils.utils import *
import my_utils.handle_huge_img as hhi
Error_record = 'record/Error_rebuild_svs.csv'
DATA_DIR = '/media/zbc/nanfang_data/南方医院项目原始数据/zbc_glioma/青岛/第一批/astroblastoma'
SAVE_DIR = 'test_wsi'
SUFFIX = '.svs'

TILE_SIZE = 32
def up_to16_manifi(hw):
    return int(ceil(hw[0]/TILE_SIZE)*TILE_SIZE), int(ceil(hw[1]/TILE_SIZE)*TILE_SIZE)

def gen_im(wsi, index):
    ind = 0
    while True:
        temp_img = hhi.gfi(wsi, index[ind])
        ind+=1
        yield temp_img

def gen_pyramid_tiff(in_file, out_file):
    svs_desc = 'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
    label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
    macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'

    odata = openslide.open_slide(in_file)
    if odata.properties.get('aperio.MPP') is None:
        fu.write_csv_row(Error_record, [[Error_record]])
        return
    mpp = float(odata.properties['aperio.MPP'])#0.5

    mag = int(float(odata.properties['aperio.AppMag']))
    if mag==40:
        mag = 20
        mpp = mpp*2

    resolution = [10000 / mpp, 10000 / mpp, 'CENTIMETER']
    

    if odata.properties.get('aperio.Filename') is not None:
        filename = odata.properties['aperio.Filename']
    else:
        filename = get_name_from_path(in_file)

    image_py = hhi.crop_and_save_wsi(in_file, return_img=True, write_img=False, resolution=['20X'],
                            input_root='/media/zhaobingchao/hanchu0011/glioma/data/raw_data/TCGA/TCGA_Glioma_DX/TCGA-02-0003/',
                            save_root='')
    image = np.array(image_py['20X'])

    thumbnail_im = np.zeros([762, 762, 3], dtype=np.uint8)
    thumbnail_im = cv2.putText(thumbnail_im, 'thumbnail', (thumbnail_im.shape[1]//4, thumbnail_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(255, 0, 0), thickness=3)

    label_im = np.zeros([762, 762, 3], dtype=np.uint8)
    label_im = cv2.putText(label_im, 'label', (label_im.shape[1]//4, label_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 255, 0), thickness=3)

    macro_im = np.zeros([762, 762, 3], dtype=np.uint8)
    macro_im = cv2.putText(macro_im, 'macro', (macro_im.shape[1]//4, macro_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 0, 255), thickness=3)


    tile_hw = np.int64([TILE_SIZE, TILE_SIZE])

    width, height = image.shape[0:2]

    multi_hw = np.int64([(width, height), (width//2, height//2), 
                         (width//4, height//4), (width//8, height//8),
                         (width//16, height//16),
                         (width//32, height//32),
                         (width//64, height//64)])


    with tifffile.TiffWriter(out_file, bigtiff=True) as tif:
        thw = tile_hw.tolist()
        compression = ['JPEG', 100, dict(outcolorspace='YCbCr')]
        kwargs = dict(subifds=0, photometric='rgb', planarconfig='CONTIG', compression=compression, dtype=np.uint8, metadata=None)

        for i, hw in enumerate(multi_hw):
            hw =  up_to16_manifi(hw)
            temp_wsi = cv2.resize(image, (hw[1], hw[0]))
            new_x, new_y = up_to16_manifi(hw)
            new_wsi = np.ones((new_x, new_y, 3),dtype=np.uint8)*255
            new_wsi[0:hw[0], 0:hw[1],:] = temp_wsi[...,0:3]
            index = hhi.gen_patches_index((new_x, new_y),img_size=TILE_SIZE,stride=TILE_SIZE)
            gen = gen_im(new_wsi, index)

            if i == 0:
                desc = svs_desc.format(mag=mag, filename=filename, mpp=mpp)
                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description=desc, **kwargs)
                _hw = up_to16_manifi(multi_hw[-2])
                thumbnail_im = cv2.resize(image, (_hw[1], _hw[0]))[...,0:3]
                tif.write(data=thumbnail_im, description='', **kwargs)
            else:

                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description='', **kwargs)
        _hw = up_to16_manifi(multi_hw[-2])
        macro_im = cv2.resize(image, (_hw[1], _hw[0]))[...,0:3]
        tif.write(data=macro_im, subfiletype=9, description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]), **kwargs)

wsi_list = find_file(DATA_DIR, 3, suffix=SUFFIX)
error_file = []
for w_name in tqdm(wsi_list):
    t1 = time.perf_counter()
    patient_name = w_name.split(os.path.sep)[-2]
    wsi_name = os.path.split(w_name)[-1]
    diff_path = fu.split_path(DATA_DIR, get_name_from_path(w_name, ret_all=True)[0])
    save_path = os.path.join(SAVE_DIR, diff_path, wsi_name)
    if just_ff(save_path,file=True) or get_name_from_path(w_name) in error_file:
        continue
    just_dir_of_file(save_path)
    gen_pyramid_tiff(w_name, save_path)
    print(f'{wsi_name}:',time.perf_counter() - t1)
