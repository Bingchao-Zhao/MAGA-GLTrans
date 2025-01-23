from openslide import OpenSlide
import pyvips
import numpy as np
from math import ceil
import openslide
import os
import tifffile
import cv2
from tqdm import tqdm
import time
import glob
import copy

TILE_SIZE = 512

gfi = lambda img,ind : copy.deepcopy(img[ind[0]:ind[1], ind[2]:ind[3]])


def find_file(path,depth_down,depth_up=0,suffix='.xml'):
    ret = []
    for i in range(depth_up,depth_down):
        _path = os.path.join(path,'*/'*i+'*'+suffix)
        ret.extend(glob.glob(_path))
    ret.sort()
    return ret

def up_to16_manifi(hw):
    return int(ceil(hw[0]/TILE_SIZE)*TILE_SIZE), int(ceil(hw[1]/TILE_SIZE)*TILE_SIZE)

def gen_im(wsi, index):
    ind = 0
    while True:
        temp_img = gfi(wsi, index[ind])
        ind+=1
        yield temp_img
def get_name_from_path(file_path:str, ret_all:bool=False):
    dir, n = os.path.split(file_path)
    n, suffix = os.path.splitext(n)
    if ret_all:
        return dir, n, suffix
    return n

def gen_patches_index(ori_size, *, img_size=224, stride = 224,keep_last_size = False):

    """
        这个函数用来按照输入的size和patch大小，生成每个patch所在原始的size上的位置

        keep_last_size：表示当size不能整除patch的size的时候，最后一个patch要不要保持输入的img_size
        
        返回：
            一个np数组，每个成员表示当前patch所在的x和y的起点和终点如：
                [[x_begin,x_end,y_begin,y_end],...]
    """
    height, width = ori_size[:2]
    index = []
    if height<img_size or width<img_size: 
        print("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        return index
        
    for h in range(0, height+1, stride):
        xe = h+img_size
        if h+img_size>height:
            xe = height
            h = xe-img_size if keep_last_size else h

        for w in range(0, width+1, stride):
            ye = w+img_size
            if w+img_size>width:
                ye = width
                w = ye-img_size if keep_last_size else w
            index.append(np.array([h, xe, w, ye]))

            if ye==width:
                break
        if xe==height:
            break
    return index

def just_ff(path:str,*,file=False,floder=True,create_floder=False, info=True):
    """
    Check the input path status. Exist or not.

    Args:
        path (str): _description_
        file (bool, optional): _description_. Defaults to False.
        floder (bool, optional): _description_. Defaults to True.
        create_floder (bool, optional): _description_. Defaults to False.
        info (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if file:
        return os.path.isfile(path)
    elif floder:
        if os.path.exists(path):
            return True
        else:
            if create_floder:
                try:
                    os.makedirs(path) 
                    if info:
                        print(r"Path '{}' does not exists, but created ！！".format(path))
                    return True
                except ValueError:
                    if info:
                        print(r"Path '{}' does not exists, and the creation failed ！！".format(path))
                    pass
            else:
                if info:
                    print(r"Path '{}' does not exists！！".format(path))
                return False
                

def just_dir_of_file(file_path:str, create_floder:bool=True):
    """_summary_
    Check the dir of the input file. If donot exist, creat it!
    Args:
        file_path (_type_): _description_
        create_floder (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    _dir = os.path.split(file_path)[0]
    return just_ff(_dir, create_floder = create_floder)

def split_path(root_path:str, input_path:str):
    path_split = os.sep
    while(root_path[-1]==path_split):
        root_path = root_path[0:len(root_path)-1]
    ret_path = input_path[len(root_path):len(input_path)]
    if len(ret_path) == 0:
        return ''
    while(ret_path[0]==path_split):
        ret_path = ret_path[1:len(ret_path)]
    return ret_path

def gen_pyramid_tiff(in_file, out_file, select_level=0):
    '''
    select_level 决定了读取那一层。第0层是倍率最高的。
    '''
    svs_desc = 'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
    label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
    macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'
    # 指定mpp值
    odata = openslide.open_slide(in_file)
    # 获取当前图像的MPP
    # 如果不是这个字段，可以找一下是哪个
    mpp = float(odata.properties['mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X'])#0.5
    # 指定缩放倍率
    mag = 40
    #mpp 在0.25左右的是40X，0.5左右的是20X
    if mpp<=0.3:
        mag = 20
        mpp = mpp*2
    # 换算mpp值到分辨率
    resolution = [10000 / mpp, 10000 / mpp, 'CENTIMETER']
    
    # 指定图像名字
    if odata.properties.get('aperio.Filename') is not None:
        filename = odata.properties['aperio.Filename']
    else:
        filename = get_name_from_path(in_file)


    print(f"loading '{in_file}'")
    start = time.time()
    # image_py = openslide.open_slide(wsi_path)
    # pyvip 比openslide更快，但如果没有，用openslide也是可以的
    image_py = pyvips.Image.openslideload(in_file, level = select_level)
    image = np.array(image_py)[...,0:3]
    print(f"finish loading '{in_file}'. costing time:{time.time()-start}")
    # image = np.array(image_py['20X'])
    # 缩略图
    thumbnail_im = np.zeros([762, 762, 3], dtype=np.uint8)
    thumbnail_im = cv2.putText(thumbnail_im, 'thumbnail', (thumbnail_im.shape[1]//4, thumbnail_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(255, 0, 0), thickness=3)
    # 标签图
    label_im = np.zeros([762, 762, 3], dtype=np.uint8)
    label_im = cv2.putText(label_im, 'label', (label_im.shape[1]//4, label_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 255, 0), thickness=3)
    # 宏观图
    macro_im = np.zeros([762, 762, 3], dtype=np.uint8)
    macro_im = cv2.putText(macro_im, 'macro', (macro_im.shape[1]//4, macro_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 0, 255), thickness=3)

    # tile 大小
    tile_hw = np.int64([TILE_SIZE, TILE_SIZE])

    width, height = image.shape[0:2]
    # 要需要的金字塔分辨率
    multi_hw = np.int64([(width, height), (width//2, height//2), 
                         (width//4, height//4), (width//8, height//8),
                         (width//16, height//16),
                         (width//32, height//32),
                         (width//64, height//64)])

    # 尝试写入 svs 格式
    with tifffile.TiffWriter(out_file, bigtiff=True) as tif:
        thw = tile_hw.tolist()
        # outcolorspace 要保持为默认的 YCbCr，不能使用rgb，否则颜色会异常
        # 95 是默认JPEG质量，值域是 0-100，值越大越接近无损
        compression = ['JPEG', 100, dict(outcolorspace='YCbCr')]
        # compression = 'JPEG'
        kwargs = dict(subifds=0, photometric='rgb', planarconfig='CONTIG', compression=compression, dtype=np.uint8, metadata=None)

        for i, hw in enumerate(multi_hw):
            hw =  up_to16_manifi(hw)
            temp_wsi = cv2.resize(image, (hw[1], hw[0]))
            new_x, new_y = up_to16_manifi(hw)
            new_wsi = np.ones((new_x, new_y, 3),dtype=np.uint8)*255
            new_wsi[0:hw[0], 0:hw[1],:] = temp_wsi[...,0:3]
            index = gen_patches_index((new_x, new_y),img_size=TILE_SIZE,stride=TILE_SIZE)
            gen = gen_im(new_wsi, index)

            if i == 0:
                desc = svs_desc.format(mag=mag, filename=filename, mpp=mpp)
                # tif.write(data=gen, resolution=resolution, description=desc, **kwargs)
                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description=desc, **kwargs)
                _hw = up_to16_manifi(multi_hw[-2])
                thumbnail_im = cv2.resize(image, (_hw[1], _hw[0]))[...,0:3]
                tif.write(data=thumbnail_im, description='', **kwargs)
            else:

                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description='', **kwargs)
        _hw = up_to16_manifi(multi_hw[-2])
        macro_im = cv2.resize(image, (_hw[1], _hw[0]))[...,0:3]
        tif.write(data=macro_im, subfiletype=9, description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]), **kwargs)

# mrxs所在的目录
DATA_DIR = '/media/zbc/hanchu001/'
#保存目录
SAVE_DIR = '/media/zbc/hanchu001/'

wsi_list = find_file(DATA_DIR, 1, suffix='.mrxs')

for w_name in tqdm(wsi_list):
    t1 = time.perf_counter()
    patient_name = w_name.split(os.path.sep)[-2]
    wsi_name = get_name_from_path(w_name)
    diff_path = split_path(DATA_DIR, get_name_from_path(w_name, ret_all=True)[0])
    save_path = os.path.join(SAVE_DIR, diff_path, f'{wsi_name}.svs')
    # 如果svs文件已经存在，则跳过
    if just_ff(save_path,file=True):
        continue
    just_dir_of_file(save_path)
    gen_pyramid_tiff(w_name, save_path)
    print(f'{wsi_name}:',time.perf_counter() - t1)