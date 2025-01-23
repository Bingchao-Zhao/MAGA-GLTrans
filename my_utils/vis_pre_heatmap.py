import os


import cv2
import numpy as np


from . utils import *
from . import handle_img as hi
from . import handle_huge_img as hhi

def vis_heatmap(attention, 
                ind,
                img_path, 
                save_path,
                clip_up=0.4,
                beta=0.25,
                down_rate=0.4,
                p_splite='#',
                concat=True,
                over_map=False,
                save_single_map=False,
                guass_blur_kenel=5,
                suffix='.jpg'):

    if len(ind)!=len(attention):
        err('Index({}) and attention({}) with diff size!'.\
            format(len(ind), len(attention)))

    ind = np.array(ind, dtype=np.int32)
    x_b, x_e, y_b, y_e = ind[:,0].min(), ind[:,1].max(), ind[:,2].min(), ind[:,3].max()
    if just_ff(img_path):
        img = hhi.read_wsi_with_img(img_path, p_splite=p_splite)
    elif just_ff(img_path, file=True):
        img = cv2.imread(img_path)
    else:
        err('Error img type:"{}"'.format(img_path))
        
    img = img[x_b:x_e, y_b:y_e]
    # img = hi.cv2_resize(img, down_rate)

    if os.path.isfile(img_path):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        # file_name = '{}.{}'.format(file_name, suffix)
    elif os.path.isdir(img_path):
        file_name = img_path.split('/')[-1]

    heatmap = np.zeros((x_e, y_e), dtype=np.uint8)
    if clip_up>0:
        np.clip(attention, 0, clip_up)
    # attention[attention<0]=0
    attention = (attention-attention.min()+0.00001)/(attention.max()-attention.min())
    # attention = 1/(1+(np.exp((-attention*5))))
    attention = np.array(attention*255, dtype=np.uint8)

    for i, value in enumerate(attention):
        heatmap[ind[i][0]:ind[i][1], ind[i][2]:ind[i][3]] = attention[i] 

    if guass_blur_kenel>0:
        h,w = heatmap.shape[0:2] 
        heatmap=cv2.resize(heatmap,(w//50,h//50))
        # print('blur')
        heatmap=cv2.GaussianBlur(heatmap,(guass_blur_kenel,guass_blur_kenel),15)
        heatmap=cv2.resize(heatmap,(w,h))
        
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_map = color_map[..., ::-1]
    color_map = color_map[x_b:x_e, y_b:y_e]
    heatmap = heatmap[x_b:x_e, y_b:y_e]

    if down_rate<1:
        th,tw = img.shape[0:2]
        th,tw = int(th*down_rate), int(tw*down_rate)
        color_map = cv2.resize(color_map, (tw,th))
        img = cv2.resize(img, (tw,th))
        heatmap = cv2.resize(heatmap, (tw,th))

    wsi_img_heat = cv2.addWeighted(img, 1, color_map, beta, gamma=0)
    wsi_img_heat[heatmap==0] = img[heatmap==0]
    just_ff(save_path, create_floder=True)
    if concat:
        save_name = os.path.join(save_path, '{}-concat.{}'.format(file_name, suffix))
        hi.cv2_writer(save_name, hi.concat_img([img, wsi_img_heat, heatmap], \
                nrow=1, channel_frist=False, padding=20))
    if over_map:
        save_name = os.path.join(save_path, '{}-overlap.{}'.format(file_name, suffix))
        hi.cv2_writer(save_name, wsi_img_heat)

    if save_single_map:
        save_name = os.path.join(save_path, '{}-heatmap.{}'.format(file_name, suffix))
        hi.cv2_writer(save_name, heatmap)
    
    