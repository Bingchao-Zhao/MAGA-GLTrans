from visdom import Visdom
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from . import handle_img as hi
from . utils import *
IMAGE_PADDING=4
CONCAT_IMG_ROW=2
IMG_CHANNEL_FRIST=True

def gpu_grad_to_tensor(data):
    ret = data
    if  isinstance(data, torch.Tensor):
        if data.is_cuda:
            ret = data.cpu()
        if data.requires_grad:
            ret = data.detach()
    return ret
class MyTB(object):
    def __init__(self, log_dir:str='./log', clean_log_dir=False) -> None:
        just_ff(log_dir, create_floder=True)
        if clean_log_dir:
            clean_dir(log_dir)
        self.scalars_name = {}
        self.image_name = {}
        self.writer = SummaryWriter(log_dir=log_dir)
        tips(f"Created my SummaryWriter. Log dir:'{log_dir}'")

    def record_name(self, record:dict, name:str) -> None:
        record[name] = 0 if record.get(name) is None else record[name]+1

    def add_scalar(self, x, name='Loss') -> None:
        self.record_name(self.scalars_name, name)
        x_record = gpu_grad_to_tensor(x)
        self.writer.add_scalar(name, x_record, self.scalars_name[name])

    def add_scalars(self, x:list, 
                    x_name:list=[],
                    name:str='Loss') -> None:
        """x should be a list. x_name this the name of each line. It can be []. 
        name is the name of the figure

        Args:
            x (_type_): _description_
            x_name (list, optional): _description_. Defaults to [].
            name (str, optional): _description_. Defaults to 'Loss'.
        """
        self.record_name(self.scalars_name, name)
        xs = {f"l_{str(i)}" if x_name==[] else x_name[i] : gpu_grad_to_tensor(x[i]) for i in range(len(x))}
        self.writer.add_scalars(name, xs, self.scalars_name[name])

    def add_image(self, img, 
                    name:str='Image', 
                    concat:bool=True, 
                    nrow:int=CONCAT_IMG_ROW, 
                    padding:int=IMAGE_PADDING, 
                    channel_frist:bool=IMG_CHANNEL_FRIST, 
                    update_record:bool=False) -> None:
        if self.image_name.get(name) is None or update_record:
            self.record_name(self.image_name, name)
        if concat:
            concat_list = [gpu_grad_to_tensor(i) for i in img]
            img = hi.concat_img(concat_list, channel_frist=channel_frist, nrow=nrow, padding=padding)
        dataformats = 'HWC' if img.shape[-1]<=3 else 'CHW'
        self.writer.add_image(name, img, self.image_name[name], dataformats=dataformats)    
    
    def close_writer(self) -> None:
        self.writer.close()

def my_transpose(img):
    shape = img.size()
    l = len(shape)
    assert l==3 or l==4 , 'image dimension can only be 3 or 4'
    if l==4:
        if shape[1]>shape[3]:
            return img.permute(0,3,1,2)
    else:
        if shape[0]>shape[2]:
            return img.permute(1,2,0)
    return img
    
class visdom_vis(object):
    def __init__(self,env_name='main',port = 8097):
        super(visdom_vis, self).__init__()
        self.vis = Visdom(server='http://127.0.0.1', port=port,env=env_name)
        assert self.vis.check_connection()
        
    def vis_line(self,X,Y,opts=dict(title='LOSS'),win=0):
        self.vis.line(Y=Y,X=X,win=win,opts=opts)
        
    def vis_img(self,img,opts=dict(title='img'),win=0,gray=False):
        if gray:
            if len(img.size())<4:
                img = torch.unsqueeze(img, 1)
            self.vis.images(img,win=win,opts=opts)
        else:
            img = my_transpose(img)
            self.vis.images(img,win=win,opts=opts)

class vis_register(object):
    def __init__(self, show_stride=4) -> None:
        self.vis_buff = [[]]
        self.line_buff = [[]]
        self.show_stride = show_stride
        self.record_time = 0
        pass

    def record_line(self, *args):
        while(len(self.line_buff)<len(args)):
            self.line_buff.append([])
        for num, arg in enumerate(args):
            if  isinstance(arg, torch.Tensor):
                arg = arg.item()
            self.line_buff[num].append(arg)

    def record_img(self, *args):
        self.record_time += 1

        while(len(self.vis_buff)<len(args)):
            self.vis_buff.append([])
        
        for member, arg in enumerate(args):
            if  isinstance(arg, torch.Tensor):
                if arg.is_cuda:
                    arg = arg.cpu()
                if arg.requires_grad:
                    arg = arg.detach()
            for num, value in enumerate(arg):
                self.vis_buff[member].append(value.squeeze())

    def ret_concat(self, row_inter=1, row_exter=-1, vis_inter_num=4,\
            channel_f=True, padding=10):
        new = []
        
        for num, value in enumerate(self.vis_buff):
            temp = self.ret_member(num, row_inter=row_inter,
                                    chennel_f=channel_f, 
                                    ret_num=vis_inter_num,
                                    padding=padding)
            new.append(temp)

        row_exter = len(self.vis_buff) if row_exter<0 else row_exter

        ret = hi.concat_img(new, channel_frist=channel_f, \
                nrow=row_exter, padding=padding)

        self.vis_buff = []
        return ret

    def ret_member(self, 
                    member=[], 
                    ret_num=4, 
                    row_inter=1, 
                    chennel_f=True, 
                    padding=10):

        assert len(member)<=len(self.vis_buff), \
            'member({}) is exceed buff length({})'.\
            format(len(member), len(self.vis_buff))

        if member==[]:
            member=[0]
        elif member[0] == 'all':
            member=[i for i in range(len(self.vis_buff))]

        ret = []
        for num in member:
            ret_num = len(self.vis_buff[member[num]]) if \
                    ret_num>len(self.vis_buff[member[num]]) else ret_num
            img = self.vis_buff[member[num]][0:ret_num]
            img = hi.concat_img(img, channel_frist=chennel_f, \
                    nrow=row_inter, padding=padding)
            ret.append(img)

        self.vis_buff = []
        return ret
    
    def ret_line(self):
        ret = []
        for line in self.line_buff:
            ret.append({'x':np.arange(0, len(line)), 'y':line})
        
        return ret
    
    def just_show(self):
        if self.record_time<=1 and self.show_stride!=1:
            return False
        
        return self.record_time%self.show_stride == 0

