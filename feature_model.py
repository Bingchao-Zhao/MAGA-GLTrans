from turtle import down, forward
import torch
import torch.nn as nn

# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': './resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers, mod='ori'):
        self.inplanes = 64
        self.mod=mod
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        if mod != 'ori':
            self.align = nn.Linear(1024,1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1_f = self.layer1(x)
        l2_f = self.layer2(l1_f)
        l3_f = self.layer3(l2_f)

        fc_f = self.avgpool(l3_f)
        fc_f = fc_f.view(fc_f.size(0), -1)
        
        if self.mod != 'ori':
            fc_f = self.align(fc_f)

        return [l1_f, l2_f, l3_f, fc_f]

def resnet50_baseline(pretrained=False, mod='ori'):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3], mod=mod)
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    # pretrained_dict = model_zoo.load_url(model_urls[name])
    pretrained_dict = torch.load(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# for name, module in model._modules.items():
#     for p in module.parameters():
#         print(p.requires_grad)

class AlignNetwork(nn.Module):
    def __init__(self, downsample_rate=0.25):
        super(AlignNetwork, self).__init__()
        self.downsample_rate = downsample_rate
        self.h_model = resnet50_baseline(pretrained=True)
        for param in self.h_model.named_parameters():
            param[1].requires_grad = False

        self.l_model = resnet50_baseline(pretrained=True, mod='align')
        self.rate = [1,5,10,100]
    def preprocess(self, h_img):
        input_size = [h_img.shape[-2], h_img.shape[-1]]
        new_size = [int(i*self.downsample_rate) for i in input_size]
        l_img = F.interpolate(h_img, size=(64, 64))
        l_img = F.interpolate(l_img, size=[256, 256])
        return l_img

    def caculate_loss_same_resolution(self, h_feature, l_feature, loss_fn):
        total_loss = []
        for i in range(len(h_feature)):
            h_f, l_f = h_feature[i], l_feature[i]
            total_loss.append(loss_fn[i](h_f, l_f)*self.rate[i])
        
        return total_loss

    def caculate_loss(self, h_feature, l_feature, loss_fn):
        total_loss = []
        for i in range(len(h_feature)):
            h_f, l_f = h_feature[i], l_feature[i]
            # the last is the feature vector: 1*1024
            if i <len(h_feature)-1:
                size = [l_f.shape[-3], l_f.shape[-2], l_f.shape[-1]]
                h_f = F.interpolate(h_f.unsqueeze(1), size=size).squeeze(1)

            total_loss.append(loss_fn[i](h_f, l_f))
        
        return total_loss

    def forward(self, h_img, loss_fn, mod='infer'):
        l_img = self.preprocess(h_img)
        l_img = Variable(l_img.float().cuda())
        h_img = Variable(h_img.float().cuda())

        if mod != 'train':
            return self.l_model(l_img)[-1]

        h_feature = self.h_model(h_img)
        l_feature = self.l_model(l_img)

        total_loss = self.caculate_loss_same_resolution(h_feature, l_feature, [loss_fn for _ in range(len(h_feature))])

        return total_loss, l_feature[-1]