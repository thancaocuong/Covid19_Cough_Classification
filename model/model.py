import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d
from .vit import *
from torch.cuda import amp


class PlainCNN(BaseModel):
    def __init__(self, inchannels=3, num_classes=1, use_coord=False, pretrained=True):
        super().__init__()
        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.act0 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=64, 
                              out_channels=256,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.act0(x)

        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.act1(x)

        x = self.bn1(x)
        
        x = self.pool(x).view(x.size(0), -1)
        #x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.drop1(x)

        x = F.relu(self.fc2(x))

        x = self.drop2(x)

        x = self.fc3(x)
        return x

class PlainCNNSmall(BaseModel):
    def __init__(self, inchannels=3, num_classes=1, use_coord=False, pretrained=True):
        super().__init__()
        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.act0 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=64, 
                              out_channels=128,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.act0(x)

        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.act1(x)

        x = self.bn1(x)
        
        x = self.pool(x).view(x.size(0), -1)
        #x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.drop1(x)

        x = F.relu(self.fc2(x))

        x = self.drop2(x)

        x = self.fc3(x)
        return x

class TimmBackbone(BaseModel):
    def __init__(self, model_name, inchannels=3, num_classes=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, in_chans=inchannels, pretrained=pretrained)
        n_features = self.backbone.num_features
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def freeze(self):
        # pass
        # print("freeze feature_extractor")
        for param in self.backbone.parameters():
            param.require_grad = False

    def unfreeze(self):
        # pass
        for param in self.backbone.parameters():
            param.require_grad = True

    def forward(self, x, fp16=False):
        with amp.autocast(enabled=fp16):
            x = x.float()
            feats = self.backbone.forward_features(x)
            x = self.pool(feats).view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = self.fc2(x)
        return x

class TropicModel(BaseModel):
    def __init__(self, model_name, drop_rate=0.2, inchannels=3, num_classes=1, coord=1, pretrained=True):
        super().__init__()
        self.trunk = timm.create_model(model_name,pretrained=pretrained,num_classes=num_classes,in_chans=inchannels+coord)
        self.do = nn.Dropout2d(drop_rate)
        self.coord = coord
    def forward(self, x):
        bs,_,freq_bins,time_bins = x.size()
        coord = torch.linspace(-1,1,freq_bins,dtype=x.dtype,device=x.device).view(1,1,-1,1).expand(bs,1,-1,time_bins)
        if self.coord: x = torch.cat((x,coord),dim=1)
        x = self.do(x)
        return self.trunk(x)


class VitModel(BaseModel):
    def __init__(self, backbone, out_dim, embedding_size=512, 
                 loss=False, pretrained=True):
        super(VitModel, self).__init__()
        self.backbone_name = backbone
        self.loss = loss
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        
        self.backbone = Backbone(backbone, pretrained=pretrained)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
            
        self.head = nn.Linear(self.embedding_size, out_dim)
        
    def forward(self, melspec, get_embeddings=False, get_attentions=False, fp16 = False):
        with amp.autocast(enabled=fp16):
            # x = input_dict['spect']
            x = melspec
            x = x.unsqueeze(1)
            x = x.expand(-1, 3, -1, -1)

            x = self.backbone(x)
            
            # if 'vit' not in backbone:
            #     x = self.global_pool(x)
            #     x = x[:,:,0,0]
            # if 'vit_deit_base_distilled_patch16_384' == backbone:
            x = x[0] + x[1]
            
            x = self.neck(x)

            logits = self.head(x)

        return logits