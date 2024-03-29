import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d




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

class FinetuneEfficientNet(BaseModel):
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

    def forward(self, x, train_state=False):
        x = x.float()
        feats = self.backbone.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
        # return self.fc(feats)

class Covid19(BaseModel):
    def __init__(self, model_name, inchannels=3, num_classes=1, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_name, in_chans=inchannels, pretrained=True)
        n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.backbone.eval()
        # self.classifier = nn.Linear(n_features, 1)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_features, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward_features(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return x

    def forward(self, x, train_state=False):
        x = x.float()
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
        # x = self.classifier(x)
        # return x

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