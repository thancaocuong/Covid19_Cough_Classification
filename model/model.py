import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d
from .cnn14 import Cnn14

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size=512, hop_size=512, mel_bins=128, fmin=0, 
        fmax=48000, classes_num=1, freeze_base=True):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.head = torch.nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, classes_num, bias=True)
        )
        if freeze_base:
            print("freeze base layer")
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        # self.init_weights()

    def init_weights(self):
        # init_layer(self.fc_transfer)
        pass

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        embedding = self.base(input, mixup_lambda)

        logit =  self.head(embedding) 
        return logit

class LargerPlainCNN(BaseModel):
    def __init__(self, inchannels=1, num_classes=1, use_coord=False, pretrained=True):
        super().__init__()
        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=128,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(128)
        self.act0 = nn.LeakyReLU(0.1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=128, 
                              out_channels=256,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.LeakyReLU(0.1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.act0(x)

        x = self.max_pool1(x)

        x = self.conv2(x)

        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.pool(x).view(x.size(0), -1)
        #x = self.flatten(x)

        x = F.leaky_relu(self.fc1(x), 0.1)

        x = self.drop1(x)

        x = F.leaky_relu(self.fc2(x), 0.05)

        x = self.drop2(x)

        x = self.fc3(x)
        return x

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