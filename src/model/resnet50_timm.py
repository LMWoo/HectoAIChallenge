import os
import shutil

import torch
import torch.nn as nn
import timm

class Resnet50_Timm(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50_Timm, self).__init__()
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=num_classes
        )   
        
        for name, param in self.backbone.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # self.feature_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()
        # self.head = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        # x = self.head(x)
        return x