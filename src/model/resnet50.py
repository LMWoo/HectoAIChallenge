import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True