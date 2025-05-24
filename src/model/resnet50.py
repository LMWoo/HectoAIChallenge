import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from src.utils.utils import model_dir, save_hash, CFG

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def save_best_epoch(model, epoch, optimizer, model_params, val_logloss, wrong_imgs):
    save_dir = model_dir(CFG['EXPERIMENT_NAME'])
    os.makedirs(save_dir, exist_ok=True)

    dst = os.path.join(save_dir, f"best_model.pth")
    torch.save({
        "epoch": epoch,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_logloss": val_logloss,
    }, dst)
    save_hash(dst)

    print(f"Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

    best_epoch_wrong_dir = os.path.join(CFG['WRONG_DIR'], 'best_model')

    if os.path.exists(best_epoch_wrong_dir):
        shutil.rmtree(best_epoch_wrong_dir)
    os.makedirs(best_epoch_wrong_dir, exist_ok=True)
    for wrong_img in wrong_imgs:
        shutil.copy(wrong_img, os.path.join(best_epoch_wrong_dir, os.path.basename(wrong_img)))