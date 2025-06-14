import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')

import seaborn as sns

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


def save_best_epoch(model, epoch, optimizer, model_params, val_logloss, save_data_params):
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

    wrong_imgs = save_data_params["wrong_imgs"]

    if os.path.exists(best_epoch_wrong_dir):
        shutil.rmtree(best_epoch_wrong_dir)
    os.makedirs(best_epoch_wrong_dir, exist_ok=True)
    for wrong_img in wrong_imgs:
        shutil.copy(wrong_img, os.path.join(best_epoch_wrong_dir, os.path.basename(wrong_img)))

    cm = save_data_params["confusion_matrix"]
    idx_to_class = save_data_params["idx_to_class"]
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    plt.figure(figsize=(50, 50))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (Epoch {epoch + 1})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['WRONG_DIR'], f"best_epoch_{epoch + 1}_confusion_matrix.png"))