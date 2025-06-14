import os
import shutil
import datetime

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')

from src.utils.utils import CFG, model_dir, save_hash

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

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    for images, labels, img_paths in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss

def validation_one_epoch(model, val_loader, model_params, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_preds = []
    wrong_imgs = []
    with torch.no_grad():
        for images, labels, img_paths in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # LogLoss
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            wrong_mask = (preds != labels).cpu()
            for idx, is_wrong in enumerate(wrong_mask):
                if is_wrong:
                    wrong_imgs.append(img_paths[idx])

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(model_params["num_classes"])))

    cm = confusion_matrix(all_labels, all_preds)

    save_data_params = {
        "wrong_imgs" : wrong_imgs,
        "confusion_matrix": cm,
        "idx_to_class": val_loader.dataset.dataset.idx_to_class
    }
    
    return avg_val_loss, val_accuracy, val_logloss, save_data_params

def train(model, train_loader, val_loader, model_params, criterion, optimizer, device):
    best_logloss = float('inf')

    for epoch in range(CFG['EPOCHS']):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        avg_val_loss, val_accuracy, val_logloss, save_data_params = validation_one_epoch(model, val_loader, model_params, criterion, device, epoch)
        wandb.log({"Loss/Train": avg_train_loss})
        wandb.log({"Loss/Valid": avg_val_loss})
        wandb.log({"LogLoss/Valid": val_logloss})
        wandb.log({"Accuracy/Valid": val_accuracy})
        
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            save_best_epoch(model, epoch, optimizer, model_params, val_logloss, save_data_params)