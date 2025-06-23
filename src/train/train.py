import os
import shutil
import datetime
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, classification_report
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
from timm.data.mixup import Mixup

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * param)
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)

    def state_dict(self):
        return self.ema_model.state_dict()

def top_n_confusion_matrix(epoch, top_n, cm, labels):
    cm = np.array(cm)
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)

    nonzero_flat_indices = np.flatnonzero(cm_offdiag)
    sorted_indices = nonzero_flat_indices[np.argsort(cm_offdiag.ravel()[nonzero_flat_indices])[::-1]]
    top_indices = sorted_indices[:top_n]
    top_pairs = np.array(np.unravel_index(top_indices, cm.shape)).T

    rows = [i for i, _ in top_pairs]
    cols = [j for _, j in top_pairs]
    unique_idxs = sorted(set(rows + cols))

    sub_cm = cm[np.ix_(unique_idxs, unique_idxs)]
    sub_labels = [labels[i] for i in unique_idxs]

    fig = plt.figure(figsize=(max(8, len(unique_idxs) * 0.7), max(6, len(unique_idxs) * 0.6)))
    sns.heatmap(sub_cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=sub_labels, yticklabels=sub_labels,
                cbar=True, square=True, linewidths=0.5, linecolor='gray')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"Top-{top_n} Confused Classes (Epoch {epoch + 1})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return fig


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

    top_n = 30
    top_n_fig = top_n_confusion_matrix(epoch, 30, cm, labels)
    top_n_cm_path = os.path.join(CFG['WRONG_DIR'], f"best_epoch_{epoch + 1}_top_n_{top_n}_confusion_matrix.png")
    top_n_fig.savefig(top_n_cm_path)
    plt.close(top_n_fig)

    plt.figure(figsize=(50, 50))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (Epoch {epoch + 1})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    full_cm_path = os.path.join(CFG['WRONG_DIR'], f"best_epoch_{epoch + 1}_confusion_matrix.png")
    plt.savefig(full_cm_path)
    plt.close()

    wandb.log({
        "ConfusionMatrix/Full": wandb.Image(full_cm_path),
        "ConfusionMatrix/Top_n/image": wandb.Image(top_n_cm_path),
        "ConfusionMatrix/Top_n/top_n": top_n,
    })

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss = 0.0
    for images, labels, img_paths in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss

def train_one_epoch_with_ema_mixup(model, ema, mixup_fn, train_loader, criterion, optimizer, scheduler, device, update_epoch, epoch):
    model.train()
    train_loss = 0.0
    for images, labels, img_paths in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)

        if epoch < update_epoch:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if epoch >= update_epoch:
            ema.update(model)
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss

def train_one_epoch_with_ema(model, ema, train_loader, criterion, optimizer, scheduler, device, update_epoch, epoch):
    model.train()
    train_loss = 0.0
    for images, labels, img_paths in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if epoch >= update_epoch:
            ema.update(model)
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

    idx_to_class = val_loader.dataset.dataset.idx_to_class
    class_names = val_loader.dataset.dataset.classes

    save_data_params = {
        "wrong_imgs" : wrong_imgs,
        "confusion_matrix": cm,
        "idx_to_class": idx_to_class
    }


    report_dict = classification_report(all_labels, all_preds, labels=list(range(len(idx_to_class))), output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T

    precision_scores = [float(report_df.loc[str(i), "precision"]) for i in range(len(idx_to_class))]
    recall_scores = [float(report_df.loc[str(i), "recall"]) for i in range(len(idx_to_class))]
    f1_scores = [float(report_df.loc[str(i), "f1-score"]) for i in range(len(idx_to_class))]

    precision_table = wandb.Table(data=[[cls, v] for cls, v in zip(class_names, precision_scores)],
                                columns=["label", "value"])
    recall_table = wandb.Table(data=[[cls, v] for cls, v in zip(class_names, recall_scores)],
                            columns=["label", "value"])
    f1_table = wandb.Table(data=[[cls, v] for cls, v in zip(class_names, f1_scores)],
                        columns=["label", "value"])

    wandb.log({
        "PerClass/Precision/Valid": wandb.plot.bar(precision_table, "label", "value", title="Per-class Precision"),
        "PerClass/Recall/Valid": wandb.plot.bar(recall_table, "label", "value", title="Per-class Recall"),
        "PerClass/F1-Score/Valid": wandb.plot.bar(f1_table, "label", "value", title="Per-class F1 Score")
    }, step=epoch)

    return avg_val_loss, val_accuracy, val_logloss, save_data_params

def train(model, train_loader, val_loader, model_params, criterion, optimizer, scheduler, freeze_epochs, device):
    best_logloss = float('inf')
    patience = 10
    trigger_times = 0

    ema = EMA(model, decay=0.999)

    mixup_fn = Mixup(
        mixup_alpha=CFG.get('mixup_alpha', 0.2),
        cutmix_alpha=CFG.get('cutmix_alpha', 1.0),
        cutmix_minmax=None,
        prob=CFG.get('mixup_prob', 1.0),
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.0,
        num_classes=model_params['num_classes']
    )

    mixup_fn = Mixup(
        mixup_alpha=0.4,
        cutmix_alpha=0.0,
        prob=1.0,
        switch_prob=0.0,
        mode='batch',
        label_smoothing=0.0,
        num_classes=model_params['num_classes']
    )

    for epoch in range(CFG['EPOCHS']):
        if epoch == freeze_epochs:
            print(f"Epoch {epoch+1}: Start Feature Extractor unfreeze and full-model fine-tuning")
            model.unfreeze()
        
        if ema is not None:
            if mixup_fn is not None:
                avg_train_loss = train_one_epoch_with_ema_mixup(model, ema, mixup_fn, train_loader, criterion, optimizer, scheduler, device, freeze_epochs, epoch)
            else:
                avg_train_loss = train_one_epoch_with_ema(model, ema, train_loader, criterion, optimizer, scheduler, device, freeze_epochs, epoch)
            avg_val_loss, val_accuracy, val_logloss, save_data_params = validation_one_epoch(model, val_loader, model_params, criterion, device, epoch)
            ema_avg_val_loss, ema_val_accuracy, ema_val_logloss, ema_save_data_params = validation_one_epoch(ema.ema_model, val_loader, model_params, criterion, device, epoch)
            wandb.log({"Loss/Train": avg_train_loss})
            wandb.log({"Loss/Valid": avg_val_loss})
            wandb.log({"LogLoss/Valid": val_logloss})
            wandb.log({"Accuracy/Valid": val_accuracy})
            wandb.log({"LogLoss/EMA": ema_val_logloss})
            wandb.log({"Accuracy/EMA": ema_val_accuracy})

            if scheduler is not None:
                wandb.log({"LearningRate/Train": scheduler.get_last_lr()[0]})
            else:
                wandb.log({"LearningRate/Train": CFG["LEARNING_RATE"]})
            
            print(f"[Epoch {epoch+1}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Val LogLoss: {val_logloss:.4f} || "
                f"EMA Val Loss: {ema_avg_val_loss:.4f} | "
                f"EMA Val Acc: {ema_val_accuracy:.2f}% | "
                f"EMA LogLoss: {ema_val_logloss:.4f}")

        else:
            avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
            avg_val_loss, val_accuracy, val_logloss, save_data_params = validation_one_epoch(model, val_loader, model_params, criterion, device, epoch)
            wandb.log({"Loss/Train": avg_train_loss})
            wandb.log({"Loss/Valid": avg_val_loss})
            wandb.log({"LogLoss/Valid": val_logloss})
            wandb.log({"Accuracy/Valid": val_accuracy})
            if scheduler is not None:
                wandb.log({"LearningRate/Train": scheduler.get_last_lr()[0]})
            else:
                wandb.log({"LearningRate/Train": CFG["LEARNING_RATE"]})
            
            print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

        if ema is not None:
            if ema_val_logloss < best_logloss:
                best_logloss = ema_val_logloss
                save_best_epoch(ema.ema_model, epoch, optimizer, model_params, ema_val_logloss, ema_save_data_params)
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    return
        else:
            if val_logloss < best_logloss:
                best_logloss = val_logloss
                save_best_epoch(model, epoch, optimizer, model_params, val_logloss, save_data_params)
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    return