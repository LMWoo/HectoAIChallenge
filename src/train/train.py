import os
import shutil
import datetime

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from sklearn.metrics import log_loss

from src.utils.utils import CFG, model_dir


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

def validation_one_epoch(model, val_loader, class_names, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
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

            wrong_mask = (preds != labels).cpu()
            for idx, is_wrong in enumerate(wrong_mask):
                if is_wrong:
                    wrong_imgs.append(img_paths[idx])
        
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

    return avg_val_loss, val_accuracy, val_logloss, wrong_imgs
        
def save_best_epoch(model, epoch, optimizer, val_logloss, wrong_imgs):
    save_dir = model_dir(CFG['EXPERIMENT_NAME'])
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.pth")
    torch.save({
        "epoch": epoch,
        # "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_logloss": val_logloss,
    }, dst)
    print(f"Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

    best_epoch_wrong_dir = os.path.join(CFG['WRONG_DIR'], 'best_model')

    if os.path.exists(best_epoch_wrong_dir):
        shutil.rmtree(best_epoch_wrong_dir)
    os.makedirs(best_epoch_wrong_dir, exist_ok=True)
    for wrong_img in wrong_imgs:
        shutil.copy(wrong_img, os.path.join(best_epoch_wrong_dir, os.path.basename(wrong_img)))

def train(model, train_loader, val_loader, class_names, criterion, optimizer, device):
    best_logloss = float('inf')

    for epoch in range(CFG['EPOCHS']):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        avg_val_loss, val_accuracy, val_logloss, wrong_imgs = validation_one_epoch(model, val_loader, class_names, criterion, device, epoch)
        wandb.log({"Loss/Train": avg_train_loss})
        wandb.log({"Loss/Valid": avg_val_loss})
        wandb.log({"LogLoss/Valid": val_logloss})
        wandb.log({"Accuracy/Valid": val_accuracy})
        
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            save_best_epoch(model, epoch, optimizer, val_logloss, wrong_imgs)