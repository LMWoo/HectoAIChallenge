import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from src.utils.utils import seed_everything, CFG

experiment_name = 'resnet50_aug_xy_rot'

wrong_dir = os.path.join('./validation_wrong_dir', experiment_name)
os.makedirs(wrong_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device : ", device)



    
seed_everything(CFG['SEED'])    


train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
model = BaseModel(num_classes=len(class_names)).to(device)
best_logloss = float('inf')

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

for epoch in range(CFG['EPOCHS']):
    
    model.train()
    train_loss = 0.0
    for images, labels, img_paths in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
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
    
    print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

    if val_logloss < best_logloss:
        best_logloss = val_logloss
        torch.save(model.state_dict(), f'best_model_{experiment_name}.pth')
        print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

        best_epoch_wrong_dir = os.path.join(wrong_dir, 'best_model')
        
        if os.path.exists(best_epoch_wrong_dir):
            shutil.rmtree(best_epoch_wrong_dir)
        os.makedirs(best_epoch_wrong_dir, exist_ok=True)
        for wrong_img in wrong_imgs:
            shutil.copy(wrong_img, os.path.join(best_epoch_wrong_dir, os.path.basename(wrong_img)))

test_dataset = CustomImageDataset(test_root, transform=val_transform, is_teat=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model = BaseModel(num_classes=len(class_names))
model.load_state_dict(torch.load(f'best_model_{experiment_name}.pth', map_location=device))
model.to(device)

model.eval()
results = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        
        for prob in probs.cpu():
            result = {
                class_names[i]: prob[i].item()
                for i in range(len(class_names))
            }
            results.append(result)

pred = pd.DataFrame(results)

submission = pd.read_csv('../data/sample_submission.csv', encoding='utf-8-sig')

class_columns = submission.columns[1:]
pred = pred[class_columns]

submission[class_columns] = pred.values
submission.to_csv(f'../data/{experiment_name}_submission.csv', index=False, encoding='utf-8-sig')