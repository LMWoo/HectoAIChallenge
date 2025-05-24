import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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


from src.utils.utils import seed_everything, project_path, CFG
from src.dataset.HectoDataset import get_datasets
from src.model.resnet50 import Resnet50
from src.train.train import train

CFG['EXPERIMENT_NAME'] = 'resnet50_aug_xy_rot'
CFG['WRONG_DIR'] = os.path.join('./validation_wrong_dir', CFG['EXPERIMENT_NAME'])
os.makedirs(CFG['WRONG_DIR'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device : ", device)
    
seed_everything(CFG['SEED'])    

train_dataset, val_dataset, test_dataset, class_names = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model = Resnet50(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

train(model, train_loader, val_loader, criterion, optimizer, device, class_names)

# test_dataset = CustomImageDataset(test_root, transform=val_transform, is_teat=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model = Resnet50(num_classes=len(class_names))
model.load_state_dict(torch.load(f"best_model_{CFG['EXPERIMENT_NAME']}.pth", map_location=device))
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

submission = pd.read_csv(os.path.join(project_path(), 'data/sample_submission.csv'), encoding='utf-8-sig')

class_columns = submission.columns[1:]
pred = pred[class_columns]

submission[class_columns] = pred.values
submission.to_csv(os.path.join(project_path(), f"data/{CFG['EXPERIMENT_NAME']}_submission.csv") , index=False, encoding='utf-8-sig')