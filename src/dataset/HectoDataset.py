import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

from src.utils.utils import project_path, CFG
from src.dataset.preprocessing import get_transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []
    
        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, label, img_path

# def read_dataset():
#     train_root = os.path.join(project_path(), 'train')
#     test_root = os.path.join(project_path(), 'test')
    
#     full_dataset = CustomImageDataset(train_root, transform=None)
#     print(f'Total image num: {len(full_dataset)}')
    
#     targets = [label for _, label in full_dataset.samples]
#     class_names = full_dataset.classes
    
#     return train_dataset, test_dataset
    
# def split_dataset():
#     train_dataset, test_dataset = read_dataset()
    
#     train_idx, val_idx = train_test_split(
#         range(len(targets)), test_size=0.2, stratify=targets, random_state=42
#     )

def get_datasets(augmentation_cls):
    train_root = os.path.join(project_path(), 'data/train')
    test_root = os.path.join(project_path(), 'data/test')

    full_dataset = CustomImageDataset(train_root, transform=None)
    print(f'Total image num: {len(full_dataset)}')

    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes

    # Stratified Split
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=CFG['SEED']
    )

    train_transform, val_transform = get_transforms(augmentation_cls)
    # Subset + transform
    train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
    val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)

    print(f'train image num: {len(train_dataset)}, valid image num: {len(val_dataset)}')

    test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
    
    return train_dataset, val_dataset, test_dataset, class_names

    
    