import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import random
import shutil
from functools import partial

import wandb
from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import fire
from sklearn.model_selection import StratifiedKFold

from src.utils.utils import seed_everything, project_path, auto_increment_run_suffix, CFG
from src.utils.constant import Optimizers, Models, Losses, Augmentations, Transforms, Datasets
from src.dataset.baselineDataset import get_datasets, get_datasets_kfold
from src.model.resnet50 import Resnet50
from src.train.train import train, train_fold
from src.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from src.postprocess.postprocess import write_db

def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")

def get_latest_run(project_name):
    # runs = get_runs(project_name)
    # if not runs:
    #     return f"{CFG['EXPERIMENT_NAME'].replace('_', '-')}-000"
    
    # return runs[0].name
    runs = get_runs(project_name)
    
    filtered = [
        run for run in runs
        if run.config.get("experiment_name") == CFG["EXPERIMENT_NAME"]
    ]
    
    if not filtered:
        default_name = f"{CFG['EXPERIMENT_NAME'].replace('_', '-')}-000"
        return default_name
    
    return filtered[0].name
    

def run_train_kfold(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    api_key  =os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = "classification" #  + CFG['EXPERIMENT_NAME'].replace("_", "-")
    try:
        run_name = get_latest_run(project_name)
    except Exception as e:
        print(f"[W&B WARNING] Failed to get previous runs: {e}")
        run_name = f"{CFG['EXPERIMENT_NAME'].replace('_', '-')}-000"
    
    next_run_name = auto_increment_run_suffix(run_name)

    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)
    Losses.validation(loss_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    full_dataset, targets, test_dataset, class_names, train_transform, val_transform = get_datasets_kfold(augmentation_cls, transform_cls, dataset_cls)

    kf = StratifiedKFold(n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG['SEED'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)), targets)):
        print(fold)
        if fold == 0:
            continue
        if "fold_" in next_run_name:
            next_run_name = next_run_name[7:]
        next_run_name = f"fold_{fold}-" + next_run_name
        
        wandb.init(
            project=project_name,
            id=f"{next_run_name}",
            name=f"{next_run_name}",
            group=f"{next_run_name}",
            notes="content-based classification model",
            tags=["content-based", "classification"],
            config={
                "experiment_name": CFG['EXPERIMENT_NAME'],
                "model_name": model_name,
                "optimizer_name": optimizer_name,
                "augmentation_name": augmentation_name,
                "transforms_name": transforms_name,
                "datasets_name": datasets_name,
                "loss_name": loss_name,
                "freeze_epochs": freeze_epochs,
                "device": str(device),
            }
        )
                
        train_root = os.path.join(project_path(), 'data/train')
        train_dataset = Subset(dataset_cls(train_root, transform=train_transform), train_idx)
        val_dataset = Subset(dataset_cls(train_root, transform=val_transform), val_idx)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

        model_class = Models[model_name.upper()].value
        
        model_params = {
            "num_classes": len(class_names),
        }

        model = model_class(**model_params).to(device)

        try:
            samples = train_loader.dataset.dataset.samples
        except:
            samples = train_loader.dataset.samples
        labels = [label for _, label in samples]
        cls_counts = np.bincount(labels)
        total_count = sum(cls_counts)

        loss_params = { 
            "alpha": torch.tensor(total_count / (len(cls_counts) * cls_counts)).to(device), 
            "gamma" : 2.0, 
            "reduction": "mean" 
        }

        loss_class = Losses[loss_name.upper()].value
        if loss_name.upper() == "FOCAL_LOSS":
            criterion = loss_class(**loss_params).to(device)
        else:
            criterion = loss_class().to(device)

        optimizer_class = Optimizers[optimizer_name.upper()].value
        if optimizer_name.upper() == "ADAMW":
            print('adamw + cosine schedule')
            optimizer = optimizer_class(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=0.05)

            total_steps = len(train_loader) * CFG['EPOCHS']
            warmup_steps = len(train_loader) * 3
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            optimizer = optimizer_class(model.parameters(), lr=CFG['LEARNING_RATE'])
            scheduler = None

        train_fold(fold, model, train_loader, val_loader, model_params, criterion, optimizer, scheduler, freeze_epochs, device)

        del model
        del optimizer
        del scheduler
        del criterion
        torch.cuda.empty_cache()
        wandb.finish()



def run_train(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    api_key  =os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = "classification" #  + CFG['EXPERIMENT_NAME'].replace("_", "-")
    try:
        run_name = get_latest_run(project_name)
    except Exception as e:
        print(f"[W&B WARNING] Failed to get previous runs: {e}")
        run_name = f"{CFG['EXPERIMENT_NAME'].replace('_', '-')}-000"
    
    next_run_name = auto_increment_run_suffix(run_name)
    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based classification model",
        tags=["content-based", "classification"],
        config={
            "experiment_name": CFG['EXPERIMENT_NAME'],
            "model_name": model_name,
            "optimizer_name": optimizer_name,
            "augmentation_name": augmentation_name,
            "transforms_name": transforms_name,
            "datasets_name": datasets_name,
            "loss_name": loss_name,
            "freeze_epochs": freeze_epochs,
            "device": str(device),
        }
    )

    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)
    Losses.validation(loss_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(augmentation_cls, transform_cls, dataset_cls)

    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

    model_class = Models[model_name.upper()].value
    
    model_params = {
        "num_classes": len(class_names),
    }

    model = model_class(**model_params).to(device)

    try:
        samples = train_loader.dataset.dataset.samples
    except:
        samples = train_loader.dataset.samples
    labels = [label for _, label in samples]
    cls_counts = np.bincount(labels)
    total_count = sum(cls_counts)

    loss_params = { 
        "alpha": torch.tensor(total_count / (len(cls_counts) * cls_counts)).to(device), 
        "gamma" : 2.0, 
        "reduction": "mean" 
    }

    loss_class = Losses[loss_name.upper()].value
    if loss_name.upper() == "FOCAL_LOSS":
        criterion = loss_class(**loss_params).to(device)
    else:
        criterion = loss_class().to(device)

    optimizer_class = Optimizers[optimizer_name.upper()].value
    if optimizer_name.upper() == "ADAMW":
        print('adamw + cosine schedule')
        optimizer = optimizer_class(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=0.05)

        total_steps = len(train_loader) * CFG['EPOCHS']
        warmup_steps = len(train_loader) * 3
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        optimizer = optimizer_class(model.parameters(), lr=CFG['LEARNING_RATE'])
        scheduler = None

    train(model, train_loader, val_loader, model_params, criterion, optimizer, scheduler, freeze_epochs, device)

    wandb.finish()
    
def run_test_tta(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)
    Losses.validation(loss_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(augmentation_cls, transform_cls, dataset_cls)

    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

    # model_class = Models[model_name.upper()].value

    checkpoint = load_checkpoint()

    model = init_model(checkpoint, model_name)

    # model = model_class(num_classes=len(class_names))
    # model.load_state_dict(torch.load(f"best_model_{CFG['EXPERIMENT_NAME']}.pth", map_location=device))
    model.to(device)
    model.eval()
    results = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # TTA: original
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # TTA: horizontal flip
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = F.softmax(outputs_flip, dim=1)

            # Average
            probs_avg = (probs_orig + probs_flip) * 0.5

            # Collect
            for prob in probs_avg.cpu():
                results.append({class_names[i]: prob[i].item() for i in range(len(class_names))})
    
    pred = pd.DataFrame(results)

    submission = pd.read_csv(os.path.join(project_path(), 'data/sample_submission.csv'), encoding='utf-8-sig')

    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(os.path.join(project_path(), f"data/{CFG['EXPERIMENT_NAME']}_submission_tta.csv") , index=False, encoding='utf-8-sig')

def run_test_kfold_tta(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(augmentation_cls, transform_cls, dataset_cls)

    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

    # model_class = Models[model_name.upper()].value

    all_probs = []

    for fold in range(CFG['N_FOLDS']):
        checkpoint = load_checkpoint(fold)

        model = init_model(checkpoint, model_name)

        # model = model_class(num_classes=len(class_names))
        # model.load_state_dict(torch.load(f"best_model_{CFG['EXPERIMENT_NAME']}.pth", map_location=device))
        model.to(device)
        model.eval()
        fold_probs = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # TTA: original
                outputs_orig = model(images)
                probs_orig = F.softmax(outputs_orig, dim=1)

                # TTA: horizontal flip
                images_flip = torch.flip(images, dims=[3])
                outputs_flip = model(images_flip)
                probs_flip = F.softmax(outputs_flip, dim=1)

                # Average
                probs_avg = (probs_orig + probs_flip) * 0.5

                fold_probs.append(probs_avg.cpu().numpy())
        
        all_probs.append(np.concatenate(fold_probs, axis=0))

    avg_probs = np.mean(all_probs, axis=0)

    results = [
        {class_names[i]: prob[i] for i in range(len(class_names))}
        for prob in avg_probs
    ]
    
    pred = pd.DataFrame(results)

    submission = pd.read_csv(os.path.join(project_path(), 'data/sample_submission.csv'), encoding='utf-8-sig')

    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(os.path.join(project_path(), f"data/{CFG['EXPERIMENT_NAME']}_submission_tta.csv") , index=False, encoding='utf-8-sig')

def run_test_kfold(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(augmentation_cls, transform_cls, dataset_cls)

    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

    # model_class = Models[model_name.upper()].value

    all_probs = []

    for fold in range(CFG['N_FOLDS']):
        checkpoint = load_checkpoint(fold)

        model = init_model(checkpoint, model_name)

        # model = model_class(num_classes=len(class_names))
        # model.load_state_dict(torch.load(f"best_model_{CFG['EXPERIMENT_NAME']}.pth", map_location=device))
        model.to(device)
        model.eval()
        fold_probs = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())
        
        all_probs.append(np.concatenate(fold_probs, axis=0))

    avg_probs = np.mean(all_probs, axis=0)

    results = [
        {class_names[i]: prob[i] for i in range(len(class_names))}
        for prob in avg_probs
    ]
    
    pred = pd.DataFrame(results)

    submission = pd.read_csv(os.path.join(project_path(), 'data/sample_submission.csv'), encoding='utf-8-sig')

    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(os.path.join(project_path(), f"data/{CFG['EXPERIMENT_NAME']}_submission.csv") , index=False, encoding='utf-8-sig')


def run_test(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device):
    Models.validation(model_name)
    Optimizers.validation(optimizer_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)
    Datasets.validation(datasets_name)

    augmentation_cls = Augmentations[augmentation_name.upper()].value
    transform_cls = Transforms[transforms_name.upper()].value
    dataset_cls = Datasets[datasets_name.upper()].value

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(augmentation_cls, transform_cls, dataset_cls)

    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=8, shuffle=False)

    # model_class = Models[model_name.upper()].value

    checkpoint = load_checkpoint()

    model = init_model(checkpoint, model_name)

    # model = model_class(num_classes=len(class_names))
    # model.load_state_dict(torch.load(f"best_model_{CFG['EXPERIMENT_NAME']}.pth", map_location=device))
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

def run_inference(model_name, loss_name, augmentation_name, transforms_name, device, batch_size=64):
    Models.validation(model_name)
    Augmentations.validation(augmentation_name)
    Transforms.validation(transforms_name)

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(Augmentations[augmentation_name.upper()].value, Transforms[transforms_name.upper()].value)

    checkpoint = load_checkpoint()

    model = init_model(checkpoint, model_name)
    
    image = torch.randn((1, 3, 224, 224))
    
    result = inference(model, image, device, augmentation_name)
    print(result, class_names.index(result))
    
    recommend_df = recommend_to_df(class_names.index(result))
    write_db(recommend_df, "mlops", "recommend")

def main(run_mode, experiment_name, model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs):
    CFG['EXPERIMENT_NAME'] = experiment_name
    CFG['WRONG_DIR'] = os.path.join('./validation_wrong_dir', CFG['EXPERIMENT_NAME'])
    os.makedirs(CFG['WRONG_DIR'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device : ", device)
        
    seed_everything(CFG['SEED'])
    if run_mode == "train":
        run_train(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
    elif run_mode == "train_kfold":
        run_train_kfold(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
    elif run_mode == "test_kfold":
        run_test_kfold(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
        run_test_kfold_tta(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
    elif run_mode == "test":
        run_test(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
        run_test_tta(model_name, loss_name, optimizer_name, augmentation_name, transforms_name, datasets_name, freeze_epochs, device)
    elif run_mode == "inference":
        run_inference(model_name, loss_name, augmentation_name, transforms_name, device)

if __name__ == '__main__':
    fire.Fire(main)