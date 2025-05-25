import os
import sys
import glob

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.resnet50 import Resnet50
from src.utils.utils import model_dir, calculate_hash, read_hash
from src.utils.utils import CFG
from src.utils.constant import Augmentations
from src.dataset.baselineDataset import get_datasets

from dotenv import load_dotenv
from src.postprocess.postprocess import write_db
from src.utils.constant import Optimizers, Models, Augmentations

def recommend_to_df(recommend):
    return pd.DataFrame(
        data=[recommend],
        columns="reommend_content_id".split()
    )

def init_model(checkpoint, model_name):
    model_class = Models[model_name.upper()].value
    model = model_class(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    return model, criterion

def model_validation(model_path):
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        print('validation sucess')
        return True
    else:
        return False

def load_checkpoint():
    target_dir = model_dir(CFG["EXPERIMENT_NAME"])
    model_path = os.path.join(target_dir, "best_model.pth")

    if model_validation(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        return checkpoint
    else:
        raise FileExistsError("Not found or invalid model file")

def inference(model, image, criterion, device, augmentation_name):
    Augmentations.validation(augmentation_name)

    _, _, _, class_names = get_datasets(Augmentations[augmentation_name.upper()].value)
    model.eval()
    model.to(device)

    results = []
    image = image.to(device)
    output = model(image)
    prob = F.softmax(output, dim=1)
    pred_idx = prob.argmax(dim=1).item()
    return class_names[pred_idx]
