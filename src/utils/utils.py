import os
import random

import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )

def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )

def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,

    'EXPERIMENT_NAME' : None,
    'WRONG_DIR' : None
}