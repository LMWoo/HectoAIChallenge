import os
import random
import hashlib

import numpy as np
import torch

def calculate_hash(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_hash(dst):
    hash_ = calculate_hash(dst)
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "w") as f:
        f.write(hash_)

def read_hash(dst):
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "r") as f:
        return f.read()


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
    'EPOCHS': 100,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,

    'EXPERIMENT_NAME' : None,
    'WRONG_DIR' : None
}