import torchvision.transforms as transforms
from src.utils.utils import CFG

def get_transforms(augmentation_cls):
    train_transform = transforms.Compose([
        transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        augmentation_cls(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform