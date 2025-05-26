import torchvision.transforms as transforms
from src.utils.utils import CFG

class CropPolicyTransforms:
    def __init__(self, img_size, crop_size=256, augmentation_cls=None):
        self.img_size = img_size
        self.crop_size = crop_size
        self.augmentation_cls = augmentation_cls

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomApply([
            transforms.RandomCrop(self.crop_size, padding=4)
            ], p=0.5),
            transforms.Resize((self.img_size, self.img_size)),
            self.augmentation_cls() if self.augmentation_cls else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return self.get_train_transform(), self.get_val_transform()
    
class PolicyTransforms:
    def __init__(self, img_size, augmentation_cls=None):
        self.img_size = img_size
        self.augmentation_cls = augmentation_cls

    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            self.augmentation_cls() if self.augmentation_cls else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return self.get_train_transform(), self.get_val_transform()

class BaselineTransforms:
    def __init__(self, img_size, augmentation_cls=None):
        self.img_size = img_size
        self.augmentation_cls = augmentation_cls

    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return self.get_train_transform(), self.get_val_transform()