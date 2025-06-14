import torchvision.transforms as transforms
from src.utils.utils import CFG

class CropPolicyColorJitterTransforms:
    def __init__(self, img_size, crop_size=128, augmentation_cls=None):
        self.img_size = img_size
        self.crop_size = crop_size
        self.augmentation_cls = augmentation_cls

    def get_train_transform(self):
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=(0.6, 1.0),
                contrast=(0.6, 1.0),
            ),
            transforms.RandomApply([
            transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 0.6))
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
    
class CropPolicyTransforms:
    def __init__(self, img_size, crop_size=128, augmentation_cls=None):
        self.img_size = img_size
        self.crop_size = crop_size
        self.augmentation_cls = augmentation_cls

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomApply([
            transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 0.6))
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

class RotateShearRandomCropsTransforms:
    def __init__(self, img_size, augmentation_cls=None):
        self.img_size = img_size
        self.augmentation_cls = augmentation_cls
        print('Using RotateShearRandomCropsTransforms')

    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size * 2, self.img_size * 2)),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15)
            ], p=0.5),

            transforms.RandomApply([
                transforms.RandomAffine(degrees=0, shear=(-10, 10))
            ], p=0.5),

            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomCrop(self.img_size / 2, padding=4),
                    transforms.RandomCrop(self.img_size, padding=4),
                    transforms.RandomCrop(self.img_size * 2, padding=4),
                ])
            ], p=0.5),
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