from enum import Enum

import torch.optim as optim

from src.model.resnet50 import Resnet50

from src.dataset.policy import (
    IdentityPolicy, TranslateXPolicy, TranslateYPolicy, RotatePolicy, ShearXPolicy, 
    ShearYPolicy, TranslateXPolicy, TranslateYPolicy, RotatePolicy, ColorPolicy,
    PosterizePolicy, SolarizePolicy, ContrastPolicy, SharpnessPolicy, BrightnessPolicy,
    AutoContrastPolicy, EqualizePolicy, InvertPolicy, TranslateXYPolicy, TranslateXYROTPolicy
)

class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]
    
    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class Models(CustomEnum):
    RESNET_50 = Resnet50

class Optimizers(CustomEnum):
    ADAM = optim.Adam
    RADAM = optim.RAdam
    NADAM = optim.NAdam
    SPARSEADAM = optim.SparseAdam
    SGD = optim.SGD
    RMSPROP = optim.RMSprop

class Augmentations(CustomEnum):
    IDENTITY_POLICY = IdentityPolicy
    TRANSLATE_X_POLICY = TranslateXPolicy
    TRANSLATE_Y_POLICY = TranslateYPolicy
    ROTATE_POLICY = RotatePolicy
    SHEAR_X_POLICY = ShearXPolicy
    SHEAR_Y_POLICY = ShearYPolicy
    COLOR_POLICY = ColorPolicy
    POSTERIZE_POLICY = PosterizePolicy
    SOLARIZE_POLICY = SolarizePolicy
    CONTRAST_POLICY = ContrastPolicy
    SHARPNESS_POLICY = SharpnessPolicy
    BRIGHTNESS_POLICY = BrightnessPolicy
    AUTO_CONTRAST_POLICY = AutoContrastPolicy
    EQUALIZE_POLICY = EqualizePolicy
    INVERT_POLICY = InvertPolicy

    TRANSLATE_XY_POLICY = TranslateXYPolicy
    TRANSLATE_XY_ROT_POLICY = TranslateXYROTPolicy