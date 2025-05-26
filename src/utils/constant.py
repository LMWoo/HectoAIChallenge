from enum import Enum

import torch.optim as optim

from src.model.resnet50 import Resnet50

from src.dataset.policy import (
    IdentityPolicy, TranslateXPolicy, TranslateYPolicy, RotatePolicy, ShearXPolicy, 
    ShearYPolicy, TranslateXPolicy, TranslateYPolicy, RotatePolicy, ColorPolicy,
    PosterizePolicy, SolarizePolicy, ContrastPolicy, SharpnessPolicy, BrightnessPolicy,
    AutoContrastPolicy, EqualizePolicy, InvertPolicy, TranslateXYPolicy, TranslateXYROTPolicy,


    RotateShearXPolicy, RotateShearYPolicy, RotateTranslateXPolicy, RotateTranslateYPolicy, 
    RotateColorPolicy, RotatePosterizePolicy, RotateSolarizePolicy, RotateContrastPolicy, 
    RotateSharpnessPolicy, RotateBrightnessPolicy, RotateAutoContrastPolicy, 
    RotateEqualizePolicy, RotateInvertPolicy,
)

from src.dataset.transform import (
    BaselineTransforms, PolicyTransforms, CropPolicyTransforms
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

class Transforms(CustomEnum):
    BASELINE_TRANSFORMS = BaselineTransforms
    POLICY_TRANSFORMS = PolicyTransforms
    CROP_POLICY_TRANSFORMS = CropPolicyTransforms

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



    ROTATE_SHEAR_X_POLICY = RotateShearXPolicy
    ROTATE_SHEAR_Y_POLICY = RotateShearYPolicy
    ROTATE_TRANSLATE_X_POLICY = RotateTranslateXPolicy
    ROTATE_TRANSLATE_Y_POLICY = RotateTranslateYPolicy
    ROTATE_COLOR_POLICY = RotateColorPolicy
    ROTATE_POSTERIZE_POLICY = RotatePosterizePolicy
    ROTATE_SOLARIZE_POLICY = RotateSolarizePolicy
    ROTATE_CONTRAST_POLICY = RotateContrastPolicy
    ROTATE_SHARPNESS_POLICY = RotateSharpnessPolicy
    ROTATE_BRIGHTNESS_POLICY = RotateBrightnessPolicy
    ROTATE_AUTO_CONTRAST_POLICY = RotateAutoContrastPolicy
    ROTATE_EQUALIZE_POLICY = RotateEqualizePolicy
    ROTATE_INVERT_POLICY = RotateInvertPolicy


    TRANSLATE_XY_POLICY = TranslateXYPolicy
    TRANSLATE_XY_ROT_POLICY = TranslateXYROTPolicy