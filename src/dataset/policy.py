import random

import numpy as np

from src.dataset.augmentation import (
    ShearX, ShearY, TranslateX, TranslateY, Rotate, 
    Color, Posterize, Solarize, Contrast, Sharpness,
    Brightness, AutoContrast, Equalize, Invert
)

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int_),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }

        self.p1 = 0.5
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][5]

        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2] 
    
    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

class IdentityPolicy(object):
    def __init__(self, fillcolor=(128)):
        pass
    
    def __call__(self, img):
        return img
    
class TranslateXPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "translateX", 5, 0.0, "translateX", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class TranslateYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "translateY", 5, 0.0, "translateY", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotatePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    

class ShearXPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.5, "shearX", 8, 0.0, "shearX", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class ShearYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.5, "shearY", 8, 0.0, "shearY", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class TranslateXPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "translateX", 5, 0.0, "translateX", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class TranslateYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "translateY", 5, 0.0, "translateY", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotatePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class ColorPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.4, "color", 5, 0.0, "color", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
        
class PosterizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "posterize", 7, 0.0, "posterize", 7, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class SolarizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.4, "solarize", 5, 0.0, "solarize", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class ContrastPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.2, "contrast", 6, 0.0, "contrast", 6, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class SharpnessPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "sharpness", 9, 0.0, "sharpness", 9, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class BrightnessPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.6, "brightness", 7, 0.0, "brightness", 7, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class AutoContrastPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.5, "autocontrast", 8, 0.0, "autocontrast", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class EqualizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.6, "equalize", 5, 0.0, "equalize", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class InvertPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.1, "invert", 3, 0.0, "invert", 3, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)


#########################################

class RotateShearXPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.5, "shearX", 8, 0.0, "shearX", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateShearYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.0, "shearY", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateTranslateXPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.3, "translateX", 5, 0.0, "translateX", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateTranslateYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.3, "translateY", 5, 0.0, "translateY", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateColorPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.4, "color", 5, 0.0, "color", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
        
class RotatePosterizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.3, "posterize", 7, 0.0, "posterize", 7, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateSolarizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.0, "solarize", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateContrastPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.2, "contrast", 6, 0.0, "contrast", 6, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateSharpnessPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.0, "sharpness", 9, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateBrightnessPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.6, "brightness", 7, 0.0, "brightness", 7, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateAutoContrastPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.0, "autocontrast", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateEqualizePolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.0, "equalize", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateInvertPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.1, "invert", 3, 0.0, "invert", 3, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

#########################################

class ShearXShearYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.5, "shearX", 8, 0.0, "shearX", 8, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.0, "shearY", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class RotateShearXShearYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.0, "rotate", 2, fillcolor),
            SubPolicy(0.5, "shearX", 8, 0.0, "shearX", 8, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.0, "shearY", 8, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

#########################################

class TranslateXYPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.3, "translateX", 5, 0.3, "translateY", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

class TranslateXYROTPolicy(object):
    def __init__(self, fillcolor=(128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 5, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateY", 5, fillcolor),
        ]
    
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)