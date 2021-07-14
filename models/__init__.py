from .wideresnet import *
from .wideresnet_lk import *
from .simple_color_cnn import SimpleColorCNN

WRN_MODELS = {
        'WideResNet':WideResNet,
        'WideResNet_Lk': WideResNet_Lk,
        'SimpleColorCNN': SimpleColorCNN
        }
