from .datasets import *
from .datasets1 import *

LOADDATA = {
        'LOAD_ORIGINAL':LoadDataset_Vanilla,
        'LOAD_LABEL_UNLABEL': LoadDataset_Label_Unlabel,
        }
