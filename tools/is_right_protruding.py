import numpy as np
import os
import cv2
from enum import IntEnum

class ClassID(IntEnum):
    ID_CARDBOAD= 2
    ID_PROP= 1
    ID_PEDESTAL= 3
    ID_SIDEBOARD = 4

def is_right_protruding(mask):
    h, w = np.shape(mask)[:2]
    mask_l = mask[:, :int(w/5*3)]
    mask_r = mask[:, int(w/5*3):]
    is_right = is_protruding(mask_r)
    return is_right

def is_protruding(mask):
    is_include_CB = np.any(mask == ClassID.ID_CARDBOAD)
    if not is_include_CB:
        return False
    hs, ws = np.shape(mask)[:2]
    for h in range(hs):
        mask_oneline = mask[h]
        is_include_CB = np.any(mask_oneline == ClassID.ID_CARDBOAD)
        is_include_SB = np.any(mask_oneline == ClassID.ID_SIDEBOARD)
        if is_include_CB and is_include_SB:
            return True
    return False
