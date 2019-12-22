import sys
import numpy as np
import os
import cv2
from enum import IntEnum


PROPS_POS_WIDTH = 133

class ClassID(IntEnum):
    ID_CARDBOAD= 2
    ID_PROP= 1
    ID_PEDESTAL= 3
    ID_SIDEBOARD = 4


# A
def cal_cardboad_HW(img):
    cardboad_height = None
    cardboad_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for w in range(mask_w):
        for h in range(mask_h):
            class_id = img[h, w]
            if class_id == ClassID.ID_CARDBOAD:
                cardboad_height = h
                cardboad_width = w
                break
        if cardboad_width is not None:
            break
    return cardboad_width, cardboad_height

# B
def cal_cardboad_lastHW(img, cardboad_height):
    cardboad_lastheight = None
    cardboad_lastwidth = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for w in range(int((mask_w/3)*2)):
        for h in range(mask_h):
            if h==cardboad_height:
                class_id = img[h, w]
                if class_id == ClassID.ID_CARDBOAD:
                    cardboad_lastheight = h
                    cardboad_lastwidth = w
    return cardboad_lastwidth, cardboad_lastheight



# C

def cal_prop_HW(img, cardboad_height):
    prop_height = None
    prop_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for w in range(int((mask_w/3)*2)):
        for h in range(mask_h):
            if h==cardboad_height:
                class_id = img[h, w]
                if class_id == ClassID.ID_PROP or class_id == ClassID.ID_PEDESTAL:
                    prop_height = h
                    prop_width = w
    return prop_width, prop_height

# D

def cal_next_prop_HW(img, cardboad_height, half_width):
    prop_height = None
    prop_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for w in range(mask_w):
        if w>half_width:
            for h in range(mask_h):
                if h == cardboad_height:
                    class_id = img[h, w]
                    #if h<half_width:
                    if class_id == ClassID.ID_PROP or class_id == ClassID.ID_PEDESTAL:
                        prop_height = h
                        prop_width = w
                        break
            if prop_width is not None:
                break
    return prop_width, prop_height

def horizontal_length(img, W):
    b=0
    try:
        b = horizontal_length_(img, W)
    except:
        pass
    return b

def horizontal_length_(img, W):
    cardboard_left_width, cardboard_left_height = cal_cardboad_HW(img)
    cardboad_right_width, cardboad_right_height = cal_cardboad_lastHW(img, cardboard_left_height)
    prop_width, prop_height = cal_prop_HW(img, cardboard_left_height)
    next_prop_width, next_prop_height = cal_next_prop_HW(img, cardboard_left_height, int(W/2))
    R=(cardboard_left_width-cardboad_right_width)*PROPS_POS_WIDTH/(prop_width-next_prop_width)
    print("horizontal length {} cm".format(R))



if __name__ == '__main__':
    argvs = sys.argv
    image_name = argvs[1]
    img = cv2.imread(image_name, 0)
    H, W = np.shape(img)
    horizontal_length(img, W)

