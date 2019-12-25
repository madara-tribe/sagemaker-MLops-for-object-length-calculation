import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from enum import Enum
import pandas as pd
from lambda_func.CalculateHorizontalLength import *
from lambda_func.CalculateVerticalLength import *
from tools.is_right_protruding import *

def load_image(inputs_dir, mask = True):
    files = []
    files += glob.glob(os.path.join(inputs_dir, '*.png'))
    files += glob.glob(os.path.join(inputs_dir, '*.jpg'))
    files += glob.glob(os.path.join(inputs_dir, '*.jpeg'))
    files += glob.glob(os.path.join(inputs_dir, '*.PNG'))
    files += glob.glob(os.path.join(inputs_dir, '*.JPG'))
    files += glob.glob(os.path.join(inputs_dir, '*.JPEG'))
    print('target files : ', len(files))
    image, names = [], []
    for idx, f in enumerate(files):
        filename, _ = os.path.splitext(os.path.basename(f))
        if mask:
            img = cv2.imread(f, 0)
        else:
            img = cv2.imread(f)
        image.append(img)
        names.append(filename)
    return image, names

def save_df(lists):
    columns = ["filename", "AH", "BH", "DH", "CH", "length", ]
    dfs = pd.DataFrame(lists)
    dfs.columns = columns
    dfs.to_csv("train_length.csv", header=True, index=False)

imgs, names = load_image("tran_image_dir")
annos, _ = load_image("tran_anno_dir")

df=[]
for c, (img, anno, name) in enumerate(zip(imgs, annos, names)):
    direction = is_right_protruding(anno)
    if direction is True:
        img = cv2.flip(img, 1)
        anno = cv2.flip(anno, 1)

    H, W = np.shape(img)
    try:
        top_prop_width, top_prop_height = cal_TopProp_HW(img)
        if top_prop_height is True:
            AH = H-top_prop_height

        bottom_prop_width, bottom_prop_height = cal_BottomProp_HW(img)
        if bottom_prop_height is True:
            BH = H-bottom_prop_height

        next_prop_width, next_prop_height = cal_nextProp_HW(img, int(W/2))
        if next_prop_height is True:
            DH = H-next_prop_height

        cardboard_width, cardborad_height = cal_cardboard_HW(img)
        if cardborad_height is True:
            CH = H-cardborad_height

        A=(DH-AH)/(next_prop_width-top_prop_width)
        B = CH-(cardboard_width*A)
        y=(top_prop_width*A)+B

        AB = bottom_prop_height-top_prop_height
        AE = y-(H-top_prop_height)
        length = AE*PROP_HIEGHT/AB

        df.append([name, AH, BH, DH, CH, length])
        save_df(df)
    except:
        print("some point is None")
        df.append([name, np.nan, np.nan, np.nan, np.nan, np.nan])
        save_df(df)
