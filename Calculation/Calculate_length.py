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

def load_image(inputs_dir, ratio=4, mask=True):
   files, img, names = [], [], []
   files += glob.glob(os.path.join(inputs_dir, '*.png'))
   files += glob.glob(os.path.join(inputs_dir, '*.jpg'))
   files += glob.glob(os.path.join(inputs_dir, '*.jpeg'))
   files += glob.glob(os.path.join(inputs_dir, '*.PNG'))
   files += glob.glob(os.path.join(inputs_dir, '*.JPG'))
   files += glob.glob(os.path.join(inputs_dir, '*.JPEG'))
   print('target files : ', len(files))
   for f in files):
       imgs = cv2.imread(f, 0)
       filename, _ = os.path.splitext(os.path.basename(f))
       if imgs is not None:
           if mask:
               h, w = np.shape(imgs)
           else:
               h, w, _ = np.shape(imgs)
           imgs = cv2.resize(imgs, (int(w/ratio), int(h/ratio)), interpolation=cv2.INTER_NEAREST)
           img.append(imgs)
           names.append(filename)
   print("load finish")
   return img, names

def save_DataFrame(df):
    columns = ["filename", "AH", "BH", "DH", "CH", "length", ]
    dfs=pd.DataFrame(df)
    dfs.columns = columns
    dfs.to_csv("v_length/train_length.csv", header=True, index=False)

def calculate_length(imgs, name_list):
    df=[]
    for c, (img, name) in enumerate(zip(imgs, name_list)):
        print(name)
        if 'R.' in name:
            img = cv2.flip(img, 1)

        H, W = np.shape(img)
        try:
            top_prop_width, top_prop_height = cal_TopProp_HW(img)
            if top_prop_height is not None:
                AH = H-top_prop_height
                print("yes AH")

            bottom_prop_width, bottom_prop_height = cal_BottomProp_HW(img)
            if bottom_prop_height is not None:
                BH = H-bottom_prop_height
                print("yes BH")

            next_prop_width, next_prop_height = cal_nextProp_HW(img, int(W/2))
            if next_prop_height is not None:
                DH = H-next_prop_height
                print("yes DH")

            cardboard_width, cardborad_height = cal_cardboard_HW(img)
            if cardborad_height is not None:
                CH = H-cardborad_height
                print("yes CH")


            A=(DH-AH)/(next_prop_width-top_prop_width)
            B = CH-(cardboard_width*A)
            y=(top_prop_width*A)+B

            AB = bottom_prop_height-top_prop_height
            AE = y-(H-top_prop_height)
            length = AE*PROP_HIEGHT/AB
            print(length)
            print("yes length")
            df.append([name, "AH", "BH", "DH", "CH", length])

        except:
            print("some point is None")
            df.append([name, np.nan, np.nan, np.nan, np.nan, np.nan])
    save_DataFrame(df)

f __name__ == '__main__':
    argvs = sys.argv
    image_dir_path = argvs[1]
    images, names = load_image(image_dir_path, mask=None)
    calculate_length(images, names)
