import sys
import numpy as np
import os
import cv2
from enum import Enum


ID_PROP= 1 
ID_CARDBOAD= 2
ID_PEDESTAL= 3
PROP_HIEGHT = 130

# A

def cal_TopProp_HW(img):
    Prop_height = None
    Prop_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(int(mask_w*2/3)):
            #dist = cv2.circle(copy_imgs, (w, h), 20, (255, 255, 255), thickness=-1)
            #plt.imshow(dist),plt.show()
            class_id = img[h, w]
            if class_id == ID_PROP:
                Prop_height = h
                Prop_width = w
                break
        if Prop_height is not None:
            break
    #dist = cv2.drawMarker(copy_imgs,(Prop_Width, Prop_Height),(255,255))
    #plt.imshow(dist),plt.show()
    return Prop_width, Prop_height


# B
def cal_BottomProp_HW(img):
    Prop_last_height = None
    Prop_last_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(int(mask_w*2/3)):
            #dist = cv2.circle(copy_imgs, (w, h), 20, (255, 255, 255), thickness=-1)
            #plt.imshow(dist),plt.show()
            class_id = img[-h, w]
            if class_id == ID_PEDESTAL:
                break
            if class_id == ID_PROP:
                Prop_last_height = mask_h-h
                Prop_last_width = w
                break
        if Prop_last_height is not None:
            break
    #dist = cv2.drawMarker(copy_imgs,(Prop_Width, Prop_Height),(255,255))
    #plt.imshow(dist),plt.show()
    return Prop_last_width, Prop_last_height


# D

def cal_nextProp_HW(img, half_width):
    Prop_height = None
    Prop_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(mask_w):
            #dist = cv2.circle(copy_imgs, (w, h), 20, (255, 255, 255), thickness=-1)
            #plt.imshow(dist),plt.show()
            class_id = img[h, w]
            if w>half_width:
                if class_id == ID_PROP:
                    Prop_height = h
                    Prop_width = w
                    break
        if Prop_height is not None:
            break
    #dist = cv2.drawMarker(copy_imgs,(Prop_Width, Prop_Height),(255,255))
    #plt.imshow(dist),plt.show()
    return Prop_width, Prop_height

# C

def cal_cardboard_HW(img):
    Cardboard_height = None
    Cardboard_width = None
    copy_imgs = img.copy()
    mask_h, mask_w = np.shape(img)
    masked = np.zeros([mask_h, mask_w], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(mask_w):
            #dist = cv2.circle(copy_imgs, (w, h), 500, (255, 255, 255), thickness=-1)
            #plt.imshow(dist),plt.show()
            class_id = img[h, w]
            if class_id == ID_CARDBOAD:
                Cardboard_height = h
                Cardboard_width = w
                break
        if Cardboard_width is not None:
            break
    #dist = cv2.drawMarker(copy_imgs,(Prop_Width, Prop_Height),(255,255))
    #plt.imshow(dist),plt.show()
    return Cardboard_width, Cardboard_height

def cal_vertical_lenth(img, W):
  a=0
  try:
      a = cal_vertical_lenth_(img, W)
  except:
      pass
  return a
      
def cal_vertical_lenth_(img, W):
    # A
    top_prop_width, top_prop_height = cal_TopProp_HW(imgs)
    AH = H-top_prop_height
    # B
    bottom_prop_width, bottom_prop_height = cal_BottomProp_HW(imgs)
    BH = H-bottom_prop_height
    # D
    next_prop_width, next_prop_height = cal_nextProp_HW(imgs, int(W/2))
    DH = H-next_prop_height
    # C
    cardboard_width, cardborad_height = cal_cardboard_HW(imgs)
    CH = H-cardborad_height
    
    A=(DH-AH)/(next_prop_width-top_prop_width)
    B = CH-(cardboard_width*A)
    y=(top_prop_width*A)+B
    
    AB = bottom_prop_height-top_prop_height
    AE = y-(H-top_prop_height)
    print("vertical length is {} cm ".format(AE*PROP_HIEGHT/AB))

    
if __name__ == '__main__':
    argvs = sys.argv
    image_name = argvs[1]
    imgs = cv2.imread(image_name, 0)
    imgs = cv2.flip(imgs, 1)
    H, W = np.shape(imgs)
    cal_vertical_lenth(imgs, W)

