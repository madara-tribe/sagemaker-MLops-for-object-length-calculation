import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from enum import IntEnum
from lambda_func.CalculateHorizontalLength import *
from lambda_func.CalculateVerticalLength import *
from tools.is_right_protruding import *

def mask_to_indexmap(mask):
    mask_h, mask_w = np.shape(mask)
    masked = np.zeros([mask_h, mask_w, 3], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(mask_w):
            class_id = mask[h, w]
            #print(idx, np.unique(class_id))
            r, b, g = (0, 0, 0)
            if class_id == 1:
                r, g, b = (0,100,0)
            elif class_id == 2:
                r, g, b = (65,105,225)
            elif class_id == 3:
                r, g, b = (255,140,0)
            elif class_id == 4:
                r, g, b = (220,20,60)
            elif class_id == 5:
                r, g, b = (186,85,211)
            elif class_id == 6:
                r, g, b = (139, 69, 19)
            else:
                r, g, b = (0, 0, 0) # white

            masked[h, w, 0] = r
            masked[h, w, 1] = g
            masked[h, w, 2] = b

    return masked


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




def save_img(img, name):
    print("yes")
    save_dir="draw_img/validation"
    img_ = img.astype(np.uint8)
    bgr = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, str(name)+".jpg"), bgr)

def draw_poinline(annos, images, name_list):
    point_size = 6
    font_size = 3
    line_size = point_size-2
    plus_text_range = 20
    for c, (anno, img, names) in enumerate(zip(annos, images, name_list)):
        name, _ = os.path.splitext(names)
        direction = is_right_protruding(anno)
        if direction in True:
            img = cv2.flip(img, 1)
            anno = cv2.flip(anno, 1)

        point_img = img.copy()
        H, W = np.shape(anno)
        color_mask = mask_to_indexmap(anno)
        point_img = cv2.addWeighted(point_img, 1.0, color_mask, 0.6, 1.0)

        print(c, name)
        cardboard_left_width, cardboard_left_height = cal_cardboad_HW(anno)
        if cardboard_left_width is not None and cardboard_left_height is not None:
            point_img = cv2.circle(point_img, (cardboard_left_width, cardboard_left_height), point_size, (255,0,0), thickness=-1)
            point_img = cv2.putText(point_img, 'A', (cardboard_left_width+plus_text_range, cardboard_left_height+plus_text_range), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 0), font_size, cv2.LINE_AA)

        cardboad_right_width, cardboad_right_height = cal_cardboad_lastHW(anno, cardboard_left_height)
        if cardboad_right_width is not None and cardboad_right_height is not None:
            point_img = cv2.circle(point_img, (cardboad_right_width, cardboad_right_height), point_size, (255,0,0), thickness=-1)
            point_img = cv2.putText(point_img, 'B', (cardboad_right_width+plus_text_range, cardboad_right_height+plus_text_range), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 0), font_size, cv2.LINE_AA)

        prop_width, prop_height = cal_prop_HW(anno, cardboard_left_height)
        if prop_width is not None and prop_height is not None:
            point_img = cv2.circle(point_img, (prop_width, prop_height), point_size, (255,0,0), thickness=-1)
            point_img = cv2.putText(point_img, 'C', (prop_width+plus_text_range, prop_height+plus_text_range), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 0), font_size, cv2.LINE_AA)

        next_prop_width, next_prop_height = cal_next_prop_HW(anno, cardboard_left_height, int(W/2))
        if next_prop_width is not None and next_prop_height is not None:
            point_img = cv2.circle(point_img, (next_prop_width, next_prop_height), point_size, (255,0,0), thickness=-1)
            point_img = cv2.putText(point_img, 'D', (next_prop_width+plus_text_range, next_prop_height+plus_text_range), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 0), font_size, cv2.LINE_AA)

        if cardboad_right_width is not None and cardboad_right_height is not None and cardboad_right_width is not None and cardboad_right_height is not None:
            point_img = cv2.line(point_img, (cardboard_left_width, cardboard_left_height), (cardboad_right_width, cardboad_right_height), (0, 0, 255), thickness=line_size, lineType=cv2.LINE_AA)

        if prop_width is not None and prop_height is not None and next_prop_width is not None and next_prop_height is not None:
            point_img = cv2.line(point_img, (prop_width, prop_height), (next_prop_width, next_prop_height), (0, 0, 255), thickness=line_size, lineType=cv2.LINE_AA)
        #plt.imshow(point_img),plt.show()
        save_img(point_img, name)

if __name__ == '__main__':
    argvs = sys.argv
    image_dir_path = argvs[1]
    anno_dir_path = argvs[2]
    images, names = load_image(image_dir_path, mask=None)
    annos, _ = load_image(anno_dir_path, mask=True)
    draw_poinline(annos, images, names)
