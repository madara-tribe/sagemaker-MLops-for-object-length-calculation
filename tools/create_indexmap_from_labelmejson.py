import os
import glob
import json
import cv2
import numpy as np
import pprint
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from enum import Enum
from tqdm import tqdm


BASE_DIR = os.path.join('.', 'data')
IMAGE_BASE_DIR = os.path.join(BASE_DIR, 'images')
ANNOTAION_BASE_DIR = os.path.join(BASE_DIR, 'annos')
MASK_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'mask')
SEGMENTATION_BASE_DIR = os.path.join(BASE_DIR, 'segmentation')
OVERLAY_BASE_DIR = os.path.join(BASE_DIR, 'overlay')
INDEX_BASE_DIR = os.path.join(BASE_DIR, 'indexmap')

TAEGET_DIRS = files_dir = [f for f in os.listdir(ANNOTAION_BASE_DIR) if os.path.isdir(os.path.join(ANNOTAION_BASE_DIR, f))]

class DrawType(Enum):
    POLYGON = 0
    LINE = 1
    CIRCLE = 2


def main(dir_name):
    annos_dir = os.path.join(ANNOTAION_BASE_DIR, dir_name)
    image_dir = os.path.join(IMAGE_BASE_DIR, dir_name)

    if not(os.path.exists(annos_dir)):
        print('annos dir is not found. : ', annos_dir)
        return
    if not(os.path.exists(image_dir)):
        print('image dir is not found. : ', image_dir)
        return

    overlay_dir = os.path.join(OVERLAY_BASE_DIR, dir_name)

    if not(os.path.exists(overlay_dir)):
        os.mkdir(overlay_dir)

    mask_dir = os.path.join(MASK_OUTPUT_BASE_DIR, dir_name)

    if not(os.path.exists(mask_dir)):
        os.mkdir(mask_dir)

    segmentation_dir = os.path.join(SEGMENTATION_BASE_DIR, dir_name)

    if not(os.path.exists(segmentation_dir)):
        os.mkdir(segmentation_dir)

    indexmap_dir = os.path.join(INDEX_BASE_DIR, dir_name)
    if not(os.path.exists(indexmap_dir)):
        os.mkdir(indexmap_dir)

    files = glob.glob(os.path.join(annos_dir, '*.json'))
    files.sort()
    print('###  ', annos_dir)
    print('### target annotation file : ', len(files))
    print('')
    pbar = tqdm(total=len(files), desc="Create", unit=" Files")
    for file in files:
        create_annotation_img(file, image_dir, overlay_dir, mask_dir, segmentation_dir, indexmap_dir)
        pbar.update(1)
    pbar.close()


def create_annotation_img(anno_json, image_dir, overlay_dir, mask_dir, segmentation_dir, indexmap_dir):
    jf = json.load(open(anno_json))
    image_name_base, _ = os.path.splitext(os.path.basename(anno_json))

    original_image_path = os.path.join(image_dir, image_name_base)
    org_img = cv2.imread(original_image_path + '.png')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.jpeg')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.jpg')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.JPG')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.JPEG')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.PNG')
    org_img = org_img.astype(np.uint8)
    image_shape = np.shape(org_img)

    base_indexmap = np.zeros(image_shape[:2])
    masks_prop = np.zeros(image_shape[:2])
    masks_cardboard = np.zeros(image_shape[:2])
    masks_pedestal = np.zeros(image_shape[:2])
    masks_sideboard = np.zeros(image_shape[:2])
    masks_paper = np.zeros(image_shape[:2])
    masks_band = np.zeros(image_shape[:2])

    seg_img = None
    for k, shape in enumerate(jf['shapes']):
        contours = shape['points']

        mask = contours
        if shape['label'] == 'Prop':
            if len(mask)>2:
                img = create_mask_image(image_shape, mask, draw_type=DrawType.POLYGON)
                masks_prop[img>0] = 1
        elif shape['label'] == 'Cardboard':
            if len(mask)>2:
                img = create_mask_image(image_shape, mask, draw_type=DrawType.POLYGON)
                masks_cardboard[img>0] = 1
        elif shape['label'] == 'Pedestal':
            if len(mask)>2:
                img = create_mask_image(image_shape, mask, draw_type=DrawType.POLYGON)
                masks_pedestal[img>0] = 1
        else:
            # skip
            continue

    # by order
    base_indexmap[masks_pedestal > 0] = 3
    base_indexmap[masks_cardboard > 0] = 2
    #base_indexmap[masks_sideboard > 0] = 4
    base_indexmap[masks_prop > 0] = 1
    #base_indexmap[masks_paper > 0] = 5
    #base_indexmap[masks_band > 0] = 6

    indexmap_imgs= base_indexmap
    indexmap_path = os.path.join(indexmap_dir, image_name_base + '.png')
    cv2.imwrite(indexmap_path, indexmap_imgs)


def to_color_mask(mask, color):
    col_r, col_g, col_b = color
    shape = np.shape(mask)
    if len(shape) < 3:
        mask = np.stack([mask, mask, mask], axis=2)

    mask_r = mask[:, :, 2]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 0]

    mask_r[mask_r > 0] = col_r
    mask_g[mask_g > 0] = col_g
    mask_b[mask_b > 0] = col_b

    mask_col = np.stack([mask_b, mask_g, mask_r], axis=2)
    return mask_col


def create_mask_image(image_shape, mask, draw_type=DrawType.POLYGON):
    img_src = np.zeros(image_shape[:2])
    img = Image.fromarray(img_src)
    xy = list(map(tuple, mask))
    if draw_type == DrawType.POLYGON:
        ImageDraw.Draw(img).polygon(xy=xy, outline=255, fill=255)
    elif draw_type == DrawType.LINE:
        ImageDraw.Draw(img).line(xy=xy, width=3, fill=255)
    elif draw_type == DrawType.CIRCLE:
        if len(xy) > 0:
            c_x, c_y = xy[0]
            radius = 4
            xy = [(c_x - radius, c_y - radius), (c_x + radius, c_y + radius)]
            ImageDraw.Draw(img).ellipse(xy, outline=255, fill=255)
    img = np.array(img) * 255
    return img


if __name__ == '__main__':
    if not(os.path.exists(OVERLAY_BASE_DIR)):
        os.mkdir(OVERLAY_BASE_DIR)

    if not(os.path.exists(MASK_OUTPUT_BASE_DIR)):
        os.mkdir(MASK_OUTPUT_BASE_DIR)

    if not(os.path.exists(SEGMENTATION_BASE_DIR)):
        os.mkdir(SEGMENTATION_BASE_DIR)

    if not(os.path.exists(INDEX_BASE_DIR)):
        os.mkdir(INDEX_BASE_DIR)

    for k, dir_name in enumerate(TAEGET_DIRS):
        print(k+1, ' / ', len(TAEGET_DIRS))
        main(dir_name)
    print('done .')
