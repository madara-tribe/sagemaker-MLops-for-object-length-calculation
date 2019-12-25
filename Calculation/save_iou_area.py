import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sagemaker_evaluation import *
from iou_score import metrics_np, mean_iou_np, mean_dice_np


def mask_to_color(mask, class_ids):
    mask_h, mask_w = np.shape(mask)
    masked = np.zeros([mask_h, mask_w, 3], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(mask_w):
            class_id = mask[h, w]
            r, b, g = (0, 0, 0)
            if class_id == class_ids:
                r, g, b = (0,100,0)
            elif class_id == class_ids:
                r, g, b = (65,105,225)
            elif class_id == class_ids:
                r, g, b = (255,140,0)
            elif class_id == class_ids:
                r, g, b = (255,140,0)
            elif class_id == class_ids:
                r, g, b = (255,140,0)
            elif class_id == class_ids:
                r, g, b = (255,140,0)
            else:
                r, g, b = (0, 0, 0)

            masked[h, w, 0] = r
            masked[h, w, 1] = g
            masked[h, w, 2] = b
    return masked


# each class score
def make_nclass_mask(gt, pred_annos, n_class):
  ground_truth, pred_annos = [], []
  for idx, (gt, pred) in enumerate(zip(gt, pred_annos)):
      gt_ = np.where((gt == n_class), 255, 0)
      pred_ = np.where((pred == n_class), 255, 0)
      H, W = np.shape(pred_)
      if idx<2:
          plt.imshow(pred.reshape(W, H), "gray"),plt.show()
      ground_truth.apppend(gt_)
      pred_annos.append(pred_)
  return np.array(ground_truth), np.array(pred_annos)

 def save_img(diff, filename, output_dir):
    diff_ = diff.astype(np.uint8)
    bgr = cv2.cvtColor(diff_, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, str(filename)+".png"), bgr)




def save_diff_image(groundtruths, pred_masks, output_dir):
    output_dir = output_dir
    for gt, pred in zip(groundtruths, pred_masks):
        filename, _ = os.path.splitext(os.path.basename(gt))
        if class_id = 1:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)
        if class_id = 2:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)
        if class_id = 3:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)
        if class_id = 4:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)
        if class_id = 5:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)
        if class_id = 6:
            n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
            diff = n_class_gt - n_class_pred
            color_diff = mask_to_color(diff, class_id)
            save_img(color_diff, filename, output_dir)

def call_gt_pred_image(train_annotation_path, train_image_path):
    groundtruth = load_image(train_annotation_path)
    train_files = load_image(train_image_path)
    pred_mask = call_predict_mask_endpoint(train_files)
    H, W = np.shape(pred_masks[1])
    groundtruths = groundtruth.reshape(len(groundtruth), W, H, 1)
    pred_masks = pred_mask.reshape(len(pred_mask), W, H, 1)
    return groundtruths, pred_masks

def save_diff():
    train_annotation_path = "ykkap_image_path"
    train_image_path = "ykkap_annotation_path"
    output_dir = "diff_images"
    groundtruths, pred_masks = call_gt_pred_image(train_annotation_path, train_image_path)
    save_diff_image(groundtruths, pred_masks, output_dir)


if __name__ == '__main__':
    save_diff()
