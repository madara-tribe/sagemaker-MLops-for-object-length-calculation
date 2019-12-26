import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sagemaker_evaluation import *
from iou_score import metrics_np, mean_iou_np, mean_dice_np


# each class score
def make_nclass_mask(gts, pred_annos, n_class):
  ground_truth, pred_annos = [], []
  for idx, (gt, pred) in enumerate(zip(gts, pred_annos)):
      gt_ = np.where((gt == n_class), 255, 0)
      pred_ = np.where((pred == n_class), 255, 0)
      H, W = np.shape(pred_)
      if idx<2:
          plt.imshow(pred.reshape(W, H), "gray"),plt.show()
      ground_truth.apppend(gt_)
      pred_annos.append(pred_)
  return np.array(ground_truth), np.array(pred_annos)




def calculate_iou_score(groundtruths, pred_masks, total = True, class_id_is = 1):
    image_name = []
    class_id = class_id_is
    if total:
        total_score = mean_iou_np(groundtruths, pred_masks)
        total_df = pd.DataFrame(total_score)
        total_df.to_csv("total_iou_score.csv")
    else:
        for gt, pred in zip(groundtruths, pred_masks):
            filename, _ = os.path.splitext(os.path.basename(gt))
            image_name.append(filename)
            if class_id = class_id:
                score1 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score1.append(score)
            elif class_id = class_id:
                score2 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score2.append(score)
            elif class_id = class_id:
                score3 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score3.append(score)
            elif class_id = class_id:
                score4 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score4.append(score)
            elif class_id = class_id:
                score5 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score5.append(score)
            elif class_id = class_id:
                score6 = []
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                score6.append(score)
            df = pd.concat([image_name, score1, score2, score3, score4, score5, score6], axis=1)
            print(df.shape)
            lists=["filename", "Prop", "Cardboard", "Pedestal", "Sideboard", "Paper", "Band"]
            df.columns=lists
            df.to_csv("each_iou_score.csv")


def call_gt_pred_image(train_annotation_path, train_image_path):
    groundtruth = load_image(train_annotation_path)
    train_files = load_image(train_image_path)
    pred_mask = call_predict_mask_endpoint(train_files)
    H, W = np.shape(pred_masks[1])
    groundtruths = groundtruth.reshape(len(groundtruth), W, H, 1)
    pred_masks = pred_mask.reshape(len(pred_mask), W, H, 1)
    return groundtruths, pred_masks


def cal_score():
    train_annotation_path = "image_path"
    train_image_path = "annotation_path"
    groundtruths, pred_masks = call_gt_pred_image(train_annotation_path, train_image_path)
    calculate_iou_score(groundtruths, pred_masks, total = None, class_id_is = 1)


if __name__ == '__main__':
    cal_score()

