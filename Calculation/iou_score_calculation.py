import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tools.iou_score import *

endpoint_name='semantic-segmentation-****'

def deploy():
    sess = sagemaker.Session()
    training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")
    print(training_image)
    role='arn:aws***/AmazonSageMaker-ExecutionRole-20190704T193457'
    model =  Model(model_data='s3://****/sample-train/model2/output/model.tar.gz',
               image=training_image,
               role=role)
    model.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')

    # make endpoint
    client = boto3.client('sagemaker')
    peinr("endpoints_inf", client.list_endpoints())

def call_predict_mask_endpoint(filename):
    im = PIL.Image.open(filename)
    im.thumbnail([480, 480],PIL.Image.ANTIALIAS)
    im.save(filename, "JPEG")
    
    with open(filename, 'rb') as image:
        img = image.read()
        img = bytearray(img)

    endpoint_response = boto3.client('sagemaker-runtime').invoke_endpoint(
        EndpointName=endpoint_name,
        Body=img,
        ContentType='image/jpeg',
        Accept = 'image/png'
    )
    results = endpoint_response['Body'].read()
    mask = np.array(Image.open(io.BytesIO(results)))
    print(mask.shape, np.unique(mask))
    return mask

# each class score
def make_nclass_mask(groundtruths, pred_annos, n_class):
  ground_truth, pred_annos = [], []
  for idx, (gt, pred) in enumerate(zip(groundtruths, pred_annos)):
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




def calculate_iou_score(groundtruths, pred_masks, total = True, class_id_is = 1):
    image_name = []
    scores = []
    class_id = class_id_is
    if total:
        total_score = mean_iou_np(groundtruths, pred_masks)
        total_df = pd.DataFrame(total_score)
        total_df.to_csv("total_iou_score.csv")
    else:
        for gt, pred in zip(groundtruths, pred_masks):
            filename, _ = os.path.splitext(os.path.basename(gt))
            image_name.append(filename)
            if class_id == 1:
                lists=["filename", "Prop"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            elif class_id == 2:
                lists=["filename", "Cardboard"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            elif class_id == 3:
                lists=["filename", "Pedestal"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            elif class_id == 4:
                lists=["filename", "Sideboard"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            elif class_id == 5:
                lists=["filename", "Paper"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            elif class_id == 6
                lists=["filename", "Band"]
                n_class_gt, n_class_pred = make_nclass_mask(gt, pred, class_id)
                score = mean_iou_np(n_class_gt, n_class_pred)
                scores.append(score)
            df = pd.concat([image_name, score], axis=1)
            print(df.shape)
            df.columns=lists
            df.to_csv("each_iou_score.csv")



def call_gt_pred_image(train_annotation_path, train_image_path):
    deploy()
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
