
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import boto3
import json
import io
from io import BytesIO
import numpy as np
import time
import cv2
import glob
import sys
from sagemaker.amazon.amazon_estimator import get_image_uri



def load_image(inputs_dir):
   files = []
   files += glob.glob(os.path.join(inputs_dir, '*.png'))
   files += glob.glob(os.path.join(inputs_dir, '*.jpg'))
   files += glob.glob(os.path.join(inputs_dir, '*.jpeg'))
   files += glob.glob(os.path.join(inputs_dir, '*.PNG'))
   files += glob.glob(os.path.join(inputs_dir, '*.JPG'))
   files += glob.glob(os.path.join(inputs_dir, '*.JPEG'))
   print('target files : ', len(files))
   return files


def save_pred_mask(files, outputs_dir):
    runtime = boto3.Session().client('sagemaker-runtime')
    for k, f in enumerate(files):
        mask = call_predict_mask_endpoint(f, runtime)
        filename, _ = os.path.splitext(os.path.basename(f))
        cv2.imwrite(os.path.join(outputs_dir, filename + '.png'), mask)
        print('[%04d] saved : ' % (k+1), filename)




def call_predict_mask_endpoint(filename, runtime):

   segment_endpoint_name='semantic-segmentation-2019-12-11-04-14-04-986'
   im = PIL.Image.open(filename)
   im.thumbnail([480, 480],Image.ANTIALIAS)
   tmp_filename = './tmp.jpeg'
   im.save(tmp_filename, "JPEG")

   mask = None
   with open(tmp_filename, 'rb') as image:
       img = image.read()
       img = bytearray(img)

       endpoint_response = runtime.invoke_endpoint(
           EndpointName=segment_endpoint_name,
           Body=img,
           ContentType='image/jpeg',
           Accept = 'image/png'
       )
       results = endpoint_response['Body'].read()
       mask = np.array(Image.open(io.BytesIO(results)))
   return mask


def plot_result(mask, num=7):
    num_classes=num
    plt.imshow(mask, vmin=0, vmax=num_classes-1, cmap='jet')
    plt.show()








def predict_save():
    # deploy
    sess = sagemaker.Session()
    training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")
    print(training_image)

    role='arn:aws***/AmazonSageMaker-ExecutionRole-20190704T193457'
    model =  Model(model_data='s3://****/sample-train/model2/output/model.tar.gz',
                   image=training_image,
                   role=role)
    model.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')

    # check & make endpoint
    runtime = boto3.client('sagemaker')
    tuntime.list_endpoints()
    endpoint_name='semantic-segmentation-*****'

    # predict & save
    dir_path = 'trains'
    files = load_image(dir_path)
    print(len(files))

    save_dir = 'train_mask'
    for idx, f in enumerate(files):
        mask = call_predict_mask_endpoint(f, runtime)
        plot_reult(mask)
        save_pred_mask(f, save_dir)





from sagemaker.analytics import TrainingJobAnalytics

def plot_training_loss_curve():
    # ジョブ名
    training_job_name = '***'
    # メトリクス名(train:loss, train:throughput, validation:mlOU, validation:pixel_accuracy, validation:throughput)
    metric_name = 'train:loss'

    # セッション作成
    botosess = boto3.Session(region_name='ap-northeast-1')
    sess = sagemaker.Session(botosess)

    # メトリクスデータをデータフレーム形式で取得
    metrics_dataframe = TrainingJobAnalytics(training_job_name=training_job_name,metric_names=[metric_name], sagemaker_session=sess).dataframe()

    # プロット
    metrics_dataframe.plot(x='timestamp', y='value', legend=False).set_ylabel(metric_name)

if __name__ == '__main__':
    predict_save()
    !tar czf train_mask.tar.gz train_mask
    plot_training_loss_curve()
