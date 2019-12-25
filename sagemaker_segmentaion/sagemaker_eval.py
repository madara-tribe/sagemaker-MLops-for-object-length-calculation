import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import boto3
import json
from io import BytesIO
import numpy as np
import time
import io
import cv2
from natsort import natsorted
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.analytics import TrainingJobAnalytics


endpoint_name='semantic-segmentation-2019-12-11-04-14-04-986'


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



def save_img(input_dir, output_dir):
    runtime = boto3.Session().client('sagemaker-runtime')
    files = load_image(inputs_dir)
    for f in files:
        filename, _ = os.path.splitext(os.path.basename(f))
        mask = call_predict_mask_endpoint(f)
        mask_ = mask.astype(np.uint8)
        bgr = cv2.cvtColor(mask_, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, str(filename)+".png"), bgr)





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




def plot_predict_result(input_dir, endpoint_name, num_classes=7, stop_idx=10):
    masks = []
    files = load_image(input_dir)
    for idx, f in enumerate(files):
        mask = call_predict_mask_endpoint(f)
        num_classes=num_classes
        plt.imshow(mask, vmin=0, vmax=num_classes-1, cmap='jet')
        plt.show()
        if idx == stop_idx:
            break


def plot_training_curve():
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
