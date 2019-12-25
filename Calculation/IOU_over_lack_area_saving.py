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

def deploy():
    sess = sagemaker.Session()

    from sagemaker.amazon.amazon_estimator import get_image_uri
    training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")
    print(training_image)


    role='arn:aws:iam::*****'
    model = Model(model_data='s3://s3-bukect/****/output/model.tar.gz',
                   image=training_image,
                   role=role)
    model.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')


def return_mask(filename, endpoint_name):
    client = boto3.client('sagemaker')
    client.list_endpoints()
    endpoint_name='semantic-segmentation-*****'

    fname, _ = os.path.splitext(os.path.basename(filename))
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
    return mask, fname


# In[58]:


def plot(mask, num=7):
    num_classes=num
    plt.imshow(mask, vmin=0, vmax=num_classes-1, cmap='jet')
    plt.show()

def create_one_class(image, N):
    IMG = []
    for idx, img in enumerate(image):
        dst = np.where((img == N), 255, 0)
        print(np.unique(dst))
        if idx<3:
            plt.imshow(dst, "gray"),plt.show()
        IMG.append(dst)
    return np.array(IMG)


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



def mask_to_indexmap(mask):
    mask_h, mask_w = np.shape(mask)
    masked = np.zeros([mask_h, mask_w, 3], dtype=np.uint8)
    for h in range(mask_h):
        for w in range(mask_w):
            class_id = mask[h, w]
            #print(idx, np.unique(class_id))
            r, b, g = (0, 0, 0)
            if class_id == 255:
                r, g, b = (139, 69, 19)
            else:
                r, g, b = (0, 0, 0) # white

            masked[h, w, 0] = r
            masked[h, w, 1] = g
            masked[h, w, 2] = b

    return masked


def lack_iou_area(diff):
    lack=[]
    for i, img in enumerate(diff):
        img[img==255]=255
        img[img==-255]=0
        print(np.unique(img), img.shape)
        Ls = mask_to_indexmap(img)
        if i<3:
            plt.imshow(Ls),plt.show()
        lack.append(Ls)
    print(len(lack))
    return lack

def over_iou_area(diff):
    over=[]
    for i, img in enumerate(diff):
        img[img==255]=0
        img[img==-255]=255
        print(np.unique(img), img.shape)
        Ls = mask_to_indexmap(img)
        if i<3:
            plt.imshow(Ls),plt.show()
        over.append(Ls)
    print(len(over))
    return over


def save_iou_over_lack_area():
    deploy()

    dir_path = 'images'
    image_path = load_image(dir_path)
    prediction, filename = [return_mask(filename, endpoint_name) for filename in image_path]
    H, W = np.shape(prediction[1])
    plot(prediction[1])


    dir_path = 'indexmap'
    image_path = load_image(dir_path)
    indexs = [cv2.imread(path, 0) for path in image_path]
    GroundTruth = np.array([cv2.resize(img,(W, H), interpolation=cv2.INTER_NEAREST) for img in indexs])
    print(GroundTruth.shape)



    # Prop
    N=6
    pred = one_class(prediction, N)
    print(pred.shape, np.unique(pred))
    gt = one_class(GroundTruth, N)
    print(gt.shape, np.unique(gt))



    diff = [g-p for g, p in zip(gt, pred)]
    lack = lack_diff(diff)
    over = over_diff(diff)
    print(len(over), len(lack))

    # In[72]:


    lack_dir ='Band_lack'
    over_dir = 'Band_over'
    for i, (la, ov, name) in enumerate(zip(lack, over, filename)):
        print(i, name)
        lack_img = la.astype(np.uint8)
        cv2.imwrite(os.path.join(lack_dir, str(name)+".png"), lack_img)
        over_img = ov.astype(np.uint8)
        cv2.imwrite(os.path.join(over_dir, str(name)+".png"), over_img)

if __name__ == '__main__':
    save_iou_over_lack_area()
