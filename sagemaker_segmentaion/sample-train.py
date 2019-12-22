# https://dev.classmethod.jp/machine-learning/2018advent-calendar-sagemaker-20181201/

%%time
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
print(role)

sess = sagemaker.Session()

bucket = "****-annotation"

prefix = '****-train'
print(bucket)


from sagemaker.amazon.amazon_estimator import get_image_uri
training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")
print (training_image)



import os
import shutil

# Create directory structure mimicing the s3 bucket where data is to be dumped.
VOC2012 = 'VOCdevkit/VOC2012'
os.makedirs('train', exist_ok=True)
os.makedirs('validation', exist_ok=True)
os.makedirs('train_annotation', exist_ok=True)
os.makedirs('validation_annotation', exist_ok=True)

# Create a list of all training images.
filename = VOC2012+'/ImageSets/Segmentation/train.txt'
with open(filename) as f:
    train_list = f.read().splitlines()

# Create a list of all validation images.
filename = VOC2012+'/ImageSets/Segmentation/val.txt'
with open(filename) as f:
    val_list = f.read().splitlines()

# Move the jpg images in training list to train directory and png images to train_annotation directory.
for i in train_list:
    shutil.copy2(VOC2012+'/JPEGImages/'+i+'.jpg', 'train/')
    shutil.copy2(VOC2012+'/SegmentationClass/'+i+'.png','train_annotation/' )

# Move the jpg images in validation list to validation directory and png images to validation_annotation directory.
for i in val_list:
    shutil.copy2(VOC2012+'/JPEGImages/'+i+'.jpg', 'validation/')
    shutil.copy2(VOC2012+'/SegmentationClass/'+i+'.png','validation_annotation/' )


# In[7]:


import glob
num_training_samples=len(glob.glob1('train',"*.jpg"))

print ( ' Num Train Images = ' + str(num_training_samples))
assert num_training_samples == len(glob.glob1('train_annotation',"*.png"))

print ( ' Num Validation Images = ' + str(len(glob.glob1('validation',"*.jpg"))))
assert len(glob.glob1('validation',"*.jpg")) == len(glob.glob1('validation_annotation',"*.png"))


# In[8]:


import json
label_map = { "scale": 1 }
with open('train_label_map.json', 'w') as lm_fname:
    json.dump(label_map, lm_fname)


# In[9]:


# Create channel names for the s3 bucket.
train_channel = prefix + '/train'
validation_channel = prefix + '/validation'
train_annotation_channel = prefix + '/train_annotation'
validation_annotation_channel = prefix + '/validation_annotation'
# label_map_channel = prefix + '/label_map'


# In[10]:


get_ipython().run_cell_magic('time', '', "# upload the appropraite directory up to s3 respectively for all directories.\nsess.upload_data(path='train', bucket=bucket, key_prefix=train_channel)\nsess.upload_data(path='validation', bucket=bucket, key_prefix=validation_channel)\nsess.upload_data(path='train_annotation', bucket=bucket, key_prefix=train_annotation_channel)\nsess.upload_data(path='validation_annotation', bucket=bucket, key_prefix=validation_annotation_channel)\n# sess.upload_data(path='train_label_map.json', bucket=bucket, key_prefix=label_map_channel)")


# In[11]:


s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
print(s3_output_location)


# In[16]:


# Create the sagemaker estimator object.
ss_model = sagemaker.estimator.Estimator(training_image,
                                         role,
                                         train_instance_count = 1,
                                         train_instance_type = 'ml.p3.2xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         output_path = s3_output_location,
                                         base_job_name = 'ss-notebook-demo',
                                         sagemaker_session = sess)


# In[17]:


# Setup hyperparameters
ss_model.set_hyperparameters(backbone='resnet-50', # This is the encoder. Other option is resnet-50
                             algorithm='fcn', # This is the decoder. Other option is 'psp' and 'deeplab'
                             use_pretrained_model='True', # Use the pre-trained model.
                             crop_size=240, # Size of image random crop.
                             num_classes=21, # Pascal has 21 classes. This is a mandatory parameter.
                             epochs=10, # Number of epochs to run.
                             learning_rate=0.0001,
                             optimizer='rmsprop', # Other options include 'adam', 'rmsprop', 'nag', 'adagrad'.
                             lr_scheduler='poly', # Other options include 'cosine' and 'step'.
                             mini_batch_size=16, # Setup some mini batch size.
                             validation_mini_batch_size=16,
                             early_stopping=True, # Turn on early stopping. If OFF, other early stopping parameters are ignored.
                             early_stopping_patience=2, # Tolerate these many epochs if the mIoU doens't increase.
                             early_stopping_min_epochs=10, # No matter what, run these many number of epochs.
                             num_training_samples=num_training_samples) # This is a mandatory parameter, 1464 in this case.


# In[18]:


# Create full bucket names
s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)
s3_train_annotation = 's3://{}/{}'.format(bucket, train_annotation_channel)
s3_validation_annotation = 's3://{}/{}'.format(bucket, validation_annotation_channel)

distribution = 'FullyReplicated'
# Create sagemaker s3_input objects
train_data = sagemaker.session.s3_input(s3_train_data, distribution=distribution,
                                        content_type='image/jpeg', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3_validation_data, distribution=distribution,
                                        content_type='image/jpeg', s3_data_type='S3Prefix')
train_annotation = sagemaker.session.s3_input(s3_train_annotation, distribution=distribution,
                                        content_type='image/png', s3_data_type='S3Prefix')
validation_annotation = sagemaker.session.s3_input(s3_validation_annotation, distribution=distribution,
                                        content_type='image/png', s3_data_type='S3Prefix')

data_channels = {'train': train_data,
                 'validation': validation_data,
                 'train_annotation_channel': train_annotation,
                 'validation_annotation_channel':validation_annotation}


# train & dep;oy


ss_model.fit(inputs=data_channels, logs=True)
ss_predictor = ss_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')
