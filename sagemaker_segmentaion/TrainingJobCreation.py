import boto3
import re
import sagemaker
from sagemaker import get_execution_role
import time
from time import gmtime, strftime
import json

role = get_execution_role()
sess = sagemaker.Session()
s3 = boto3.resource('s3')

training_image = sagemaker.amazon.amazon_estimator.get_image_uri(boto3.Session().region_name,
                                                                 'semantic-segmentation', repo_version='latest')
print(training_image)




augmented_manifest_filename_train = 'train-ano/semantic-train/manifests/output/output.manifest' # Replace with the filename for your training data.
augmented_manifest_filename_validation = 'valid-ano/semantic-valid/manifests/output/output.manifest' # Replace with the filename for your validation data.
bucket_name = "sagemaker-magi-segment" # Replace with your bucket name.

s3_output_path = 's3://{}/output'.format(bucket_name) # Replace with your desired output directory.

# Defines paths for use in the training job request.
s3_train_data_path = 's3://{}/{}'.format(bucket_name, augmented_manifest_filename_train)
s3_validation_data_path = 's3://{}/{}'.format(bucket_name, augmented_manifest_filename_validation)

print("Augmented manifest for training data: {}".format(s3_train_data_path))
print("Augmented manifest for validation data: {}".format(s3_validation_data_path))


# In[3]:


augmented_manifest_s3_key = s3_train_data_path.split(bucket_name)[1][1:]
s3_obj = s3.Object(bucket_name, augmented_manifest_s3_key)
augmented_manifest = s3_obj.get()['Body'].read().decode('utf-8')
augmented_manifest_lines = augmented_manifest.split('\n')

num_training_samples = len(augmented_manifest_lines) # Compute number of training samples for use in training job request.


print('Preview of Augmented Manifest File Contents')
print('-------------------------------------------')
print('\n')

for i in range(1):
    print('Line {}'.format(i+1))
    print(augmented_manifest_lines[i])
    print('\n')


# In[4]:


augmented_manifest_s3_key = s3_validation_data_path.split(bucket_name)[1][1:]
s3_obj = s3.Object(bucket_name, augmented_manifest_s3_key)
augmented_manifest = s3_obj.get()['Body'].read().decode('utf-8')
augmented_manifest_lines = augmented_manifest.split('\n')

num_training_samples = len(augmented_manifest_lines) # Compute number of training samples for use in training job request.


print('Preview of Augmented Manifest File Contents')
print('-------------------------------------------')
print('\n')

for i in range(1):
    print('Line {}'.format(i+1))
    print(augmented_manifest_lines[i])
    print('\n')


attribute_names = list(json.loads(augmented_manifest_lines[0]).keys())
attribute_names = [attrib for attrib in attribute_names if 'meta' not in attrib]



attribute_names = ["source-ref","semantic-train-ref"]
valid_attribute_names = ["source-ref", "semantic-valid-ref"]



try:
    if attribute_names == ["source-ref","XXXX"]:
        raise Exception("The 'attribute_names' variable is set to default values. Please check your augmented manifest file for the label attribute name and set the 'attribute_names' variable accordingly.")
except NameError:
    raise Exception("The attribute_names variable is not defined. Please check your augmented manifest file for the label attribute name and set the 'attribute_names' variable accordingly.")

# Create unique job name 
job_name_prefix = 'groundtruth-augmented-manifest-demo'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
job_name = job_name_prefix + timestamp

training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": training_image, # NB. This is one of the named constants defined in the first cell.
        "TrainingInputMode": "Pipe"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": s3_output_path
    },
    "ResourceConfig": {
        "InstanceCount": 1,   
        "InstanceType": "ml.p3.2xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": { # NB. These hyperparameters are at the user's discretion and are beyond the scope of this demo.
         "backbone": "resnet-50",
         "use_pretrained_model": "True",
         "algorithm" : "fcn",
         "lr_scheduler": "poly", 
         "num_classes": "3",
         "num_training_samples": str(num_training_samples),
         "epochs": "30",
         "mini_batch_size":str(1),
         "validation_mini_batch_size":str(1),
         "learning_rate": "0.001",
         "gamma1": "0.90",
         "gamma2": "0.90",
         "optimizer": "sgd",
         "weight_decay": "0.0001",
         "momentum": "0.9"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 86400
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "AugmentedManifestFile", # NB. Augmented Manifest
                    "S3Uri": s3_train_data_path,
                    "S3DataDistributionType": "FullyReplicated",
                    "AttributeNames": attribute_names # NB. This must correspond to the JSON field names in your augmented manifest.
                }
            },
            "ContentType": "application/x-recordio",
            "RecordWrapperType": "RecordIO",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "AugmentedManifestFile", # NB. Augmented Manifest
                    "S3Uri": s3_validation_data_path,
                    "S3DataDistributionType": "FullyReplicated",
                    "AttributeNames": valid_attribute_names # NB. This must correspond to the JSON field names in your augmented manifest.
                }
            },
            "ContentType": "application/x-recordio",
            "RecordWrapperType": "RecordIO",
            "CompressionType": "None"
        }
    ]
}
 
print('Training job name: {}'.format(job_name))
print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))




client = boto3.client(service_name='sagemaker')
client.create_training_job(**training_params)

# Confirm that the training job has started
status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))



TrainingJobStatus = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
SecondaryStatus = client.describe_training_job(TrainingJobName=job_name)['SecondaryStatus']
print(TrainingJobStatus, SecondaryStatus)
while TrainingJobStatus !='Completed' and TrainingJobStatus!='Failed':
    time.sleep(60)
    TrainingJobStatus = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    SecondaryStatus = client.describe_training_job(TrainingJobName=job_name)['SecondaryStatus']
    print(TrainingJobStatus, SecondaryStatus)



training_info = client.describe_training_job(TrainingJobName=job_name)
print(training_info)
