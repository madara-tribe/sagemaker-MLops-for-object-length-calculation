import json
import boto3
import traceback
import asyncio
import numpy as np
import cv2
import base64
import ast
from io import BytesIO
import io
import PIL
from PIL import Image
from CalculateHorizontalLength import *
from CalculateVerticalLength import *

async def call_predict_mask_endpoint(runtime, img):

    segment_endpoint_name='semantic-segmentation-*****'
    filename = "/tmp/dum.jpg"
    cv2.imwrite(filename, img)
    im = PIL.Image.open(filename)
    im.thumbnail([480, 480],Image.ANTIALIAS)
    im.save(filename, "JPEG")


    with open(filename, 'rb') as image:
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


def Base64ToNadarry(img_base64):
    img_data = base64.b64decode(img_base64)
    img_np = np.frombuffer(img_data, np.uint8)
    src = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)
    return src


def lambda_handler(event, context):
    try:
        #print(event)
        base64L = event['left'][23:]
        base64R = event["right"][23:]
        Limage = Base64ToNadarry(base64L)
        Rimage = Base64ToNadarry(base64R)
        if Rimage is not None:
            Limage = cv2.flip(Rimage, 1)
        runtime = boto3.Session().client('sagemaker-runtime')

        async def call_apis():
            coroutines = (
            call_predict_mask_endpoint(runtime, Limage),
            call_predict_mask_endpoint(runtime, Rimage)
            )
            return await asyncio.gather(*coroutines)
        loop = asyncio.get_event_loop()
        call_result = loop.run_until_complete(call_apis())
        #print("finished looped")
        _, W = np.shape(call_result[0])
        print(np.shape(call_result[0]))
        L_vertical_lenth = cal_vertical_lenth(call_result[0])
        L_horizontal_length = horizontal_length(call_result[0])
        if L_vertical_lenth == False:
            print('left image retlv Error : ', L_vertical_lenth)
            L_vertical_lenth = 0
        if L_horizontal_length == False:
            print('left image retlh Error : ', L_horizontal_length)
            L_horizontal_length = 0

        R_vertical_length = cal_vertical_lenth(call_result[1])
        R_horizontal_length = horizontal_length(call_result[1])
        if R_vertical_length == False:
            print('right image retrv Error : ', R_vertical_lenth)
            R_vertical_lenth = 0
        if R_horizontal_length == False:
            print('right image retrh Error : ', R_horizontal_length)
            R_horizontal_length = 0
        Vlength = [R_vertical_length if R_vertical_length > L_vertical_lenth else L_vertical_lenth]
        response = {'statusCode' : 200,
                    "result_left": L_horizontal_length,
                    "result_right":R_horizontal_length,
                    "vertical" : Vlength,
                    'completed' : 1
                  }
        return response

    except:
        traceback.print_exc()
        return {
            'statusCode' : 200,
            'headers': {
                    'access-control-allow-origin' : '*',
                    'content-type' : 'application/json'
                            },
            'completed' : 0
        }
