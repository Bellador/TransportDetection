import os
import cv2
import json
import requests
#import nsvision as nv
# from PIL import Image

'''
script to the sending requests to the tensorflow model serving RestAPI of the FSNS OCR model

'''
# 0. set serving model url
model_url = "http://localhost:8501/v1/models/fsns_attention_model:predict"

# 1. load test image
image_path = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\attention_ocr_fsns\python\testdata\fsns_train_06.png"

# image = nv.imread(image_path, resize=(150,150), normalize=True)
# image = nv.expand_dims(image, axis=0)

#data = json.dumps({
#	"instances": [{"inputs": image.tolist()}]
#	})

image_content = cv2.imread(image_path, 1).astype('str').tolist()
data = json.dumps({
	"instances": [{"inputs": image_content}]
	})

headers = {"content-type": "application/json"}

print(f'Data: {data}')

response = requests.post(model_url, data=data, headers=headers)
print(response.json())
#result = int(response.json()['predictions'][0][0])
# print(label[result])