import torch
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
import json
import requests


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.30


class Item(BaseModel):
    image_base64: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"Health": "OK"}

@app.post("/predict")
def predict_class(item: Item):

    img_data = base64.b64decode(item.image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_copy = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    
    bbox_df = results.pandas().xyxy[0]
    
    print(bbox_df)

    result_dict = {}
    
    for idx, row in bbox_df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        cropped_img = img[ymin:ymax, xmin:xmax]
        
        cv2.imwrite(f'cropped_object.png', cropped_img)

        payload = {
            'image_path': 'cropped_object.png',
        }

        url = 'http://127.0.0.1:8000/predict'

        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)

        class_name = response.json()['class']
        probability = response.json()['probability']

        if probability>=40:
            try:
                result_dict[class_name] += 1
            except:
                result_dict[class_name] = 1

            cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite(f'final_image.png', img_copy)

    success, encoded_image = cv2.imencode('.jpg', img_copy)

    if success:
        base64_str = base64.b64encode(encoded_image).decode('utf-8')

    return json.dumps({"image": base64_str, "items": result_dict})