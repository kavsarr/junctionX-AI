import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.20

def detect_and_crop_objects(image_path):

    img = cv2.imread(image_path)
    img_copy = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    
    bbox_df = results.pandas().xyxy[0]
    
    print(bbox_df)
    
    cropped_images = []
    for idx, row in bbox_df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        cropped_img = img[ymin:ymax, xmin:xmax]
        cropped_images.append(cropped_img)
        
        cv2.imwrite(f'cropped_object.png', cropped_img)
        
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite(f'final_image.png', cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))

    return cropped_images

image_path = 'test_images\image.jpg'
cropped_objects = detect_and_crop_objects(image_path, save_cropped=True)
