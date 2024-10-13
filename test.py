import cv2
import base64
import requests
import json

# Function to convert an image to base64 string
def image_to_base64(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Encode the image as a JPEG (or PNG) and then convert to base64
    _, buffer = cv2.imencode('.jpg', img)  # Change to '.png' if needed
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str

# Define your image path
image_path = 'notebooks.jpeg'  # Replace with your image file path

# Convert the image to base64 string
base64_image = image_to_base64(image_path)

# Define the request payload
payload = {
    'image_base64': base64_image,
}

# Define the endpoint URL
url = 'http://127.0.0.1:8001/predict'

headers = {
    'Content-Type': 'application/json',
}

# Send the request to the API
response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)

# Check the response
if response.status_code == 200:
    print('Image uploaded successfully:', response.json())
else:
    print('Failed to upload image:', response.status_code, response.text)
