import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation.segmentanything.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam_checkpoint = "segmentation\sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)

image_path = "test_images\image_1.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image_rgb)

def get_bbox_from_mask(mask):
    # Get the coordinates of the mask where it's non-zero
    pos = np.where(mask)
    
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return xmin, ymin, xmax, ymax

# Draw bounding boxes on the image
image_with_boxes = image_rgb.copy()

cropped_images = []

for idx, mask in enumerate(masks):
    bbox = get_bbox_from_mask(mask['segmentation'])
    xmin, ymin, xmax, ymax = bbox

    cropped_img = img[ymin:ymax, xmin:xmax]
    cropped_images.append(cropped_img)

    cv2.imwrite(f'cropped_object_{idx}.png', cropped_img)

    cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Draw blue boxes

# Show the image with bounding boxes
plt.figure(figsize=(10, 10))
cv2.imwrite(f'final_image.png', cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))