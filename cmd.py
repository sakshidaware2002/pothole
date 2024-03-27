import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())
post_training_files_path = 'C:\\Users\\prasad\\PycharmProjects\\potholeDetection\\runs\\segment\\train6'
#valid_images_path = os.path.join(dataset_path, 'valid', 'images')
best_model_path = os.path.join(post_training_files_path, 'weights/best.pt')
best_model = YOLO(best_model_path)
#new commit
image = cv2.imread(args["input"])
results = best_model.predict(source=image, imgsz=640, conf=0.5)
annotated_image = results[0].plot()
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
num_subplots = 1 + (len(results[0].masks.data) if results[0].masks is not None else 0)
fig, axes = plt.subplots(1, num_subplots, figsize=(20, 10))
axes[0].imshow(annotated_image_rgb)
axes[0].set_title('Result')
axes[0].axis('off')
if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    for i, mask in enumerate(masks):
        # Threshold the mask to make sure it's binary
        # Any value greater than 0 is set to 255, else it remains 0
        binary_mask = (mask > 0).astype(np.uint8) * 255
        axes[i+1].imshow(binary_mask, cmap='gray')
        #axes[i+1].set_title(f'Segmented Mask {i+1}')
        axes[i+1].axis('off')

# Adjust layout and display the subplot
plt.tight_layout()
plt.show()
cv2.imwrite(args["output"], annotated_image_rgb)


