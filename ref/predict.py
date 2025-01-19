from ultralytics import YOLO
import shutil
import os
import cv2

# Load the model from the specified path
model_path = '/mnt/fs1/AI Models/Pole Detection/v02/yolov8n-obb-trained-1280_TEST.pt'
model = YOLO(model_path)  # Initialize the model with pre-trained weights

# Image path
image_path = '/mnt/fs1/Sorted_For_CVAT_Annotation/batch_1_excellent_1000/section2/images/Recording_20220323_110843_Flare48M_282_132924676605899189.jpg'

results = model.predict(source=image_path, save = True)

# Save the inferenced Image
save_path = results[0].save_dir
filename =  os.path.basename(results[0].path)
saved_file = os.path.join(save_path, filename)

if os.path.exists(saved_file):
    destination_file = os.path.join('./inferencing', filename)
    
    # Copy the file
    shutil.copy(saved_file, destination_file)
    print(f"File copied to {destination_file}")
else:
    print("No output file found to copy.")
