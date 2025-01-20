from ultralytics import YOLO
import shutil
import os
import cv2

# Load the model from the specified path
model_path = '/app/training_projects/runs/v00/weights/best.pt'
model = YOLO(model_path)  # Initialize the model with pre-trained weights

# Input Image path
image_path = '/app/data/other/test.png'

results = model.predict(source=image_path, save = True)


#Save the result
for result in results:
    result.save(filename="/app/data/other/result.jpg")

# Save the inferenced Image
# save_path = results[0].save_dir
# filename =  os.path.basename(results[0].path)
# saved_file = os.path.join(save_path, filename)

# if os.path.exists(saved_file):
#     destination_file = os.path.join('/app/data/other', filename[:-4] + '_inference.png')
    
#     # Copy the file
#     shutil.copy(saved_file, destination_file)
#     print(f"File copied to {destination_file}")
# else:
#     print("No output file found to copy.")
