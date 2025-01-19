from ultralytics import YOLO
import torch
import os

# Set CUDA environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the pretrained YOLO model
model = YOLO('yolov8n.pt')

# Move the model to the specified device
model.to(device)

# Train the model
results = model.train(
    data='./data/data.yaml',
    epochs=1,
    imgsz=640,
    device=0,
    batch=16,  # Adjusted batch size
    amp=True,  # Mixed precision training
    workers=8,  # Reduced number of data loader workers
    project="./training_project/runs",
    name="v00"
)

#Save the trained model
model_save_path = '/mnt/fs1/AI Models/Pole Detection/v02/yolov8n-obb-trained-640.pt'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
