import argparse
import torch
import os
from ultralytics import YOLO

# Set up argument parser to accept the --cpu flag
parser = argparse.ArgumentParser(description="Train YOLO model on CPU or GPU.")
parser.add_argument('--cpu', action='store_true', help="Force the use of CPU for training.")
args = parser.parse_args()

# Set CUDA environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set device to CPU or GPU based on the argument
device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pretrained YOLO model
model = YOLO('yolov8n.pt')

# Move the model to the specified device
model.to(device)

# Set the device parameter to 0 if using CUDA, or 'cpu' otherwise
device_param = 0 if device.type == 'cuda' else 'cpu'

# Train the model
results = model.train(
    data='/app/data/dataset.yaml',
    epochs=600,
    imgsz=640,
    device=device_param,
    batch=16,  # Adjusted batch size
    amp=True,  # Mixed precision training
    workers=8,  # Reduced number of data loader workers
    project="/app/training_projects/runs",
    name="v00",
    save=True,
    val=True,
    plots=True,
)
