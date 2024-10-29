import time
import torch
from ultralytics import YOLO

model = YOLO('yolo11x_threat.pt')

# Prepare a sample image or batch of images
image = "./datasets/dangerous-objects/test/images/000ebb94-Shotgun_4_jpeg.rf.1090d4e4dc526c9ccda1d8027951e98b.jpg"  # or torch tensor of shape (B, C, H, W)

# Warm-up run (important for GPU)
for _ in range(10):
    _ = model(image)

# Measure inference time
n_runs = 100
total_time = 0

with torch.no_grad():  # Disable gradient computation for inference
    for _ in range(n_runs):
        start_time = time.time()
        results = model(image)
        end_time = time.time()
        total_time += (end_time - start_time)

avg_time = total_time / n_runs
print(f"Average inference time over {n_runs} runs: {avg_time*1000:.2f} ms")