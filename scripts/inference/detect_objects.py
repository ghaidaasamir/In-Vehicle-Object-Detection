import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "in_vehicle_objects_dataset/test") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/weights", "best.pt") 

CONF_THRESH = 0.5
IOU_THRESH = 0.45

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

print(f"Found {len(image_files)} images to process")

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(INPUT_DIR, image_file)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Warning: Could not read image {image_file}")
        continue
    
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=640,
        device="0",
        augment=False
    )
    
    annotated_frame = results[0].plot()
    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, annotated_frame)

print(f"\nInference complete! Annotated images saved to: {OUTPUT_DIR}")

print("\nClass mapping:")
for i, name in model.names.items():
    print(f"{i}: {name}")