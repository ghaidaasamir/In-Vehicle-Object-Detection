from ultralytics import YOLO
import os
from datetime import datetime
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "in_vehicle_objects.yaml") 

class_config = {
    "class_names": {
        0: "belt",
        1: "cup",
        2: "front",
        3: "left",
        4: "phone",
        5: "right",
        6: "wheel",
    },
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "imgsz": 640,
    "base_lr": 0.01
}

def setup_dirs(base_path="main_7_classes_detection"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(base_path, f"exp_{timestamp}")
    os.makedirs(run_path, exist_ok=True)
    return run_path

run_path = setup_dirs()

model = YOLO("yolo11x.pt")  


results = model.train(
    data=CONFIG_PATH,
    epochs=200,
    batch=16 if torch.cuda.is_available() else 8,
    imgsz=640,
    rect=True,  
    device="0",
    workers=8,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.8, 
    hsv_v=0.4,
    degrees=15.0,  
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.001,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    project="main_7_classes_detection",
    name=os.path.basename(run_path),
    save_period=5,
    exist_ok=False,
    val=True,
    plots=True,
    save_json=True
)

print("\nTraining Metrics:")
train_csv = os.path.join(run_path, "results.csv")
if os.path.exists(train_csv):
    print(f"Metrics saved to: {train_csv}")

val_results = model.val(
    data=CONFIG_PATH,
    batch=class_config["batch_size"],
    imgsz=class_config["imgsz"]
)

print("\nValidation Metrics:")
for class_id, class_name in class_config["class_names"].items():
    print(f"{class_name}: mAP50 = {val_results.box.maps[class_id]:.4f}")

print(f"\nBest weights: {os.path.join(run_path, 'weights', 'best.pt')}")
