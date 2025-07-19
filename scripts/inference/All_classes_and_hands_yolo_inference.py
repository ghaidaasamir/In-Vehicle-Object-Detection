import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "in_vehicle_objects_dataset/test") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/weights", "best.pt") 

CONF_THRESH = 0.5
IOU_THRESH = 0.45  
IMAGE_SIZE = 640   

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

print(f"Found {len(image_files)} images to process")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=3,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

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
        imgsz=IMAGE_SIZE,
        device="0", 
        augment=False  
    )
    
    output_frame = frame.copy()
    
    yolo_annotated = results[0].plot()
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            box_width = (x_max - x_min) / w
            box_height = (y_max - y_min) / h
            
            print(f"\nHand detected in {image_file} - YOLO format: 0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            
            cv2.rectangle(yolo_annotated, 
                         (int(x_min), int(y_min)), 
                         (int(x_max), int(y_max)), 
                         (0, 255, 0), 2)  # Green box for hands
            cv2.putText(yolo_annotated, 'Hand', (int(x_min), int(y_min) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    print(f"\nImage: {image_file}")
    result = results[0]
    if len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            print(f"Detected: {class_name} (confidence: {confidence:.2f})")
    else:
        print("No objects detected by YOLO model")

    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, yolo_annotated)

print(f"\nInference complete! Annotated images saved to: {OUTPUT_DIR}")

print("\nClass mapping:")
for i, name in model.names.items():
    print(f"{i}: {name}")