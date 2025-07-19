import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import mediapipe as mp

def calculate_iou_hand(box1, box2):
    """Intersection over Union (IoU) between two bounding boxes"""
    # box: [x_min, y_min, x_max, y_max]
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # IoU modified
    # iou = intersection_area / float(box1_area + box2_area - intersection_area)
    iou = intersection_area / float(box2_area)
    print("iou: ",iou)
    return iou


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
    model_complexity=1,           
    min_detection_confidence=0.3,
    min_tracking_confidence=0.4
    )
mp_drawing = mp.solutions.drawing_utils

iou_results = {}

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
    
    # Process hands first
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    
    # Get hand boxes
    hand_boxes = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            hand_boxes.append([x_min, y_min, x_max, y_max])
    
    # Get steering wheel boxes and calculate IoU
    wheel_boxes = []
    result = results[0]
    if len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name.lower() == 'wheel':
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                wheel_boxes.append([x1, y1, x2, y2])
    
    # Calculate max IoU between hands and steering wheel
    max_iou = 0.0
    print("len hand_boxes : ",hand_boxes)
    print("len wheel_boxes : ",wheel_boxes)
    if wheel_boxes and hand_boxes:
        for wheel_box in wheel_boxes:
            for hand_box in hand_boxes:
                current_iou = calculate_iou_hand(wheel_box, hand_box) 
                if current_iou > max_iou:
                    max_iou = current_iou
    
    iou_results[image_file] = max_iou
    
    annotated_frame = frame.copy()
    
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Change label for steering wheel based on IoU
        if class_name.lower() == 'wheel':
            if max_iou > 0.4:
                label = "Hands_on_wheel"
                color = (0, 255, 0)  # Green
            else:
                label = "Hands_off_wheel"
                color = (0, 0, 255)  # Red
        else:
            label = class_name
            color = (255, 0, 0)  
            
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    for hand_box in hand_boxes:
        x_min, y_min, x_max, y_max = hand_box
        cv2.rectangle(annotated_frame, 
                     (int(x_min), int(y_min)), 
                     (int(x_max), int(y_max)), 
                     (0, 255, 255), 2)  
        cv2.putText(annotated_frame, 'Hand', (int(x_min), int(y_min) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, annotated_frame)

print("\nIoU between hand and steering wheel for each image:")
for image_name, iou in iou_results.items():
    print(f"{image_name}: {iou:.4f}")

print(f"\nInference complete! Annotated images saved to: {OUTPUT_DIR}")
print("\nClass mapping:")
for i, name in model.names.items():
    print(f"{i}: {name}")