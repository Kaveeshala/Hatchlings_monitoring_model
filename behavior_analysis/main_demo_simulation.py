import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "models/best.pt"
BEHAVIOR_MODEL_PATH = "models/behavior_gru.h5"
SOURCE_VIDEO = "test_videos/IMG_3142.MOV" 

# --- THRESHOLDS ---
# If speed is > 3 cm/s, we force "Normal" status to prevent false alarms

SPEED_THRESHOLD_NORMAL = 3.0 
PIXELS_PER_CM = 25.0 # Adjust this based on your camera zoom

# --- LOAD MODELS ---
print("Loading AI Models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
behavior_model = tf.keras.models.load_model(BEHAVIOR_MODEL_PATH)
print("✅ Models Loaded!")

# Classes: 0=Normal, 1=Floater, 2=Circler
behavior_classes = ["Normal", "Floater (Sick)", "Circler (Sick)"]
behavior_colors = [(0, 255, 0), (0, 0, 255), (0, 165, 255)] # Green, Red, Orange

# --- TRACKING VARIABLES ---
track_history = defaultdict(lambda: [])
speed_history = {} # To store previous position for speed calc

cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. DETECT & TRACK
    results = yolo_model.track(frame, persist=True, verbose=False, conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center_x, center_y = float(x), float(y)

            # --- 2. CALCULATE SPEED ---
            current_speed = 0.0
            if track_id in speed_history:
                prev_x, prev_y = speed_history[track_id]
                dist_px = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                dist_cm = dist_px / PIXELS_PER_CM
                current_speed = dist_cm * fps # cm per second
            
            speed_history[track_id] = (center_x, center_y)

            # --- 3. RECORD HISTORY ---
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 30: track.pop(0)

            # --- 4. DETERMINE BEHAVIOR (THE HYBRID LOGIC) ---
            status = "Analyzing..."
            color = (255, 255, 0) # Yellow

            if len(track) == 30:
                # LOGIC A: If moving fast, it is HEALTHY (Ignore AI)
                if current_speed > SPEED_THRESHOLD_NORMAL:
                    status = "Normal (Active)"
                    color = (0, 255, 0) # Green
                
                # LOGIC B: If slow, ask the AI (Floater vs Circler)
                else:
                    # Prepare data for GRU
                    input_data = np.array(track)
                    input_data = input_data / 640.0 # Normalize
                    input_data = input_data.reshape(1, 30, 2)

                    # AI Prediction
                    pred = behavior_model.predict(input_data, verbose=0)
                    behavior_idx = np.argmax(pred)
                    confidence = np.max(pred)

                    # If AI says "Normal" but it's slow, call it "Resting"
                    if behavior_idx == 0: 
                        status = f"Resting ({confidence:.0%})"
                        color = (0, 255, 0)
                    else:
                        status = f"{behavior_classes[behavior_idx]} ({confidence:.0%})"
                        color = behavior_colors[behavior_idx]

            # --- 5. DRAW UI ---
            # Box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Text Label
            label_text = f"ID:{track_id} | {status}"
            speed_text = f"Speed: {current_speed:.1f} cm/s"
            
            cv2.putText(frame, label_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, speed_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Draw Trail
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow("Hatchling Monitor - FINAL SYSTEM", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()