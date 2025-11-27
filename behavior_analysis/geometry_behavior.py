import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIGURATION ---
# 1. POINT TO YOUR VIDEO (Make sure filename is correct!)
SOURCE_VIDEO = "test_videos/IMG_8066.MOV" 

# 2. POINT TO YOUR MODEL
YOLO_MODEL = "models/best.pt"

# 3. SET THE RULES (The Math)
# If speed is less than 2.0, we say it's SICK (Floater)
SPEED_THRESHOLD = 2.0  
# If it turns too much (ratio > 3.0), we say it's SICK (Circler)
TORTUOSITY_THRESHOLD = 3.0
# How many pixels = 1 cm? (Guess 25 for now)
PIXELS_PER_CM = 25.0       

# --- LOAD EVERYTHING ---
print(f"Loading Model: {YOLO_MODEL}")
print(f"Loading Video: {SOURCE_VIDEO}")

model = YOLO(YOLO_MODEL)
track_history = defaultdict(lambda: [])
cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    print(" ERROR: Could not open video! Check the file name.")
    exit()

print("Video started! Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. TRACKING (Use YOLO to find the turtle)
    results = model.track(frame, persist=True, verbose=False, conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            
            # 2. RECORD HISTORY (Remember where it was)
            track = track_history[track_id]
            track.append((float(x), float(y)))
            
            # Keep only last 60 frames (2 seconds of history)
            if len(track) > 60: track.pop(0)

            # 3. DO THE MATH (Geometry Analysis)
            status = "Analyzing..."
            color = (200, 200, 200) # Grey
            speed_val = 0

            if len(track) > 30: # Wait until we have 1 second of data
                
                # A. CALCULATE SPEED
                # Measure distance between every point in history
                path_distance = 0
                for i in range(1, len(track)):
                    p1 = np.array(track[i-1])
                    p2 = np.array(track[i])
                    path_distance += np.linalg.norm(p2 - p1)
                
                # Convert pixels to cm/s
                path_distance_cm = path_distance / PIXELS_PER_CM
                duration_sec = len(track) / fps
                speed_val = path_distance_cm / duration_sec

                # B. CALCULATE TURNING (Tortuosity)
                start_point = np.array(track[0])
                end_point = np.array(track[-1])
                displacement = np.linalg.norm(end_point - start_point)
                
                if displacement < 5: 
                    tortuosity = 100 # It's stuck in one spot
                else:
                    tortuosity = path_distance / displacement

                # C. DECIDE HEALTH
                if speed_val < SPEED_THRESHOLD:
                    status = "FLOATER (Sick)"
                    color = (0, 0, 255) # Red
                elif tortuosity > TORTUOSITY_THRESHOLD:
                    status = "CIRCLER (Sick)"
                    color = (0, 165, 255) # Orange
                else:
                    status = "NORMAL (Active)"
                    color = (0, 255, 0) # Green

            # 4. DRAW THE RESULT
            x1, y1 = int(x - w/2), int(y - h/2)
            cv2.rectangle(frame, (x1, y1), (int(x+w/2), int(y+h/2)), color, 2)
            
            # Show Species + Status
            species_name = model.names[class_id]
            label = f"{species_name} | {status}"
            stats = f"Spd:{speed_val:.1f}"
            
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, stats, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw the path lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow("Geometry Behavior Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()