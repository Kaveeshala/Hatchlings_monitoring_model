import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIGURATION ---
SOURCE_VIDEO = "test_videos/IMG_8091.MOV" 
YOLO_MODEL = "models/best.pt"

# --- CALIBRATION ---
PIXELS_PER_CM = 25.0
SPEED_THRESHOLD = 2.0    # < 2.0 cm/s = FLOATER
WALL_MARGIN = 50

# --- NEW RULES FOR SMALL TANKS ---
# A healthy turtle swims straight at least 20% of the time (between turns)
STRAIGHTNESS_REQUIRED = 0.20  
# What counts as "Straight"? (An angle change of less than 20 degrees)
MAX_STRAIGHT_ANGLE = 20.0     

# --- HELPER: CALCULATE ANGLE ---
def get_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    dot = np.dot(v1, v2)
    # Clip to avoid numerical errors outside -1 to 1
    angle = np.degrees(np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0)))
    return 180 - angle 

print(f"Loading Model: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)
track_history = defaultdict(lambda: [])
cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. TRACKING
    results = model.track(frame, persist=True, verbose=False, conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center_x, center_y = float(x), float(y)
            
            # 2. RECORD HISTORY
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 60: track.pop(0)

            # 3. ANALYSIS
            status = "Analyzing..."
            color = (200, 200, 200) # Grey
            speed_val = 0
            straight_pct = 0

            if len(track) > 30:
                # --- A. SPEED CHECK ---
                dist_pixels = 0
                for i in range(1, len(track)):
                    dist_pixels += np.linalg.norm(np.array(track[i]) - np.array(track[i-1]))
                speed_val = (dist_pixels / PIXELS_PER_CM) / (len(track) / fps)

                # --- B. STRAIGHTNESS CHECK (The Fix!) ---
                # Instead of average turning, we count "Straight Segments"
                straight_frames = 0
                total_frames = 0
                
                # Check angle at every single step in history
                for i in range(2, len(track)): 
                    a = get_angle(track[i-2], track[i-1], track[i])
                    total_frames += 1
                    # If angle is small, it counts as "Swimming Straight"
                    if a < MAX_STRAIGHT_ANGLE:
                        straight_frames += 1
                
                # Calculate Percentage (0.0 to 1.0)
                if total_frames > 0:
                    straight_pct = straight_frames / total_frames

                # --- C. FINAL DECISION ---
                
                # 1. FLOATER CHECK
                if speed_val < SPEED_THRESHOLD:
                    status = "FLOATER (Sick)"
                    color = (0, 0, 255) # Red
                
                # 2. CIRCLER CHECK
                # Only sick if it NEVER swims straight (Straight % is very low)
                # AND it is not near the wall (Wall Logic)
                elif straight_pct < STRAIGHTNESS_REQUIRED:
                    # Double check: Is it in the center?
                    in_center = (WALL_MARGIN < center_x < width - WALL_MARGIN) and \
                                (WALL_MARGIN < center_y < height - WALL_MARGIN)
                    
                    if in_center:
                        status = "CIRCLER (Sick)"
                        color = (0, 165, 255) # Orange
                    else:
                        status = "NORMAL (Turning)"
                        color = (0, 255, 0)
                
                else:
                    # If it has enough straight segments, it's just swimming laps
                    status = "NORMAL (Active)"
                    color = (0, 255, 0) # Green

            # 4. DRAW UI
            x1, y1 = int(x - w/2), int(y - h/2)
            cv2.rectangle(frame, (x1, y1), (int(x+w/2), int(y+h/2)), color, 2)
            
            # Show "Str%" (Straight Percentage) so  can see logic working
            label = f"{status}"
            stats = f"Spd:{speed_val:.1f} | Str%:{straight_pct:.2f}"
            
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, stats, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow("High Accuracy Behavior Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()