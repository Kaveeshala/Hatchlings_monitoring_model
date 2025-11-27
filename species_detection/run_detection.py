from ultralytics import YOLO

# 1. Load your trained model
model = YOLO("models/best.pt")

# 2. Define which video to test
source_video = "test_videos/IMG_8091.MOV"

# 3. Run the detection
print(f"Processing {source_video}...")

results = model.predict(
    source=source_video, 
    show=True,      # This tries to pop up a window to show you live
    save=True,      # This saves the resulting video
    conf=0.5        # Only show boxes if model is 50% sure
)

print("Done! Check the 'runs/detect' folder for your video.")