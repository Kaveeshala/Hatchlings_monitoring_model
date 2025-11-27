import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
# Path to your model file (make sure this matches your folder structure!)
MODEL_PATH = "models/behavior_gru.h5"

# --- 1. LOAD THE MODEL ---
print(f"Loading model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print(" ERROR: Could not find the model file!")
    print("Make sure 'behavior_gru.h5' is inside the 'models' folder.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully!")

# --- 2. GENERATE FAKE TEST DATA (Just to test the brain) ---
# We use the same math logic as training to create 3 clear examples
TANK_SIZE = 640

def get_fake_normal_path():
    # A fast, straight diagonal swim
    path = []
    for i in range(30):
        # Moving 10 pixels per frame (Fast)
        path.append([i * 10, i * 10]) 
    return np.array(path)

def get_fake_floater_path():
    # Staying in roughly the same spot (100, 100)
    path = []
    for i in range(30):
        # Moving barely at all (0.5 pixels jitter)
        path.append([100 + np.random.uniform(-1, 1), 100 + np.random.uniform(-1, 1)])
    return np.array(path)

def get_fake_circler_path():
    # Moving in a perfect circle
    path = []
    center_x, center_y = 320, 240
    for i in range(30):
        angle = 0.2 * i
        x = center_x + 50 * np.cos(angle)
        y = center_y + 50 * np.sin(angle)
        path.append([x, y])
    return np.array(path)

# --- 3. RUN PREDICTIONS ---
# Create the 3 examples
samples = np.array([
    get_fake_normal_path(),
    get_fake_floater_path(),
    get_fake_circler_path()
])

# Normalize (Important! The model expects numbers between 0-1, not 0-640)
samples_normalized = samples / TANK_SIZE

print("\nRunning predictions on 3 test cases...")
predictions = model.predict(samples_normalized)

# --- 4. PRINT RESULTS ---
class_names = ["Normal (Wall Swimmer)", "Floater (Sick)", "Circler (Sick)"]

print("\n--- RESULTS ---")
for i, pred in enumerate(predictions):
    # 'pred' is a list of 3 probabilities, e.g. [0.01, 0.98, 0.01]
    # np.argmax finds the highest probability
    predicted_class = np.argmax(pred)
    confidence = np.max(pred) * 100
    
    print(f"Test Case {i+1}: Predicted -> {class_names[predicted_class]} ({confidence:.1f}%)")