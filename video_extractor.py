import cv2  # This is the OpenCV library for computer vision
import os   # This is for creating folders and managing file paths

# --- Configuration ---

# 1. Set the path to your folder containing the 32 training videos
INPUT_VIDEO_FOLDER = "train_videos"

# 2. Set the path to the folder where you want to save the frames
OUTPUT_FRAMES_FOLDER = "extracted_frames"

# 3. Set how often you want to save a frame.
# We will save one frame every 30 frames.
# (If your video is 30fps, this is about 1 frame per second. This is a good number.)
FRAME_INTERVAL = 30

# --- End of Configuration ---


def extract_frames():
    print(f"Starting frame extraction...")
    print(f"Input folder: {INPUT_VIDEO_FOLDER}")
    print(f"Output folder: {OUTPUT_FRAMES_FOLDER}")

    # Create the output folder if it doesn't already exist
    os.makedirs(OUTPUT_FRAMES_FOLDER, exist_ok=True)

    # Get a list of all files in your video folder
    video_files = os.listdir(INPUT_VIDEO_FOLDER)

    # Loop through every video file in that folder
    for video_filename in video_files:
        # We only want to process actual video files (e.g., .mp4, .mov)
        # This check is basic; it assumes all files in the folder are videos.
        # A more robust check would be: if video_filename.endswith(".mp4"):
        
        # Create the full path to the video
        video_path = os.path.join(INPUT_VIDEO_FOLDER, video_filename)
        
        print(f"\nProcessing video: {video_filename}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_filename}")
            continue  # Skip to the next video

        # A counter to keep track of which frame we are on
        frame_count = 0

        while True:
            # Read one frame from the video
            # 'ret' is a boolean (True/False) on if a frame was read
            # 'frame' is the actual image data
            ret, frame = cap.read()

            # If 'ret' is False, it means we are at the end of the video
            if not ret:
                break  # Exit the loop for this video

            # This is the magic line!
            # We use the "modulo" operator (%) to check if the frame_count
            # is perfectly divisible by our FRAME_INTERVAL.
            if frame_count % FRAME_INTERVAL == 0:
                # We want to save this frame!

                # Create a unique filename for the frame, e.g.:
                # "video_01.mp4_frame_30.jpg"
                # "video_01.mp4_frame_60.jpg"
                output_filename = f"{video_filename}_frame_{frame_count}.jpg"
                
                # Create the full path to save the image
                save_path = os.path.join(OUTPUT_FRAMES_FOLDER, output_filename)

                # Save the frame as a .jpg file
                cv2.imwrite(save_path, frame)
                
                # Print a progress message to the terminal
                print(f"  Saved: {save_path}")

            # Move to the next frame
            frame_count += 1

        # When the video is finished, release the video capture
        cap.release()
        print(f"Finished processing {video_filename}")

    print("\n--- All videos processed! ---")
    print(f"Your extracted frames are in the '{OUTPUT_FRAMES_FOLDER}' folder.")
    print("You can now upload this folder to Roboflow.")


# --- This line makes the script run when you call it from the terminal ---
if __name__ == "__main__":
    extract_frames()