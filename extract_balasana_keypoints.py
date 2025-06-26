import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
INPUT_FOLDER = r"C:\Users\kadam\Desktop\Dataset\BALASANA"
OUTPUT_FOLDER = r"C:\Users\kadam\Desktop\Dataset\BALASANA_keypoints"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints)
    return None

# Loop through all images
for idx, file in enumerate(os.listdir(INPUT_FOLDER)):
    file_path = os.path.join(INPUT_FOLDER, file)
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        keypoints = extract_keypoints(file_path)
        if keypoints is not None:
            out_path = os.path.join(OUTPUT_FOLDER, f"balasana_{idx}.npy")
            np.save(out_path, keypoints)
            print(f"✅ Saved {out_path}")
        else:
            print(f"⚠️ No pose detected in {file}")
