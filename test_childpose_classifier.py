import cv2
import numpy as np
import joblib
import mediapipe as mp

# Load the trained classifier
model = joblib.load("models/child_pose_classifier.pkl")

# MediaPipe pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints_from_frame(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints), results.pose_landmarks
    return None, None

# Start webcam feed
cap = cv2.VideoCapture(0)

print("🎥 Starting webcam. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, landmarks = extract_keypoints_from_frame(frame)

    if keypoints is not None:
        prediction = model.predict([keypoints])[0]
        confidence = model.predict_proba([keypoints])[0][prediction]

        label = "Child's Pose" if prediction == 1 else "Not Child's Pose"
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

        # Display result on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('🧘 Live Child Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
