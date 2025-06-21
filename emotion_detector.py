# emotion_detector.py
# This module provides functionality to detect emotions from video frames using the FER library.
from fer import FER

# Initialize once to avoid reloading the model every time
emotion_detector = FER(mtcnn=True)

def detect_emotion_from_frame(frame):
    analysis = emotion_detector.detect_emotions(frame)
    if analysis:
        emotions = analysis[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        return dominant, emotions
    else:
        return None, None