import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# --- CONFIG ---
SAMPLE_RATE = 16000
DURATION = 5
FILENAME = "output.wav"

pain_to_poses = {
    "back": ["Bridge", "Cat", "Cow", "Child's Pose", "Wheel"],
    "lower back": ["Bridge", "Child's Pose", "Cat", "Cow"],
    "upper back": ["Cat", "Cow", "Child's Pose"],
    "neck": ["Child's Pose", "Cat", "Cow"],
    "shoulder": ["Downward-Facing Dog", "Upward-Facing Dog", "Plank"],
    "shoulders": ["Downward-Facing Dog", "Upward-Facing Dog", "Plank"],
    "legs": ["Tree", "Warrior One", "Warrior Two", "Half-Moon", "Extended Side Angle"],
    "leg": ["Tree", "Warrior One", "Warrior Two", "Half-Moon", "Extended Side Angle"],
    "knee": ["Child's Pose", "Bridge"],
    "core": ["Boat", "Half-Boat", "Plank"],
    "spine": ["Bow", "Bridge", "Wheel"],
    "hip": ["Pigeon", "Bridge", "Child's Pose"],
    "hips": ["Pigeon", "Bridge", "Child's Pose"],
    "stress": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
    "anxiety": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
    "fatigue": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
    "tired": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
    "energy": ["Wheel", "Bow", "Upward-Facing Dog"],
    "relax": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
    "relaxation": ["Child's Pose", "Sphinx", "Seated Forward Bend"],
}
default_poses = ["Mountain Pose", "Tree Pose", "Corpse Pose", "Butterfly Pose", "Legs Up the Wall"]

# Load your model once
@st.cache_resource
def load_model():
    return joblib.load("tree_pose_classifier.pkl")
clf = load_model()

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

ANGLE_NAMES = [
    'left_knee', 'right_knee',
    'left_hip', 'right_hip',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow'
]
ANGLE_LANDMARKS = {
    'left_knee': [23, 25, 27],
    'right_knee': [24, 26, 28],
    'left_hip': [11, 23, 25],
    'right_hip': [12, 24, 26],
    'left_shoulder': [13, 11, 23],
    'right_shoulder': [14, 12, 24],
    'left_elbow': [15, 13, 11],
    'right_elbow': [16, 14, 12],
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_coords(landmarks, idx, width, height):
    lm = landmarks[idx]
    return (int(lm.x * width), int(lm.y * height))

def extract_angles(landmarks, width, height):
    angles = {}
    for name, idxs in ANGLE_LANDMARKS.items():
        a = get_coords(landmarks, idxs[0], width, height)
        b = get_coords(landmarks, idxs[1], width, height)
        c = get_coords(landmarks, idxs[2], width, height)
        angles[name] = calculate_angle(a, b, c)
    return angles

def record_audio(duration=DURATION, filename=FILENAME):
    try:
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        write(filename, SAMPLE_RATE, recording)
        return True
    except Exception as e:
        st.error(f"Recording error: {e}")
        return False

def transcribe_audio(filename=FILENAME):
    recognizer = sr.Recognizer()
    try:
        if not os.path.exists(filename):
            st.error("Audio file not found")
            return None
        with sr.AudioFile(filename) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please speak more clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def identify_pain_area_and_poses(text: str) -> tuple:
    text = text.lower()
    for pain_area in pain_to_poses:
        if pain_area in text:
            return pain_area, pain_to_poses[pain_area].copy()
    return "general", default_poses.copy()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Yoga Pose Suggester", page_icon="🧘‍♀️", layout="centered")

st.markdown("""
    <style>
        .main-header {
            font-size: 3.5rem;
            color: #ffd700;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
        }
        .pose-container {
            background-color: #2c3e50;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #ffd700;
            text-align: center;
        }
        .pain-area-header {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            color: #ffd700;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .pose-img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #ffd700;
            margin-bottom: 10px;
        }
        .reorder-btn {
            font-size: 1.5em;
            margin: 0 5px;
            background: #ffd700;
            color: #233142;
            border: none;
            border-radius: 5px;
            padding: 0.2em 0.7em;
            cursor: pointer;
        }
        .reorder-btn:disabled {
            background: #ccc;
            color: #888;
            cursor: not-allowed;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧘‍♀️ Yoga Pose Suggester</h1>', unsafe_allow_html=True)
st.markdown("### Click the button below and describe your pain area")
st.markdown("*(e.g., 'My lower back hurts' or 'I have shoulder pain')*")

if 'poses' not in st.session_state:
    st.session_state.poses = []
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'pain_area' not in st.session_state:
    st.session_state.pain_area = ""

if st.button("🎤 Start Recording (5 seconds)", type="primary"):
    with st.spinner("Recording... Please speak now!"):
        success = record_audio()
    if success:
        with st.spinner("Processing your audio..."):
            text = transcribe_audio()
            if text:
                st.session_state.transcribed_text = text
                pain_area, poses = identify_pain_area_and_poses(text)
                st.session_state.pain_area = pain_area
                st.session_state.poses = poses
                st.success("Audio processed successfully!")
            else:
                st.error("Sorry, could not understand your speech. Please try again.")
    else:
        st.error("Failed to record audio. Please check your microphone.")

if st.session_state.transcribed_text:
    st.markdown(f"### 📝 You said: *'{st.session_state.transcribed_text}'*")
    if st.session_state.pain_area != "general":
        st.markdown(f'<div class="pain-area-header">🎯 Detected Pain Area: {st.session_state.pain_area.title()}</div>', unsafe_allow_html=True)
        st.markdown(f"### 🧘‍♀️ Your Complete {st.session_state.pain_area.title()} Workout Routine:")
    else:
        st.markdown('<div class="pain-area-header">🧘‍♀️ General Yoga Routine</div>', unsafe_allow_html=True)
        st.markdown("### 🧘‍♀️ Your General Yoga Routine:")

    st.markdown("#### 🔄 Customize Your Routine Order:")
    st.markdown("*Use the buttons below to reorder poses*")

    for i, pose in enumerate(st.session_state.poses):
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.image("static/images/test.jpg", width=150, caption="")
        with col2:
            st.markdown(f'<div class="pose-container"><h4>{i + 1}. {pose}</h4></div>', unsafe_allow_html=True)
        with col3:
            up_disabled = i == 0
            down_disabled = i == len(st.session_state.poses) - 1
            up = st.button("⬆️", key=f"up_{i}", disabled=up_disabled)
            down = st.button("⬇️", key=f"down_{i}", disabled=down_disabled)
            if up:
                st.session_state.poses[i], st.session_state.poses[i-1] = st.session_state.poses[i-1], st.session_state.poses[i]
                st.rerun()
            if down:
                st.session_state.poses[i], st.session_state.poses[i+1] = st.session_state.poses[i+1], st.session_state.poses[i]
                st.rerun()

    if st.button("🔄 Reset to Original Order", type="secondary"):
        if st.session_state.pain_area != "general":
            st.session_state.poses = pain_to_poses[st.session_state.pain_area].copy()
        else:
            st.session_state.poses = default_poses.copy()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Your Workout Summary:")
    workout_text = " → ".join([f"{i+1}. {pose}" for i, pose in enumerate(st.session_state.poses)])
    st.markdown(f"**Routine:** {workout_text}")

    if st.button("📋 Copy Routine to Clipboard"):
        routine_list = "\n".join([f"{i+1}. {pose}" for i, pose in enumerate(st.session_state.poses)])
        st.code(routine_list, language="text")
        st.success("Routine displayed above - you can copy it manually!")

st.markdown("---")
st.markdown("### 📋 Instructions:")
st.markdown("""
1. **Record:** Click the recording button and describe your pain area
2. **Review:** See all poses for your specific pain area
3. **Customize:** Use ⬆️ and ⬇️ buttons to reorder poses
4. **Practice:** Follow your personalized routine
5. **Reset:** Use the reset button to return to original order
""")

st.markdown("### 🎯 Supported Pain Areas:")
pain_areas = list(pain_to_poses.keys())
st.markdown(", ".join([area.title() for area in pain_areas]))

# --- Start Workout Button ---
if st.button("▶️ Start Workout"):
    st.info("Starting webcam. Press 'q' in the webcam window to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(img_rgb)
            correct = False
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width = frame.shape[:2]
                angles = extract_angles(landmarks, width, height)
                features = np.array([angles[name] for name in ANGLE_NAMES]).reshape(1, -1)
                pred = clf.predict(features)[0]
                correct = bool(pred == 1)
                color = (0, 255, 0) if correct else (0, 0, 255)
                # Draw pose landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2)
                )
            cv2.imshow("Yoga Pose Accuracy", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success("Workout session ended.")