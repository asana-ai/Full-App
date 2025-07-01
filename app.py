# app.py

import cv2
import numpy as np
import mediapipe as mp
import joblib
import base64
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import time
import threading
import random
import tempfile
import os
from gtts import gTTS
import pygame
import json

from pose import record_audio, transcribe_audio, identify_pain_area
from emotion_detector import detect_emotion_from_frame

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# --- Voice Feedback System ---
class VoiceFeedbackSystem:
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_cooldown = 3
        self.last_pose_state = None
        self.correct_pose_start = None
        self.audio_enabled = True
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
        except pygame.error as e:
            print(f"Warning: Could not initialize audio system: {e}")
            self.audio_enabled = False
        
        # Pose-specific feedback messages
        self.pose_specific_feedback = {
            'Tree': {
                'encouragement': [
                    "Great balance! Keep your tree pose steady!",
                    "Perfect! Stay rooted like a tree!",
                    "Excellent focus! Hold that balance!",
                    "Beautiful tree pose! Keep breathing!"
                ],
                'corrections': {
                    'left_knee': "Keep your standing leg straight",
                    'right_knee': "Lift your bent knee higher",
                    'left_hip': "Keep your standing hip straight",
                    'right_hip': "Open your lifted hip more",
                    'left_shoulder': "Relax your shoulders",
                    'right_shoulder': "Keep your shoulders level"
                },
                'general': [
                    "Focus on your balance point",
                    "Press your foot firmly into your leg",
                    "Find a spot to focus on ahead"
                ]
            },
            'Bridge': {
                'encouragement': [
                    "Strong bridge! Keep lifting those hips!",
                    "Perfect! Feel the strength in your legs!",
                    "Great bridge pose! Keep breathing deeply!",
                    "Excellent! Your spine is getting a wonderful stretch!"
                ],
                'corrections': {
                    'left_knee': "Keep your left knee bent at 90 degrees",
                    'right_knee': "Keep your right knee bent at 90 degrees", 
                    'left_hip': "Lift your hips higher",
                    'right_hip': "Push your hips up evenly",
                    'left_shoulder': "Keep your arms straight on the ground",
                    'right_shoulder': "Press your arms down for support"
                },
                'general': [
                    "Squeeze your glutes and lift higher",
                    "Keep your knees parallel",
                    "Press your feet firmly into the ground"
                ]
            },
            'Cat': {
                'encouragement': [
                    "Perfect cat stretch! Feel that spine curve!",
                    "Great! Round your back like an angry cat!",
                    "Excellent cat pose! Keep breathing!",
                    "Beautiful spine movement! Hold that curve!"
                ],
                'corrections': {
                    'left_knee': "Keep your knee under your hip",
                    'right_knee': "Keep your knee under your hip",
                    'left_hip': "Keep your hips level",
                    'right_hip': "Keep your hips level", 
                    'left_shoulder': "Keep your hand under your shoulder",
                    'right_shoulder': "Keep your hand under your shoulder"
                },
                'general': [
                    "Round your spine more toward the ceiling",
                    "Tuck your chin to your chest",
                    "Really arch your back upward"
                ]
            },
            'Cow': {
                'encouragement': [
                    "Perfect cow stretch! Feel that back arch!",
                    "Great! Lift your heart and tailbone!",
                    "Excellent cow pose! Keep breathing!",
                    "Beautiful counter-stretch! Hold that arch!"
                ],
                'corrections': {
                    'left_knee': "Keep your knee under your hip",
                    'right_knee': "Keep your knee under your hip",
                    'left_hip': "Keep your hips level",
                    'right_hip': "Keep your hips level",
                    'left_shoulder': "Keep your hand under your shoulder", 
                    'right_shoulder': "Keep your hand under your shoulder"
                },
                'general': [
                    "Arch your back and look up gently",
                    "Lift your chest and tailbone",
                    "Create a gentle curve in your spine"
                ]
            }
        }
        
        self.general_corrections = [
            "Let's adjust your pose and try again",
            "Focus on your alignment and try again", 
            "Take a breath and reset your position",
            "Almost there! Make small adjustments"
        ]
        
        self.return_messages = [
            "Great! You're back in position!",
            "Perfect! Let's continue!",
            "Nice correction! Keep going!"
        ]

    def generate_and_play_speech(self, text):
        if not self.audio_enabled:
            print(f"Audio disabled. Would say: '{text}'")
            return
        def play_audio():
            temp_file = None
            try:
                tts = gTTS(text=text, lang='en', tld='us', slow=False)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts.save(temp_file.name)
                temp_file.close()
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error with voice feedback: {e}")
            finally:
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
        threading.Thread(target=play_audio, daemon=True).start()

    def should_give_feedback(self):
        current_time = time.time()
        return current_time - self.last_feedback_time > self.feedback_cooldown

    def give_feedback(self, is_correct, pose_name, incorrect_angles=None):
        if not self.should_give_feedback():
            return
        current_time = time.time()
        message = None
        
        # Get pose-specific feedback
        pose_feedback = self.pose_specific_feedback.get(pose_name, self.pose_specific_feedback['Tree'])
        
        if is_correct:
            if self.last_pose_state != 'correct':
                if self.last_pose_state == 'incorrect':
                    message = random.choice(self.return_messages)
                elif self.last_pose_state is None:
                    message = f"Great! You're in the correct {pose_name} pose!"
                self.correct_pose_start = current_time
            elif self.correct_pose_start and (current_time - self.correct_pose_start) > 10 and (current_time - self.correct_pose_start) % 10 < 3:
                message = random.choice(pose_feedback['encouragement'])
        else:
            if self.last_pose_state == 'correct' or self.last_pose_state is None:
                if incorrect_angles and len(incorrect_angles) <= 2:
                    angle_name = list(incorrect_angles.keys())[0]
                    message = pose_feedback['corrections'].get(angle_name, random.choice(pose_feedback['general']))
                else:
                    message = random.choice(pose_feedback['general'])
        
        if message:
            self.generate_and_play_speech(message)
            self.last_feedback_time = current_time
        self.last_pose_state = 'correct' if is_correct else 'incorrect'

voice_feedback = VoiceFeedbackSystem()

# --- Pose Accuracy Model Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Load Poses Data ---
with open('poses.json', 'r') as f:
    all_poses_data = json.load(f)['poses']
poses_lookup = {pose['name']: pose for pose in all_poses_data}

# --- Model Cache ---
model_cache = {}

def load_pose_model(pose_name):
    """Load the appropriate model for the given pose name from models folder"""
    # Convert pose name to filename format
    model_filename = f"{pose_name.lower().replace(' ', '_').replace("'", '')}_classifier.pkl"
    model_path = os.path.join('models', model_filename)
    
    # Check if model is already cached
    if model_path not in model_cache:
        try:
            if os.path.exists(model_path):
                model_cache[model_path] = joblib.load(model_path)
                print(f"Loaded model: {model_path}")
            else:
                # Fallback to tree pose model
                fallback_path = os.path.join('models', 'tree_pose_classifier.pkl')
                model_cache[model_path] = joblib.load(fallback_path)
                print(f"Model {model_path} not found. Using tree pose model as fallback.")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            # Final fallback to tree pose model
            fallback_path = os.path.join('models', 'tree_pose_classifier.pkl')
            model_cache[model_path] = joblib.load(fallback_path)
            print(f"Fallback: Using tree pose model for {pose_name}")
    
    return model_cache[model_path]

def get_pose_thresholds(pose_name):
    """Get angle thresholds for specific poses based on CSV analysis"""
    pose_thresholds = {
        'Tree': {
            'left_knee': (175, 185),    # Standing leg very straight
            'right_knee': (45, 55),     # Bent leg at about 50 degrees
            'left_hip': (175, 185),     # Standing hip straight  
            'right_hip': (105, 115),    # Bent leg hip angle
            'left_shoulder': (165, 185), # Shoulders level
            'right_shoulder': (165, 185),
            'left_elbow': (155, 170),   # Arms in prayer or at sides
            'right_elbow': (155, 170),
        },
        'Bridge': {
            'left_knee': (75, 85),      # Knees bent at ~80 degrees
            'right_knee': (75, 85),     
            'left_hip': (165, 180),     # Hips lifted high
            'right_hip': (165, 180),    
            'left_shoulder': (165, 185), # Arms on ground
            'right_shoulder': (165, 185),
            'left_elbow': (165, 185),   # Arms straight
            'right_elbow': (165, 185),
        },
        'Cat': {
            'left_knee': (85, 95),      # Knees under hips
            'right_knee': (85, 95),
            'left_hip': (95, 105),      # Hips over knees
            'right_hip': (95, 105),
            'left_shoulder': (160, 175), # Hands under shoulders
            'right_shoulder': (160, 175),
            'left_elbow': (170, 185),   # Arms straight down
            'right_elbow': (170, 185),
        },
        'Cow': {
            'left_knee': (85, 95),      # Same base as cat
            'right_knee': (85, 95),
            'left_hip': (95, 105),      # Hips over knees
            'right_hip': (95, 105), 
            'left_shoulder': (160, 175), # Hands under shoulders
            'right_shoulder': (160, 175),
            'left_elbow': (165, 185),   # Arms supporting
            'right_elbow': (165, 185),
        },
        'Child\'s Pose': {
            'left_knee': (30, 60),      # Knees deeply bent
            'right_knee': (30, 60),
            'left_hip': (30, 60),       # Sitting back on heels
            'right_hip': (30, 60),
            'left_shoulder': (140, 185), # Arms extended forward
            'right_shoulder': (140, 185),
            'left_elbow': (160, 185),
            'right_elbow': (160, 185),
        },
        'Wheel': {
            'left_knee': (140, 170),    # Legs somewhat straight
            'right_knee': (140, 170),
            'left_hip': (140, 180),     # Hips very extended
            'right_hip': (140, 180),
            'left_shoulder': (100, 140), # Arms overhead
            'right_shoulder': (100, 140),
            'left_elbow': (140, 185),
            'right_elbow': (140, 185),
        }
    }
    
    # Return pose-specific thresholds or default to Tree pose
    return pose_thresholds.get(pose_name, pose_thresholds['Tree'])

# --- Update poses.json with correct model paths ---
def update_poses_json_with_model_paths():
    """Update poses.json to use correct model paths from models folder"""
    try:
        with open('poses.json', 'r') as f:
            data = json.load(f)
        
        for pose in data['poses']:
            pose_name = pose['name']
            model_filename = f"{pose_name.lower().replace(' ', '_').replace("'", '')}_classifier.pkl"
            model_path = os.path.join('models', model_filename)
            
            # Check if specific model exists, otherwise use tree pose
            if os.path.exists(model_path):
                pose['model_path'] = model_path
            else:
                pose['model_path'] = os.path.join('models', 'tree_pose_classifier.pkl')
        
        # Save updated JSON
        with open('poses.json', 'w') as f:
            json.dump(data, f, indent=2)
            
        print("Updated poses.json with correct model paths")
        
    except Exception as e:
        print(f"Error updating poses.json: {e}")

# Call this on startup to ensure correct model paths
update_poses_json_with_model_paths()

# --- Emotion-based Adaptive Logic ---
def get_adaptive_pose_recommendation(current_emotion, current_pose_name, pose_sequence, current_index):
    """
    Recommend next pose based on user's emotion and current pose difficulty
    """
    # Get current pose data
    current_pose = poses_lookup.get(current_pose_name, {'difficulty': 'Beginner'})
    current_difficulty = current_pose['difficulty']
    
    # Categorize emotions
    struggling_emotions = ['angry', 'sad', 'fear', 'disgust']
    confident_emotions = ['happy', 'surprise']
    neutral_emotions = ['neutral']
    
    # Easy poses for stress relief
    easy_poses = ['Child\'s Pose', 'Corpse Pose', 'Sphinx', 'Mountain Pose', 'Butterfly Pose']
    
    # Challenging poses
    challenging_poses = ['Wheel', 'Bow', 'Half-Moon', 'Extended Side Angle', 'Pigeon']
    
    # Get remaining poses in sequence
    remaining_poses = pose_sequence[current_index + 1:] if current_index + 1 < len(pose_sequence) else []
    
    recommendation_type = None
    message_emotion = current_emotion
    
    if current_emotion in struggling_emotions:
        if current_difficulty in ['Intermediate', 'Advanced']:
            # Struggling with hard pose - suggest easier
            recommendation_type = "easier"
            # Find easier pose from remaining sequence or fallback to easy poses
            for pose in remaining_poses:
                if poses_lookup.get(pose['name'], {'difficulty': 'Advanced'})['difficulty'] == 'Beginner':
                    return pose, recommendation_type, message_emotion
            # Fallback to easy poses
            for easy_pose_name in easy_poses:
                if easy_pose_name in poses_lookup:
                    return poses_lookup[easy_pose_name], recommendation_type, message_emotion
        else:
            # Struggling with easy pose - try rest pose
            recommendation_type = "easier"
            return poses_lookup.get('Child\'s Pose', remaining_poses[0] if remaining_poses else poses_lookup['Mountain Pose']), recommendation_type, message_emotion
    
    elif current_emotion in confident_emotions:
        # Happy/confident - suggest more challenging
        recommendation_type = "more challenging"
        # Find challenging pose from remaining sequence
        for pose in remaining_poses:
            if poses_lookup.get(pose['name'], {'difficulty': 'Beginner'})['difficulty'] in ['Intermediate', 'Advanced']:
                return pose, recommendation_type, message_emotion
        # Fallback to challenging poses
        for challenging_pose_name in challenging_poses:
            if challenging_pose_name in poses_lookup:
                return poses_lookup[challenging_pose_name], recommendation_type, message_emotion
    
    # Default: continue with next pose in sequence
    if remaining_poses:
        return remaining_poses[0], "next", message_emotion
    else:
        return None, "complete", message_emotion

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

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('voice.html')

@app.route('/get_poses', methods=['POST'])
def get_poses():
    try:
        record_audio()
        text = transcribe_audio()
        if text:
            result = identify_pain_area(text)
            return jsonify({
                'success': True,
                'text': text,
                'pain_area': result['pain_area'],
                'poses': result['poses']
            })
        return jsonify({'success': False, 'error': 'Could not understand audio'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_session', methods=['POST'])
def start_session():
    poses = request.json.get('poses', [])
    # Fallback if poses is empty or not a list of dicts
    if not poses or not isinstance(poses, list) or not isinstance(poses[0], dict):
        poses = [{'name': 'Tree', 'image': 'images/tree.png'}]
    session['pose_sequence'] = poses
    session['current_pose_index'] = 0
    return jsonify({'redirect': url_for('pose_accuracy')})

@app.route('/next_pose', methods=['POST'])
def next_pose():
    current_index = session.get('current_pose_index', 0)
    pose_sequence = session.get('pose_sequence', [])
    current_emotion = request.json.get('emotion', 'neutral') if request.is_json else 'neutral'
    
    if current_index >= len(pose_sequence):
        # Workout complete
        session.clear()
        return jsonify({
            'success': True,
            'workout_complete': True,
            'message': 'Great job! Your workout is complete!',
            'recommendation_type': 'complete'
        })
    
    # Get current pose name for adaptive recommendation
    current_pose_name = pose_sequence[current_index]['name']
    
    # Get adaptive recommendation based on emotion
    recommended_pose, recommendation_type, emotion = get_adaptive_pose_recommendation(
        current_emotion, current_pose_name, pose_sequence, current_index
    )
    
    if recommendation_type == "complete" or recommended_pose is None:
        # Workout complete
        session.clear()
        return jsonify({
            'success': True,
            'workout_complete': True,
            'message': 'Great job! Your workout is complete!',
            'recommendation_type': 'complete'
        })
    
    # Update session - if adaptive recommendation, modify the sequence
    if recommendation_type in ["easier", "more challenging"]:
        # Replace next pose with recommended pose
        if current_index + 1 < len(pose_sequence):
            pose_sequence[current_index + 1] = recommended_pose
        else:
            pose_sequence.append(recommended_pose)
        session['pose_sequence'] = pose_sequence
    
    # Move to next pose
    session['current_pose_index'] = current_index + 1
    
    pose_name = recommended_pose['name']
    pose_image = url_for('static', filename=recommended_pose['image'])
    
    return jsonify({
        'success': True,
        'workout_complete': False,
        'pose_name': pose_name,
        'pose_image': pose_image,
        'recommendation_type': recommendation_type,
        'emotion': emotion,
        'announcement': f"Adapting based on your emotion: {pose_name}"
    })

@app.route('/announce_next_pose', methods=['POST'])
def announce_next_pose():
    """Generate TTS announcement for next pose with countdown and emotion-based adaptation"""
    data = request.json
    pose_name = data.get('pose_name', 'next pose')
    current_emotion = data.get('emotion', 'neutral')
    recommendation_type = data.get('recommendation_type', 'next')
    
    # Stop any ongoing voice feedback
    voice_feedback.last_feedback_time = time.time() + 10  # Prevent feedback for 10 seconds
    
    def generate_announcement():
        try:
            # Emotion-based announcement
            if recommendation_type == "easier":
                announcement_text = f"You seem {current_emotion}. Here's an easier pose: {pose_name}."
            elif recommendation_type == "more challenging":
                announcement_text = f"You seem {current_emotion}. Here's a more challenging pose: {pose_name}."
            elif recommendation_type == "complete":
                announcement_text = "Great job! Your workout is complete!"
            else:
                announcement_text = f"Great job! Next pose is {pose_name}."
            
            tts = gTTS(text=announcement_text, lang='en', tld='us', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()
            
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            os.unlink(temp_file.name)
            
            # Only do countdown if not workout complete
            if recommendation_type != "complete":
                # Countdown
                for count in [5, 4, 3, 2, 1]:
                    countdown_text = str(count)
                    tts_count = gTTS(text=countdown_text, lang='en', tld='us', slow=False)
                    temp_file_count = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts_count.save(temp_file_count.name)
                    temp_file_count.close()
                    
                    pygame.mixer.music.load(temp_file_count.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    os.unlink(temp_file_count.name)
                    time.sleep(0.3)  # Brief pause between numbers
                
                # Final "Go!"
                go_text = "Go!"
                tts_go = gTTS(text=go_text, lang='en', tld='us', slow=False)
                temp_file_go = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts_go.save(temp_file_go.name)
                temp_file_go.close()
                
                pygame.mixer.music.load(temp_file_go.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                os.unlink(temp_file_go.name)
            
        except Exception as e:
            print(f"Error with announcement: {e}")
    
    # Run announcement in background thread
    threading.Thread(target=generate_announcement, daemon=True).start()
    
    return jsonify({'success': True})

@app.route('/pose')
def pose_accuracy():
    pose_sequence = session.get('pose_sequence', [])
    current_index = session.get('current_pose_index', 0)
    # Fallback if pose_sequence is empty or index out of range
    if not pose_sequence or current_index >= len(pose_sequence):
        pose = {'name': 'Tree', 'image': 'images/tree.png'}
    else:
        pose = pose_sequence[current_index]
    pose_name = pose['name']
    pose_image = url_for('static', filename=pose['image'])
    return render_template('pose.html', pose_name=pose_name, pose_image=pose_image)

@app.route('/predict', methods=['POST'])
def predict():
    detect_emotion = request.form.get('detect_emotion', 'false').lower() == 'true'
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Get current pose name from session
    pose_sequence = session.get('pose_sequence', [])
    current_index = session.get('current_pose_index', 0)
    current_pose_name = 'Tree'  # Default fallback
    
    if pose_sequence and current_index < len(pose_sequence):
        current_pose_name = pose_sequence[current_index]['name']
    
    # Load appropriate model and thresholds for current pose
    clf = load_pose_model(current_pose_name)
    ANGLE_THRESHOLDS = get_pose_thresholds(current_pose_name)

    # Pose detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    correct = False
    incorrect_angles = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width = frame.shape[:2]
        angles = extract_angles(landmarks, width, height)
        features = np.array([angles[name] for name in ANGLE_NAMES]).reshape(1, -1)
        pred = clf.predict(features)[0]
        correct = bool(pred == 1)
        for name, idxs in ANGLE_LANDMARKS.items():
            a = get_coords(landmarks, idxs[0], width, height)
            b = get_coords(landmarks, idxs[1], width, height)
            c = get_coords(landmarks, idxs[2], width, height)
            angle = angles[name]
            min_th, max_th = ANGLE_THRESHOLDS[name]
            is_angle_correct = min_th <= angle <= max_th
            if not is_angle_correct:
                incorrect_angles[name] = angle
            color = (0, 255, 0) if is_angle_correct else (0, 0, 255)
            cv2.line(frame, a, b, color, 4)
            cv2.line(frame, b, c, color, 4)
            cv2.putText(frame, f"{int(angle)}", b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Pass pose name to voice feedback
        voice_feedback.give_feedback(correct, current_pose_name, incorrect_angles if not correct else None)

    # Detect emotion ONLY if detect_emotion is True
    if detect_emotion:
        emotion, emotions = detect_emotion_from_frame(frame)
    else:
        emotion = "undetected"

    # Draw emotion on frame
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_b64,
        'correct': correct,
        'emotion': emotion
    })

if __name__ == '__main__':
    app.run(debug=True)