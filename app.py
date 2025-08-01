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
from datetime import datetime, timedelta

from pose import record_audio, transcribe_audio, identify_pain_area
from emotion_detector import detect_emotion_from_frame

# Import modular components
from modules.scoring import PoseScorer
from modules.routine_manager import RoutineManager
from modules.voice_feedback import VoiceFeedbackSystem

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# --- Pose Scoring System ---
class PoseScorer:
    def __init__(self):
        self.session_data = {}
        
    def start_pose_session(self, pose_name):
        """Initialize tracking for a new pose session"""
        session_id = f"{pose_name}_{int(time.time())}"
        self.session_data[session_id] = {
            'pose_name': pose_name,
            'start_time': time.time(),
            'total_frames': 0,
            'correct_frames': 0,
            'angle_accuracy_sum': {},
            'angle_frame_count': {},
            'hold_duration': 0,
            'stability_score': 0,
            'emotion_consistency': [],
            'feedback_given': 0,
            'improvement_trend': [],
            'difficulty_bonus': 0
        }
        return session_id
    
    def update_pose_session(self, session_id, is_correct, angles, target_thresholds, emotion="neutral"):
        """Update pose session with current frame data"""
        if session_id not in self.session_data:
            return
            
        data = self.session_data[session_id]
        data['total_frames'] += 1
        
        if is_correct:
            data['correct_frames'] += 1
            
        # Track angle accuracy for each joint
        for angle_name, angle_value in angles.items():
            if angle_name not in data['angle_accuracy_sum']:
                data['angle_accuracy_sum'][angle_name] = 0
                data['angle_frame_count'][angle_name] = 0
                
            # Calculate accuracy percentage for this angle
            min_thresh, max_thresh = target_thresholds.get(angle_name, (0, 180))
            target_angle = (min_thresh + max_thresh) / 2
            tolerance = (max_thresh - min_thresh) / 2
            
            # Calculate accuracy (100% if within range, decreasing as it gets further)
            if min_thresh <= angle_value <= max_thresh:
                accuracy = 100
            else:
                deviation = min(abs(angle_value - min_thresh), abs(angle_value - max_thresh))
                accuracy = max(0, 100 - (deviation / tolerance) * 50)  # 50% penalty for being outside range
                
            data['angle_accuracy_sum'][angle_name] += accuracy
            data['angle_frame_count'][angle_name] += 1
            
        # Track emotions for consistency
        data['emotion_consistency'].append(emotion)
        
        # Calculate improvement trend (last 10 measurements)
        current_accuracy = (data['correct_frames'] / data['total_frames']) * 100
        data['improvement_trend'].append(current_accuracy)
        if len(data['improvement_trend']) > 10:
            data['improvement_trend'].pop(0)
    
    def increment_feedback(self, session_id):
        """Track when feedback is given to user"""
        if session_id in self.session_data:
            self.session_data[session_id]['feedback_given'] += 1
    
    def end_pose_session(self, session_id):
        """Calculate final score and metrics for completed pose"""
        if session_id not in self.session_data:
            return None
            
        data = self.session_data[session_id]
        end_time = time.time()
        data['hold_duration'] = end_time - data['start_time']
        
        # Calculate comprehensive score
        score_breakdown = self._calculate_comprehensive_score(data)
        
        # Clean up session data
        del self.session_data[session_id]
        
        return score_breakdown
    
    def _calculate_comprehensive_score(self, data):
        """Calculate detailed scoring breakdown"""
        total_score = 0
        breakdown = {}
        
        # 1. Basic Accuracy Score (40 points max)
        if data['total_frames'] > 0:
            accuracy_percentage = (data['correct_frames'] / data['total_frames']) * 100
            accuracy_score = min(40, (accuracy_percentage / 100) * 40)
        else:
            accuracy_percentage = 0
            accuracy_score = 0
            
        breakdown['accuracy'] = {
            'score': round(accuracy_score, 1),
            'percentage': round(accuracy_percentage, 1),
            'description': self._get_accuracy_description(accuracy_percentage)
        }
        total_score += accuracy_score
        
        # 2. Joint Alignment Score (25 points max)
        joint_scores = {}
        joint_total = 0
        joint_count = 0
        
        for angle_name, accuracy_sum in data['angle_accuracy_sum'].items():
            if data['angle_frame_count'][angle_name] > 0:
                avg_accuracy = accuracy_sum / data['angle_frame_count'][angle_name]
                joint_scores[angle_name] = round(avg_accuracy, 1)
                joint_total += avg_accuracy
                joint_count += 1
                
        if joint_count > 0:
            joint_alignment_avg = joint_total / joint_count
            joint_alignment_score = (joint_alignment_avg / 100) * 25
        else:
            joint_alignment_avg = 0
            joint_alignment_score = 0
            
        breakdown['joint_alignment'] = {
            'score': round(joint_alignment_score, 1),
            'average': round(joint_alignment_avg, 1),
            'individual_joints': joint_scores,
            'description': self._get_joint_description(joint_alignment_avg)
        }
        total_score += joint_alignment_score
        
        # 3. Hold Duration Score (15 points max)
        duration_score, duration_rating = self._calculate_duration_score(data['hold_duration'])
        breakdown['hold_duration'] = {
            'score': duration_score,
            'seconds': round(data['hold_duration'], 1),
            'rating': duration_rating,
            'description': self._get_duration_description(data['hold_duration'])
        }
        total_score += duration_score
        
        # 4. Stability & Consistency Score (10 points max)
        stability_score = self._calculate_stability_score(data)
        breakdown['stability'] = {
            'score': stability_score,
            'description': self._get_stability_description(stability_score)
        }
        total_score += stability_score
        
        # 5. Improvement Trend Score (5 points max)
        improvement_score = self._calculate_improvement_score(data['improvement_trend'])
        breakdown['improvement'] = {
            'score': improvement_score,
            'description': self._get_improvement_description(improvement_score)
        }
        total_score += improvement_score
        
        # 6. Focus & Mindfulness Score (5 points max)
        focus_score = self._calculate_focus_score(data)
        breakdown['focus'] = {
            'score': focus_score,
            'description': self._get_focus_description(data['emotion_consistency'], data['feedback_given'])
        }
        total_score += focus_score
        
        # Calculate final grade and overall performance
        final_score = min(100, round(total_score, 1))
        grade = self._get_grade(final_score)
        performance_level = self._get_performance_level(final_score)
        
        return {
            'total_score': final_score,
            'grade': grade,
            'performance_level': performance_level,
            'breakdown': breakdown,
            'pose_name': data['pose_name'],
            'session_stats': {
                'total_frames': data['total_frames'],
                'correct_frames': data['correct_frames'],
                'duration': round(data['hold_duration'], 1),
                'feedback_given': data['feedback_given']
            },
            'recommendations': self._generate_recommendations(breakdown, final_score)
        }
    
    def _calculate_duration_score(self, duration):
        """Score based on how long pose was held"""
        if duration < 5:
            return 2, "Too Short"
        elif duration < 10:
            return 5, "Short"
        elif duration < 20:
            return 10, "Good"
        elif duration < 30:
            return 15, "Excellent"
        else:
            return 15, "Outstanding"
    
    def _calculate_stability_score(self, data):
        """Score based on consistency of correct poses"""
        if data['total_frames'] < 10:
            return 5  # Not enough data
            
        # Calculate how consistent the pose accuracy was
        frame_window = 5
        consistency_scores = []
        
        for i in range(frame_window, len(data['improvement_trend'])):
            recent_window = data['improvement_trend'][i-frame_window:i]
            window_std = np.std(recent_window) if len(recent_window) > 1 else 0
            consistency = max(0, 100 - window_std)  # Lower std deviation = higher consistency
            consistency_scores.append(consistency)
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            return (avg_consistency / 100) * 10
        return 5
    
    def _calculate_improvement_score(self, trend):
        """Score based on improvement during the session"""
        if len(trend) < 3:
            return 2.5  # Not enough data
            
        # Compare first third vs last third of session
        first_third = trend[:len(trend)//3] if len(trend) >= 3 else trend[:1]
        last_third = trend[-len(trend)//3:] if len(trend) >= 3 else trend[-1:]
        
        first_avg = np.mean(first_third)
        last_avg = np.mean(last_third)
        
        improvement = last_avg - first_avg
        
        if improvement > 10:
            return 5  # Significant improvement
        elif improvement > 5:
            return 4  # Good improvement
        elif improvement > 0:
            return 3  # Some improvement
        elif improvement > -5:
            return 2  # Maintained performance
        else:
            return 1  # Declined performance
    
    def _calculate_focus_score(self, data):
        """Score based on emotion consistency and learning"""
        emotions = data['emotion_consistency']
        feedback_count = data['feedback_given']
        
        # Emotion consistency score
        if emotions:
            positive_emotions = ['happy', 'neutral', 'surprise']
            positive_ratio = sum(1 for e in emotions if e in positive_emotions) / len(emotions)
            emotion_score = positive_ratio * 3
        else:
            emotion_score = 1.5
            
        # Learning efficiency (fewer corrections needed = better)
        total_frames = data['total_frames']
        if total_frames > 0:
            feedback_efficiency = max(0, 2 - (feedback_count / max(1, total_frames / 10)))
        else:
            feedback_efficiency = 1
            
        return min(5, emotion_score + feedback_efficiency)
    
    def _get_accuracy_description(self, percentage):
        if percentage >= 90:
            return "Excellent accuracy! Your poses were nearly perfect."
        elif percentage >= 75:
            return "Great accuracy! Minor adjustments will make you perfect."
        elif percentage >= 60:
            return "Good accuracy! Focus on holding positions longer."
        elif percentage >= 40:
            return "Fair accuracy. Practice the pose alignment more."
        else:
            return "Keep practicing! Focus on basic pose structure."
    
    def _get_joint_description(self, average):
        if average >= 90:
            return "Perfect joint alignment! Outstanding body positioning."
        elif average >= 75:
            return "Excellent alignment! Very precise joint positioning."
        elif average >= 60:
            return "Good alignment! Some joints need minor adjustments."
        else:
            return "Focus on proper joint positioning and form."
    
    def _get_duration_description(self, duration):
        if duration >= 30:
            return "Outstanding endurance! You held the pose beautifully."
        elif duration >= 20:
            return "Excellent hold time! Great strength and stability."
        elif duration >= 10:
            return "Good hold time! Try to extend it gradually."
        else:
            return "Practice holding poses longer for better benefits."
    
    def _get_stability_description(self, score):
        if score >= 8:
            return "Rock-steady stability! Excellent control."
        elif score >= 6:
            return "Good stability with minor fluctuations."
        elif score >= 4:
            return "Fair stability. Work on maintaining consistent form."
        else:
            return "Focus on building stability and core strength."
    
    def _get_improvement_description(self, score):
        if score >= 4:
            return "Amazing improvement during the session!"
        elif score >= 3:
            return "Good progress throughout the practice."
        elif score >= 2:
            return "Steady performance maintained."
        else:
            return "Keep practicing for better consistency."
    
    def _get_focus_description(self, emotions, feedback_count):
        positive_emotions = sum(1 for e in emotions if e in ['happy', 'neutral', 'surprise'])
        total_emotions = len(emotions) if emotions else 1
        
        if positive_emotions / total_emotions >= 0.8:
            return "Excellent mindfulness and emotional balance."
        elif positive_emotions / total_emotions >= 0.6:
            return "Good focus with positive energy."
        else:
            return "Try to stay relaxed and focused during practice."
    
    def _get_grade(self, score):
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        else:
            return "Keep Practicing!"
    
    def _get_performance_level(self, score):
        if score >= 90:
            return "Master Yogi"
        elif score >= 80:
            return "Advanced Practitioner"
        elif score >= 70:
            return "Intermediate"
        elif score >= 60:
            return "Developing"
        else:
            return "Beginner"
    
    def _generate_recommendations(self, breakdown, total_score):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Accuracy recommendations
        if breakdown['accuracy']['score'] < 30:
            recommendations.append("Focus on basic pose alignment - watch tutorial videos")
            recommendations.append("Practice poses slowly with a mirror for better form")
        
        # Joint alignment recommendations
        joint_scores = breakdown['joint_alignment']['individual_joints']
        weak_joints = [joint for joint, score in joint_scores.items() if score < 70]
        if weak_joints:
            recommendations.append(f"Work on these specific areas: {', '.join(weak_joints)}")
        
        # Duration recommendations
        if breakdown['hold_duration']['score'] < 10:
            recommendations.append("Gradually increase pose hold time - start with 15-30 seconds")
            recommendations.append("Build core strength to improve pose endurance")
        
        # Stability recommendations
        if breakdown['stability']['score'] < 6:
            recommendations.append("Practice balance exercises to improve stability")
            recommendations.append("Focus on slow, controlled movements")
        
        # Overall recommendations
        if total_score >= 90:
            recommendations.append("Outstanding! Try more advanced variations of this pose")
        elif total_score >= 70:
            recommendations.append("Great progress! Challenge yourself with longer holds")
        else:
            recommendations.append("Keep practicing regularly - consistency is key to improvement")
        
        return recommendations[:4]  # Limit to top 4 recommendations

# Initialize global scorer
pose_scorer = PoseScorer()

# --- Routine Management System ---
class RoutineManager:
    def __init__(self):
        self.original_routine = []
        self.current_routine_index = 0
        self.emotion_detour_poses = []
        self.detour_active = False
        self.detour_index = 0
        
    def start_routine(self, poses):
        """Initialize a new routine"""
        self.original_routine = poses.copy()
        self.current_routine_index = 0
        self.emotion_detour_poses = []
        self.detour_active = False
        self.detour_index = 0
        
    def get_current_pose(self):
        """Get the current pose based on routine state"""
        if self.detour_active and self.detour_index < len(self.emotion_detour_poses):
            return self.emotion_detour_poses[self.detour_index]
        elif self.current_routine_index < len(self.original_routine):
            return self.original_routine[self.current_routine_index]
        else:
            return None  # Routine complete
            
    def advance_pose(self):
        """Move to the next pose"""
        if self.detour_active:
            self.detour_index += 1
            # Check if detour is complete
            if self.detour_index >= len(self.emotion_detour_poses):
                self._end_detour()
        else:
            self.current_routine_index += 1
            
    def start_emotion_detour(self, emotion_poses):
        """Start an emotion-based detour"""
        self.detour_active = True
        self.emotion_detour_poses = emotion_poses
        self.detour_index = 0
        
    def _end_detour(self):
        """End emotion detour and return to original routine"""
        self.detour_active = False
        self.emotion_detour_poses = []
        self.detour_index = 0
        
    def is_routine_complete(self):
        """Check if the entire routine is complete"""
        return (not self.detour_active and 
                self.current_routine_index >= len(self.original_routine))
        
    def get_progress_info(self):
        """Get current progress information"""
        total_original = len(self.original_routine)
        completed_original = self.current_routine_index
        
        status = "detour" if self.detour_active else "routine"
        
        return {
            'total_poses': total_original,
            'completed_poses': completed_original,
            'current_pose_number': completed_original + 1,
            'status': status,
            'detour_progress': f"{self.detour_index + 1}/{len(self.emotion_detour_poses)}" if self.detour_active else None,
            'routine_complete': self.is_routine_complete(),
            'current_pose': self.get_current_pose()
        }

# Initialize global routine manager
routine_manager = RoutineManager()

# --- Voice Feedback System ---
class VoiceFeedbackSystem:
    def __init__(self):
        self.last_feedback_time = 0
        self.feedback_cooldown = 2  # Reduced from 3 to 2 seconds for more responsive feedback
        self.last_pose_state = None
        self.correct_pose_start = None
        self.audio_enabled = True
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            print("Voice feedback system initialized successfully")
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
        print(f"Voice feedback called: is_correct={is_correct}, pose={pose_name}, audio_enabled={self.audio_enabled}")
        if not self.should_give_feedback():
            print("Feedback skipped due to cooldown")
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
            elif self.correct_pose_start and (current_time - self.correct_pose_start) > 8:
                # Give encouragement every 8 seconds during correct poses
                message = random.choice(pose_feedback['encouragement'])
                self.correct_pose_start = current_time  # Reset timer for next encouragement
        else:
            if self.last_pose_state == 'correct' or self.last_pose_state is None:
                if incorrect_angles and len(incorrect_angles) <= 2:
                    angle_name = list(incorrect_angles.keys())[0]
                    message = pose_feedback['corrections'].get(angle_name, random.choice(pose_feedback['general']))
                else:
                    message = random.choice(pose_feedback['general'])
        
        if message:
            print(f"Playing voice message: '{message}'")
            self.generate_and_play_speech(message)
            self.last_feedback_time = current_time
        else:
            print("No message generated for voice feedback")
        self.last_pose_state = 'correct' if is_correct else 'incorrect'

voice_feedback = VoiceFeedbackSystem()

# --- Pose Accuracy Model Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # Use lighter model (0=lite, 1=full, 2=heavy)
)

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

def get_coords_scaled(landmarks, idx, width, height):
    """Get coordinates scaled to the specified dimensions"""
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
    
    # Initialize routine manager with the original poses
    routine_manager.start_routine(poses)
    
    # Store in session for backup
    session['original_pose_sequence'] = poses
    session['session_start_time'] = datetime.now().isoformat()
    
    return jsonify({'redirect': url_for('pose_accuracy')})

@app.route('/next_pose', methods=['POST'])
def next_pose():
    data = request.json
    current_emotion = data.get('emotion', 'neutral')
    
    # Check if routine is complete
    if routine_manager.is_routine_complete():
        # Give completion feedback
        voice_feedback.generate_and_play_speech("Congratulations! You have completed your entire yoga routine! Great job!")
        return jsonify({
            'success': True,
            'workout_complete': True,
            'message': 'Congratulations! You have completed your entire yoga routine!',
            'progress': routine_manager.get_progress_info()
        })
    
    # Get current pose to determine if we need emotion adaptation
    current_pose = routine_manager.get_current_pose()
    if not current_pose:
        voice_feedback.generate_and_play_speech("Workout completed! Well done!")
        return jsonify({
            'success': True,
            'workout_complete': True,
            'message': 'Workout completed!',
            'progress': routine_manager.get_progress_info()
        })
    
    # Give success feedback for completing the current pose
    current_pose_name = current_pose['name']
    success_messages = [
        f"Great job on {current_pose_name}! Moving to the next pose.",
        f"Excellent {current_pose_name}! Let's continue.",
        f"Perfect {current_pose_name}! Well done!",
        f"Beautiful {current_pose_name}! Keep up the great work!"
    ]
    import random
    success_message = random.choice(success_messages)
    voice_feedback.generate_and_play_speech(success_message)
    
    # Advance to next pose
    routine_manager.advance_pose()
    
    # Get the next pose
    next_pose_data = routine_manager.get_current_pose()
    
    # If no next pose, routine is complete
    if not next_pose_data:
        voice_feedback.generate_and_play_speech("Amazing! Your workout is complete!")
        return jsonify({
            'success': True,
            'workout_complete': True,
            'message': 'Great job! Your workout is complete!',
            'progress': routine_manager.get_progress_info()
        })
    
    # Check if we need emotion-based adaptation for the NEXT pose
    recommendation_type = "continue"
    emotion_message = ""
    
    # Only adapt if we're not already in a detour AND emotion suggests adaptation
    if not routine_manager.detour_active:
        if current_emotion in ['angry', 'sad', 'fear']:
            # User is struggling - insert rest poses before next pose
            rest_poses = [
                {'name': 'Child\'s Pose', 'image': 'images/childs_pose.webp'},
                {'name': 'Corpse Pose', 'image': 'images/corpse.jpg'}
            ]
            routine_manager.start_emotion_detour(rest_poses, "easier")
            next_pose_data = routine_manager.get_current_pose()
            recommendation_type = "easier"
            emotion_message = "I sense you might need a break. Let's try some gentle poses."
            
        elif current_emotion in ['happy', 'surprise']:
            # User is confident - insert challenging poses
            challenging_poses = [
                {'name': 'Wheel', 'image': 'images/wheel.webp'}
            ]
            routine_manager.start_emotion_detour(challenging_poses, "challenging")
            next_pose_data = routine_manager.get_current_pose()
            recommendation_type = "challenging"
            emotion_message = "You're doing great! Let's add a challenge."
    
    pose_name = next_pose_data['name']
    pose_image_path = get_pose_image_path(pose_name)
    pose_image = url_for('static', filename=pose_image_path)
    
    # Get updated progress
    progress = routine_manager.get_progress_info()
    
    # Create appropriate announcement based on whether we're in a detour
    if routine_manager.detour_active:
        # We're starting or continuing a detour
        if emotion_message:
            # Just started a detour - announce the emotion message first
            voice_feedback.generate_and_play_speech(f"{emotion_message} Now let's do {pose_name}.")
        else:
            # Continuing existing detour
            if progress['status'] == 'detour':
                detour_type = getattr(routine_manager, 'detour_type', 'special')
                if detour_type == "easier":
                    voice_feedback.generate_and_play_speech(f"Continuing with gentle poses. Now let's do {pose_name}.")
                elif detour_type == "challenging":
                    voice_feedback.generate_and_play_speech(f"Continuing with challenging poses. Now let's do {pose_name}.")
                else:
                    voice_feedback.generate_and_play_speech(f"Now let's do {pose_name}.")
    else:
        # Check if we just returned from a detour
        if hasattr(routine_manager, 'returning_from_detour') and routine_manager.returning_from_detour:
            voice_feedback.generate_and_play_speech(f"Great work! Now returning to your main routine. Let's do {pose_name}.")
            routine_manager.returning_from_detour = False  # Reset the flag
        else:
            # Regular routine progression
            voice_feedback.generate_and_play_speech(f"Now let's do {pose_name}.")
    
    return jsonify({
        'success': True,
        'workout_complete': False,
        'pose_name': pose_name,
        'pose_image': pose_image,
        'recommendation_type': recommendation_type,
        'emotion': current_emotion,
        'progress': progress,
        'announcement': get_pose_announcement(pose_name, recommendation_type, current_emotion, progress),
        'success_message': success_message,
        'emotion_message': emotion_message
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

def get_pose_image_path(pose_name):
    """Get the correct image path for a pose, with fallbacks for different file extensions"""
    # Clean pose name for filename matching
    clean_name = pose_name.lower().replace(' ', '_').replace("'", '').replace('-', '_')
    
    # Define possible image extensions
    extensions = ['.png', '.jpg', '.webp', '.jpeg']
    
    # Define image name mappings for poses that might have different names
    image_mappings = {
        'tree': 'tree',
        'bridge': 'bridge', 
        'cat': 'cat',
        'cow': 'cow',
        'child_pose': 'childs_pose',
        'childs_pose': 'childs_pose',
        'corpse_pose': 'corpse',
        'corpse': 'corpse',
        'downward_facing_dog': 'downward_facing_dog',
        'mountain_pose': 'mountain',
        'mountain': 'mountain',
        'warrior_i': 'warrior_one',
        'warrior_1': 'warrior_one',
        'warrior_one': 'warrior_one',
        'warrior_ii': 'warrior_two',
        'warrior_2': 'warrior_two', 
        'warrior_two': 'warrior_two',
        'wheel': 'wheel',
        'plank': 'plank',
        'boat': 'boat',
        'bow': 'bow',
        'butterfly': 'butterfly',
        'extended_side_angle': 'extended_side_angle',
        'half_boat': 'half_boat',
        'half_moon': 'half_moon',
        'legs_up_the_wall': 'legs_up_the_wall',
        'pigeon': 'pigeon',
        'seated_forward_bend': 'seated_forward_bend',
        'sphinx': 'sphinx',
        'upward_facing_dog': 'upward_facing_dog'
    }
    
    # Get the mapped image name or use the clean name
    image_name = image_mappings.get(clean_name, clean_name)
    
    # Try different extensions
    for ext in extensions:
        image_path = f'images/{image_name}{ext}'
        full_path = os.path.join('static', image_path)
        if os.path.exists(full_path):
            return image_path
    
    # Fallback to tree pose if image not found
    return 'images/tree.png'

def get_pose_announcement(pose_name, recommendation_type, emotion, progress):
    """Generate appropriate announcement based on context"""
    if progress['status'] == 'detour':
        if recommendation_type == "easier":
            return f"You seem {emotion}. Let's take a break with {pose_name}. We'll return to your routine after this."
        elif recommendation_type == "challenging":
            return f"Great energy! Let's try {pose_name} before continuing your routine."
        else:
            return f"Continuing with {pose_name}"
    else:
        routine_info = f"Pose {progress['current_pose_number']} of {progress['total_poses']}"
        return f"Back to your routine! {routine_info}: {pose_name}"

# Update pose_accuracy route to use routine manager
@app.route('/pose')
def pose_accuracy():
    # Get current pose from routine manager
    current_pose = routine_manager.get_current_pose()
    progress = routine_manager.get_progress_info()
    
    if current_pose is None:
        # No pose available, redirect back to voice interface
        return redirect(url_for('home'))
    
    pose_name = current_pose['name']
    pose_image_path = get_pose_image_path(pose_name)
    pose_image = url_for('static', filename=pose_image_path)
    
    return render_template('pose.html', 
                         pose_name=pose_name, 
                         pose_image=pose_image,
                         progress=progress)

# Add route to get current progress
@app.route('/get_progress', methods=['GET'])
def get_progress():
    """Get current routine progress"""
    progress = routine_manager.get_progress_info()
    
    return jsonify({
        'success': True,
        'progress': progress
    })

@app.route('/start_pose_tracking', methods=['POST'])
def start_pose_tracking():
    """Start tracking a new pose session"""
    data = request.get_json()
    pose_name = data.get('pose_name', 'Unknown Pose')
    
    session_id = pose_scorer.start_pose_session(pose_name)
    session['current_scoring_session'] = session_id
    
    return jsonify({'success': True, 'session_id': session_id})

@app.route('/end_pose_tracking', methods=['POST'])
def end_pose_tracking():
    """End pose tracking and get comprehensive score"""
    session_id = session.get('current_scoring_session')
    
    if not session_id:
        return jsonify({'success': False, 'error': 'No active tracking session'})
    
    score_data = pose_scorer.end_pose_session(session_id)
    
    if score_data:
        # Clear the session
        session.pop('current_scoring_session', None)
        return jsonify({'success': True, 'score_data': score_data})
    else:
        return jsonify({'success': False, 'error': 'Could not calculate score'})

# Update existing predict route to include scoring
@app.route('/predict', methods=['POST'])
def predict():
    global pose
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

    # Pose detection with optimized processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame for faster processing while maintaining accuracy
    height, width = frame.shape[:2]
    if width > 640:
        scale_factor = 640 / width
        new_width = 640
        new_height = int(height * scale_factor)
        img_rgb_resized = cv2.resize(img_rgb, (new_width, new_height))
    else:
        img_rgb_resized = img_rgb
        scale_factor = 1.0
    
    try:
        results = pose.process(img_rgb_resized)
    except ValueError as e:
        if "timestamp mismatch" in str(e):
            # Reinitialize pose processor to fix timestamp issues
            pose = mp_pose.Pose(
                static_image_mode=False, 
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1  # Use lighter model for better performance
            )
            results = pose.process(img_rgb_resized)
        else:
            raise e
    
    correct = False
    incorrect_angles = {}
    angles = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Scale back coordinates if we resized the image
        if scale_factor != 1.0:
            display_height, display_width = frame.shape[:2]
            angles = extract_angles(landmarks, int(display_width), int(display_height))
        else:
            angles = extract_angles(landmarks, width, height)
            
        features = np.array([angles[name] for name in ANGLE_NAMES]).reshape(1, -1)
        pred = clf.predict(features)[0]
        correct = bool(pred == 1)
        
        # Draw pose annotations on original frame for better visual quality
        for name, idxs in ANGLE_LANDMARKS.items():
            if scale_factor != 1.0:
                # Scale coordinates back to original frame size
                a = get_coords_scaled(landmarks, idxs[0], width, height)
                b = get_coords_scaled(landmarks, idxs[1], width, height)
                c = get_coords_scaled(landmarks, idxs[2], width, height)
            else:
                a = get_coords(landmarks, idxs[0], width, height)
                b = get_coords(landmarks, idxs[1], width, height)
                c = get_coords(landmarks, idxs[2], width, height)
                
            angle = angles[name]
            min_th, max_th = ANGLE_THRESHOLDS[name]
            is_angle_correct = min_th <= angle <= max_th
            if not is_angle_correct:
                incorrect_angles[name] = angle
            color = (0, 255, 0) if is_angle_correct else (0, 0, 255)
            cv2.line(frame, a, b, color, 3)
            cv2.line(frame, b, c, color, 3)
            cv2.putText(frame, f"{int(angle)}", b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Pass pose name to voice feedback for both correct and incorrect poses
        if angles:  # Only give feedback if we have valid pose data
            voice_feedback.give_feedback(correct, current_pose_name, incorrect_angles)

    # Detect emotion ONLY if detect_emotion is True (reduced frequency)
    emotion = "neutral"
    if detect_emotion:
        try:
            emotion, emotions = detect_emotion_from_frame(frame)
        except Exception as e:
            print(f"Emotion detection error: {e}")
            emotion = "neutral"

    # Update scoring session if active
    scoring_session_id = session.get('current_scoring_session')
    if scoring_session_id and angles:
        pose_scorer.update_pose_session(
            scoring_session_id, 
            correct, 
            angles, 
            ANGLE_THRESHOLDS, 
            emotion
        )
        
        # Track if feedback was given
        if not correct and incorrect_angles:
            pose_scorer.increment_feedback(scoring_session_id)

    # Draw emotion on frame with better positioning
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Optimize image encoding for faster transmission
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Balanced quality/speed
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_b64,
        'correct': correct,
        'emotion': emotion
    })

@app.route('/score')
def show_score():
    """Display pose score page"""
    return render_template('score.html')

if __name__ == '__main__':
    app.run(debug=True)