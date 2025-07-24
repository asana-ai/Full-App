"""
Voice Feedback System Module for Yoga App

This module provides comprehensive text-to-speech feedback during yoga practice.
Designed for modular integration and easy customization of feedback messages.
"""

import time
import random
import threading
import tempfile
import os
from typing import Dict, List, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class VoiceFeedbackSystem:
    """
    Comprehensive voice feedback system for yoga practice.
    
    Features:
    - Pose-specific encouragement messages
    - Targeted correction guidance
    - Emotion-based adaptation announcements
    - Progress and completion feedback
    - Configurable feedback timing
    """
    
    def __init__(self, audio_enabled: bool = True, feedback_cooldown: float = 3.0):
        """
        Initialize voice feedback system
        
        Args:
            audio_enabled: Whether to enable audio output
            feedback_cooldown: Minimum seconds between feedback messages
        """
        self.audio_enabled = audio_enabled and PYGAME_AVAILABLE and GTTS_AVAILABLE
        self.feedback_cooldown = feedback_cooldown
        self.last_feedback_time = 0
        self.last_pose_state = None
        self.correct_pose_start = None
        
        # Initialize audio system
        if self.audio_enabled:
            try:
                pygame.mixer.init()
                print("Voice feedback system initialized successfully")
            except Exception as e:
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
            },
            'Warrior One': {
                'encouragement': [
                    "Strong warrior! Feel your power!",
                    "Perfect warrior stance! Hold that strength!",
                    "Excellent! You're a powerful warrior!",
                    "Beautiful warrior pose! Keep that focus!"
                ],
                'corrections': {
                    'left_knee': "Keep your front knee over your ankle",
                    'right_knee': "Straighten your back leg",
                    'left_hip': "Square your hips forward",
                    'right_hip': "Keep your hips facing forward",
                    'left_shoulder': "Reach your arms up strongly",
                    'right_shoulder': "Keep your shoulders over your hips"
                },
                'general': [
                    "Keep your front knee over your ankle",
                    "Straighten your back leg strongly",
                    "Square your hips toward the front"
                ]
            },
            'Downward Facing Dog': {
                'encouragement': [
                    "Strong downward dog! Great foundation!",
                    "Perfect! Feel the stretch through your spine!",
                    "Excellent dog pose! Keep breathing!",
                    "Beautiful inversion! Hold that strength!"
                ],
                'corrections': {
                    'left_knee': "Keep your legs straight",
                    'right_knee': "Straighten your back leg",
                    'left_hip': "Lift your hips high",
                    'right_hip': "Keep your hips level",
                    'left_shoulder': "Press your hands down firmly",
                    'right_shoulder': "Keep your arms straight"
                },
                'general': [
                    "Press your hands down and lift your hips",
                    "Straighten your legs and reach your tailbone up",
                    "Create an inverted V-shape with your body"
                ]
            }
        }
        
        # General feedback messages
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
        
        # Success and completion messages
        self.completion_messages = [
            "Excellent work! Moving to the next pose.",
            "Beautiful pose! Let's continue your practice.",
            "Perfect execution! Ready for the next challenge.",
            "Outstanding! Your practice is flowing beautifully."
        ]
        
        # Emotion-based messages
        self.emotion_messages = {
            'struggle': [
                "I sense you might need a gentler approach. Let's try some restorative poses.",
                "Take a breath. Let's slow down with some calming poses.",
                "Let's give your body some rest with easier variations."
            ],
            'confident': [
                "You're doing amazing! Ready for a challenge?",
                "Your energy is great! Let's add something more dynamic.",
                "I can see your confidence! Time for an advanced pose."
            ]
        }

    def generate_and_play_speech(self, text: str):
        """
        Generate and play text-to-speech audio
        
        Args:
            text: Text to convert to speech
        """
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

    def should_give_feedback(self) -> bool:
        """Check if enough time has passed since last feedback"""
        current_time = time.time()
        return current_time - self.last_feedback_time > self.feedback_cooldown

    def give_feedback(self, is_correct: bool, pose_name: str, incorrect_angles: Dict = None):
        """
        Provide pose-specific feedback based on current state
        
        Args:
            is_correct: Whether the pose is currently correct
            pose_name: Name of the current pose
            incorrect_angles: Dictionary of incorrect joint angles
        """
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

    def announce_pose_completion(self, pose_name: str) -> str:
        """
        Announce successful pose completion
        
        Args:
            pose_name: Name of completed pose
            
        Returns:
            Success message that was announced
        """
        success_messages = [
            f"Great job on {pose_name}! Moving to the next pose.",
            f"Excellent {pose_name}! Let's continue.",
            f"Perfect {pose_name}! Well done!",
            f"Beautiful {pose_name}! Keep up the great work!"
        ]
        
        message = random.choice(success_messages)
        self.generate_and_play_speech(message)
        return message

    def announce_emotion_adaptation(self, emotion: str, adaptation_type: str) -> str:
        """
        Announce emotion-based routine adaptation
        
        Args:
            emotion: Detected emotion
            adaptation_type: Type of adaptation (easier/harder)
            
        Returns:
            Adaptation message that was announced
        """
        if adaptation_type == "easier":
            message = random.choice(self.emotion_messages['struggle'])
        elif adaptation_type == "harder":
            message = random.choice(self.emotion_messages['confident'])
        else:
            message = "Let's continue with your practice."
            
        self.generate_and_play_speech(message)
        return message

    def announce_next_pose(self, pose_name: str):
        """
        Announce the next pose in sequence
        
        Args:
            pose_name: Name of the next pose
        """
        message = f"Now let's do {pose_name}."
        self.generate_and_play_speech(message)

    def announce_routine_completion(self):
        """Announce completion of entire routine"""
        completion_messages = [
            "Congratulations! You have completed your entire yoga routine! Great job!",
            "Amazing work! Your yoga session is complete!",
            "Outstanding! You've finished your full routine beautifully!",
            "Wonderful practice! You should be proud of your dedication!"
        ]
        
        message = random.choice(completion_messages)
        self.generate_and_play_speech(message)
        return message

    def add_custom_pose_feedback(self, pose_name: str, feedback_data: Dict):
        """
        Add custom feedback for a new pose
        
        Args:
            pose_name: Name of the pose
            feedback_data: Dictionary with encouragement, corrections, and general feedback
        """
        self.pose_specific_feedback[pose_name] = feedback_data

    def set_feedback_cooldown(self, cooldown: float):
        """
        Set the feedback cooldown period
        
        Args:
            cooldown: Seconds between feedback messages
        """
        self.feedback_cooldown = cooldown

    def enable_audio(self, enabled: bool):
        """
        Enable or disable audio output
        
        Args:
            enabled: Whether to enable audio
        """
        if enabled and PYGAME_AVAILABLE and GTTS_AVAILABLE:
            if not self.audio_enabled:
                try:
                    pygame.mixer.init()
                    self.audio_enabled = True
                    print("Audio enabled")
                except Exception as e:
                    print(f"Could not enable audio: {e}")
        else:
            self.audio_enabled = False
            print("Audio disabled")
