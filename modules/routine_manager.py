"""
Routine Management Module for Yoga App

This module handles workout sequences, emotion-based detours, and routine progression.
Designed for easy backend integration with database persistence for user routines.
"""

from typing import List, Dict, Optional, Any


class RoutineManager:
    """
    Manages yoga workout routines with emotion-based adaptations.
    
    Features:
    - Original routine tracking
    - Emotion-based detours (easier/harder poses)
    - Automatic return to main routine after detours
    - Progress tracking and analytics
    """
    
    def __init__(self):
        self.original_routine = []
        self.current_routine_index = 0
        self.emotion_detour_poses = []
        self.detour_active = False
        self.detour_index = 0
        self.routine_id = None
        self.user_id = "default"
        
    def start_routine(self, poses: List[Dict], routine_id: str = None, user_id: str = "default"):
        """
        Initialize a new routine
        
        Args:
            poses: List of pose dictionaries with 'name' and 'image' keys
            routine_id: Optional ID for routine tracking
            user_id: User identifier for personalization
        """
        self.original_routine = poses.copy()
        self.current_routine_index = 0
        self.emotion_detour_poses = []
        self.detour_active = False
        self.detour_index = 0
        self.routine_id = routine_id
        self.user_id = user_id
        
    def get_current_pose(self) -> Optional[Dict]:
        """
        Get the current pose based on routine state
        
        Returns:
            Dict with pose information or None if routine complete
        """
        if self.detour_active and self.detour_index < len(self.emotion_detour_poses):
            return self.emotion_detour_poses[self.detour_index]
        elif self.current_routine_index < len(self.original_routine):
            return self.original_routine[self.current_routine_index]
        else:
            return None  # Routine complete
            
    def advance_pose(self):
        """
        Move to the next pose in sequence
        
        Handles both main routine and detour progression
        """
        if self.detour_active:
            self.detour_index += 1
            # Check if detour is complete
            if self.detour_index >= len(self.emotion_detour_poses):
                self._end_detour()
        else:
            self.current_routine_index += 1
            
    def start_emotion_detour(self, emotion_poses: List[Dict], detour_type: str = "neutral"):
        """
        Start an emotion-based detour
        
        Args:
            emotion_poses: List of poses for the detour
            detour_type: Type of detour (easier/harder/neutral)
        """
        self.detour_active = True
        self.emotion_detour_poses = emotion_poses
        self.detour_index = 0
        self.detour_type = detour_type
        
    def _end_detour(self):
        """End emotion detour and return to original routine"""
        self.detour_active = False
        self.emotion_detour_poses = []
        self.detour_index = 0
        self.detour_type = None
        self.returning_from_detour = True  # Flag to indicate we just returned from detour
        
    def is_routine_complete(self) -> bool:
        """Check if the entire routine is complete"""
        return (not self.detour_active and 
                self.current_routine_index >= len(self.original_routine))
        
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information
        
        Returns:
            Comprehensive progress data for UI and analytics
        """
        total_original = len(self.original_routine)
        completed_original = self.current_routine_index
        
        status = "detour" if self.detour_active else "routine"
        
        # Calculate completion percentage
        completion_percentage = (completed_original / total_original * 100) if total_original > 0 else 0
        
        progress_data = {
            'routine_id': self.routine_id,
            'user_id': self.user_id,
            'total_poses': total_original,
            'completed_poses': completed_original,
            'current_pose_number': completed_original + 1,
            'completion_percentage': round(completion_percentage, 1),
            'status': status,
            'detour_progress': f"{self.detour_index + 1}/{len(self.emotion_detour_poses)}" if self.detour_active else None,
            'detour_type': getattr(self, 'detour_type', None),
            'routine_complete': self.is_routine_complete(),
            'current_pose': self.get_current_pose(),
            'remaining_poses': total_original - completed_original
        }
        
        return progress_data
        
    def get_routine_analytics(self) -> Dict[str, Any]:
        """
        Get analytics data for routine completion
        
        Returns:
            Analytics data for performance tracking
        """
        progress = self.get_progress_info()
        
        analytics = {
            'routine_completion_rate': progress['completion_percentage'],
            'total_poses_attempted': progress['completed_poses'],
            'detours_taken': 1 if hasattr(self, 'detour_type') and self.detour_type else 0,
            'routine_status': 'completed' if self.is_routine_complete() else 'in_progress'
        }
        
        return analytics
        
    def get_recommended_poses_for_emotion(self, emotion: str) -> List[Dict]:
        """
        Get recommended poses based on detected emotion
        
        Args:
            emotion: Detected emotion string
            
        Returns:
            List of recommended poses for emotion adaptation
        """
        emotion_pose_map = {
            # Struggling/Negative emotions - easier poses
            'angry': [
                {'name': 'Child\'s Pose', 'image': 'images/childs_pose.webp'},
                {'name': 'Corpse Pose', 'image': 'images/corpse.jpg'},
                {'name': 'Seated Forward Bend', 'image': 'images/seated_forward_bend.webp'}
            ],
            'sad': [
                {'name': 'Child\'s Pose', 'image': 'images/childs_pose.webp'},
                {'name': 'Legs Up The Wall', 'image': 'images/legs_up_the_wall.jpg'},
                {'name': 'Corpse Pose', 'image': 'images/corpse.jpg'}
            ],
            'fear': [
                {'name': 'Mountain Pose', 'image': 'images/mountain.png'},
                {'name': 'Child\'s Pose', 'image': 'images/childs_pose.webp'},
                {'name': 'Corpse Pose', 'image': 'images/corpse.jpg'}
            ],
            
            # Confident/Positive emotions - challenging poses
            'happy': [
                {'name': 'Wheel', 'image': 'images/wheel.webp'},
                {'name': 'Warrior One', 'image': 'images/warrior_one.jpg'},
                {'name': 'Extended Side Angle', 'image': 'images/extended_side_angle.jpg'}
            ],
            'surprise': [
                {'name': 'Wheel', 'image': 'images/wheel.webp'},
                {'name': 'Half Moon', 'image': 'images/half_moon.png'},
                {'name': 'Warrior Two', 'image': 'images/warrior_two.jpg'}
            ]
        }
        
        return emotion_pose_map.get(emotion, [])
        
    def should_adapt_for_emotion(self, emotion: str) -> bool:
        """
        Determine if routine should be adapted based on emotion
        
        Args:
            emotion: Detected emotion
            
        Returns:
            Boolean indicating if adaptation is recommended
        """
        adaptation_emotions = ['angry', 'sad', 'fear', 'happy', 'surprise']
        return emotion in adaptation_emotions and not self.detour_active
        
    def reset_routine(self):
        """Reset routine to beginning"""
        self.current_routine_index = 0
        self.emotion_detour_poses = []
        self.detour_active = False
        self.detour_index = 0
