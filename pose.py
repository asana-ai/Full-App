import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr

def record_audio(duration=5, fs=44100, filename="output.wav"):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("Recording finished.")

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

def get_pose_with_image(pose_name: str) -> dict:
    return {
        'name': pose_name,
        'image': 'images/test.jpg'  # Correct: relative to static/
    }

def get_safe_poses(pain_area: str) -> list:
    # Create a pool of all poses excluding the ones for the painful area
    safe_poses = []
    for area, poses in pain_to_poses.items():
        if area != pain_area:
            safe_poses.extend(poses)
    
    # Randomly select 5 unique poses and add image info
    from random import sample
    selected_poses = sample(safe_poses, 5)
    return [get_pose_with_image(pose) for pose in selected_poses]

def identify_pain_area(text: str) -> dict:
    text = text.lower()
    for pain_area in pain_to_poses:
        if pain_area in text:
            poses = pain_to_poses[pain_area]
            return {
                'pain_area': pain_area,
                'poses': [get_pose_with_image(pose) for pose in poses]
            }
    # Default
    return {
        'pain_area': None,
        'poses': [get_pose_with_image(pose) for pose in default_poses]
    }

def suggest_yoga_poses():
    print("🎤 Please describe your pain area (e.g., 'My lower back hurts')...")
    record_audio(duration=5)
    
    text = transcribe_audio()
    if text:
        print(f"\n📝 You said: {text}")
        result = identify_pain_area(text)
        poses = result['poses']
        print("\n🧘‍♀️ Recommended Yoga Poses:")
        for i, pose in enumerate(poses, 1):
            print(f"{i}. {pose['name']}")
    else:
        print("❌ Could not process your request. Please try again.")

def transcribe_audio(filename="output.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == "__main__":
    suggest_yoga_poses()
