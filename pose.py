import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import json

# --- Load Pose Data from JSON at startup ---
with open('poses.json', 'r') as f:
    all_poses_data = json.load(f)['poses']

# Create a lookup dictionary for quick access by name
poses_lookup = {pose['name']: pose for pose in all_poses_data}

# --- Pain to Poses Mapping (uses pose names) ---
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

default_poses_names = ["Mountain Pose", "Tree Pose", "Corpse Pose", "Butterfly Pose", "Legs Up the Wall"]

def record_audio(duration=5, fs=44100, filename="output.wav"):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("Recording finished.")

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

def identify_pain_area(text: str) -> dict:
    """
    Identifies the pain area from text and returns a list of full pose objects.
    """
    text = text.lower()
    pain_area_found = "general"
    recommended_pose_names = default_poses_names

    for pain_area, pose_names in pain_to_poses.items():
        if pain_area in text:
            pain_area_found = pain_area
            recommended_pose_names = pose_names
            break

    # Use the lookup table to get the full pose objects for the recommended names
    recommended_poses = [poses_lookup[name] for name in recommended_pose_names if name in poses_lookup]

    return {
        'pain_area': pain_area_found,
        'poses': recommended_poses
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

if __name__ == "__main__":
    suggest_yoga_poses()
