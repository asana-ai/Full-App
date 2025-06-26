import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- CONFIGURATION ---
positive_folder = r"C:\Users\kadam\Desktop\Dataset\BALASANA_keypoints"
negative_folder = r"C:\Users\kadam\Desktop\Dataset\TreePose_keypoints"  # Optional
model_output_path = "models/child_pose_classifier.pkl"

def load_keypoints_from_folder(folder_path, label):
    X, y = [], []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder does not exist: {folder_path}")
        return X, y

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file))
            X.append(data)
            y.append(label)
    return X, y

def main():
    print("📥 Loading data...")
    X_pos, y_pos = load_keypoints_from_folder(positive_folder, 1)

    if os.path.exists(negative_folder):
        X_neg, y_neg = load_keypoints_from_folder(negative_folder, 0)
    else:
        print("⚠️ No negative folder found. Generating random negatives.")
        X_neg = [np.random.rand(len(X_pos[0])) for _ in range(len(X_pos))]
        y_neg = [0] * len(X_neg)

    X = np.array(X_pos + X_neg)
    y = np.array(y_pos + y_neg)

    print(f"🧠 Training model on {len(X)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"✅ Model saved to {model_output_path}")

if __name__ == "__main__":
    main()
