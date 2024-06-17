import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import ttk, font
from sklearn.cluster import KMeans

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path, frame_skip=5, resize_factor=0.5):
    print(f"Starting video processing for {video_path}")
    cap = cv2.VideoCapture(video_path)
    joint_coordinates = []
    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return joint_coordinates

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            joint_coordinates.append(frame_landmarks)
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Video Processing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return joint_coordinates

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def analyze_squat(joint_coordinates, view_angle):
    hip_feedback = []
    knee_feedback = []
    ankle_feedback = []
    torso_feedback = []

    for frame in joint_coordinates:
        hip = frame[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = frame[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = frame[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        shoulder = frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        foot = frame[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        if view_angle in ["side", "45-degree front-left", "45-degree front-right", "45-degree behind-left", "45-degree behind-right"]:
            hip_knee_angle = calculate_angle(hip, knee, shoulder)
            knee_angle = calculate_angle(hip, knee, ankle)
            ankle_angle = calculate_angle(knee, ankle, foot)
            torso_angle = calculate_angle((shoulder[0], shoulder[1] - 1), shoulder, hip)

            if hip_knee_angle < 90:
                hip_feedback.append("Hip flexion is adequate.")
            else:
                hip_feedback.append("Hip flexion is not adequate. Squat deeper.")

            if knee_angle >= 90:
                knee_feedback.append("Knee flexion is adequate.")
            else:
                knee_feedback.append("Knee flexion is not adequate. Squat deeper.")

            if 25 <= ankle_angle <= 30:
                ankle_feedback.append("Ankle dorsiflexion is within recommended range.")
            else:
                ankle_feedback.append("Ankle dorsiflexion is not within recommended range. Bend forward more.")

            if 30 <= torso_angle <= 50:
                torso_feedback.append("Torso angle is within recommended range.")
            else:
                torso_feedback.append("Torso angle is not within recommended range. Keep your torso upright.")

        if view_angle in ["front", "45-degree front-left", "45-degree front-right"]:
            knee_distance = np.linalg.norm(np.array(frame[mp_pose.PoseLandmark.LEFT_KNEE.value]) - np.array(frame[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            if knee_distance < 0.2:
                knee_feedback.append("Keep knees aligned with toes.")
            else:
                knee_feedback.append("Don't let your knees cave in.")

    # Consolidate feedback for each joint
    consolidated_feedback = {
        "Hip Flexion": max(set(hip_feedback), key=hip_feedback.count),
        "Knee Flexion": max(set(knee_feedback), key=knee_feedback.count),
        "Ankle Dorsiflexion": max(set(ankle_feedback), key=ankle_feedback.count),
        "Torso Angle": max(set(torso_feedback), key=torso_feedback.count)
    }

    return consolidated_feedback

def analyze_deadlift(joint_coordinates, view_angle):
    bottom_phase_feedback = []
    for frame in joint_coordinates:
        hip, knee, ankle = frame[mp_pose.PoseLandmark.LEFT_HIP.value], frame[mp_pose.PoseLandmark.LEFT_KNEE.value], frame[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        if view_angle in ["side", "45-degree front-left", "45-degree front-right", "45-degree behind-left", "45-degree behind-right"]:
            hip_knee_angle = calculate_angle(hip, knee, ankle)
            bottom_phase_feedback.append("Keep your back straight and bend at the hips." if hip_knee_angle < 160 else "Good form, keep it up!")
        elif view_angle in ["front", "45-degree front-left", "45-degree front-right"]:
            knee_distance = np.linalg.norm(np.array(frame[mp_pose.PoseLandmark.LEFT_KNEE.value]) - np.array(frame[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            bottom_phase_feedback.append("Keep knees aligned with toes." if knee_distance < 0.2 else "Don't let your knees cave in.")
    consolidated_feedback = {
        "Deadlift Feedback": max(set(bottom_phase_feedback), key=bottom_phase_feedback.count)
    }
    return consolidated_feedback

def analyze_bench_press(joint_coordinates, view_angle):
    bottom_phase_feedback = []
    for frame in joint_coordinates:
        shoulder, elbow, wrist = frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value], frame[mp_pose.PoseLandmark.LEFT_ELBOW.value], frame[mp_pose.PoseLandmark.LEFT_WRIST.value]
        if view_angle in ["side", "45-degree front-left", "45-degree front-right", "45-degree behind-left", "45-degree behind-right"]:
            shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)
            bottom_phase_feedback.append("Keep your elbows at a 90-degree angle." if shoulder_elbow_angle < 90 else "Good form, keep it up!")
        elif view_angle in ["front", "45-degree front-left", "45-degree front-right"]:
            shoulder_distance = np.linalg.norm(np.array(frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) - np.array(frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]))
            bottom_phase_feedback.append("Keep shoulders stable and symmetrical." if shoulder_distance < 0.1 else "Don't let your shoulders shift.")
    consolidated_feedback = {
        "Bench press Feedback": max(set(bottom_phase_feedback), key=bottom_phase_feedback.count)
    }
    return consolidated_feedback

def classify_exercise(joint_coordinates):
    if not joint_coordinates:
        print("Error: No joint coordinates found.")
        return "unknown"

    angles = []
    for frame in joint_coordinates:
        shoulder, elbow, wrist = frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value], frame[mp_pose.PoseLandmark.LEFT_ELBOW.value], frame[mp_pose.PoseLandmark.LEFT_WRIST.value]
        hip, knee, ankle = frame[mp_pose.PoseLandmark.LEFT_HIP.value], frame[mp_pose.PoseLandmark.LEFT_KNEE.value], frame[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_knee_angle = calculate_angle(hip, knee, ankle)
        angles.append([shoulder_elbow_angle, hip_knee_angle])

    angles = np.array(angles)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(angles)
    labels = kmeans.labels_

    squat_count = np.sum(labels == 0)
    deadlift_count = np.sum(labels == 1)
    benchpress_count = np.sum(labels == 2)

    print(f"Squat count: {squat_count}, Deadlift count: {deadlift_count}, Benchpress count: {benchpress_count}")

    if squat_count > deadlift_count and squat_count > benchpress_count:
        return "squat"
    elif deadlift_count > squat_count and deadlift_count > benchpress_count:
        return "deadlift"
    else:
        return "benchpress"

def analyze_video(video_path, view_angle):
    print(f"Analyzing video: {video_path}")
    joint_coordinates = process_video(video_path)
    if not joint_coordinates:
        return {"Error": ["Could not process video or no landmarks detected."]}

    exercise_type = classify_exercise(joint_coordinates)
    print(f"Detected exercise: {exercise_type}")

    if exercise_type == "squat":
        feedback = analyze_squat(joint_coordinates, view_angle)
    elif exercise_type == "deadlift":
        feedback = analyze_deadlift(joint_coordinates, view_angle)
    elif exercise_type == "benchpress":
        feedback = analyze_bench_press(joint_coordinates, view_angle)
    else:
        feedback = {"Error": ["Unknown exercise detected. Please try again with a different video."]}

    return feedback

# GUI Setup
def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
    if file_path:
        video_path.set(file_path)

def process_selected_video():
    file_path = video_path.get()
    view_angle = view_angle_var.get()
    if not file_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    feedback = analyze_video(file_path, view_angle)
    feedback_text.set("\n".join([f"{key}: {value}" for key, value in feedback.items()]))

app = tk.Tk()
app.title("Exercise Form Analyzer")

# Define a custom style for the frame
style = ttk.Style()
style.configure("LightPink.TFrame", background="#FFD1DC")  # Set the background color to light pink
style.configure("LightPink.TLabel", background="#FFD1DC", foreground="black")  # Set label background and text color
style.configure("LightPink.TButton", background="#FFD1DC", foreground="black")  # Set button background and text color
style.configure("LightPink.TEntry", fieldbackground="#FFD1DC", foreground="black")  # Set entry field background and text color
style.configure("LightPink.TCombobox", fieldbackground="#FFD1DC", foreground="black")  # Set combobox field background and text color

video_path = tk.StringVar()
feedback_text = tk.StringVar()
view_angle_var = tk.StringVar(value="side")

frame = ttk.Frame(app, padding="10", style="LightPink.TFrame")  # Apply the custom style to the frame
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Title
title_label = ttk.Label(frame, text="Exercise Form Analyzer", style="Bold.TLabel", background="#FFD1DC")
title_label.grid(row=0, column=0, columnspan=3, sticky=tk.W)

# Configure font to make title bold or increase font size
title_font = font.Font(title_label, title_label.cget("font"))
title_font.configure(weight="bold", size=14)  # Adjust weight and size as needed
title_label.configure(font=title_font)

# View Angle Definitions
view_angle_definitions = """
View Angle Definitions:
- Front: Directly facing the person.
- Side: 90 degrees to the left or right of the person.
- 45-degree front-left: 45 degrees to the left of the front view.
- 45-degree front-right: 45 degrees to the right of the front view.
- 45-degree behind-left: 45 degrees to the left of the back view.
- 45-degree behind-right: 45 degrees to the right of the back view.
"""
# Create a label with light pink background for the view angle definitions
view_angle_definitions_label = ttk.Label(frame, text=view_angle_definitions, wraplength=500, justify="left", anchor="w", foreground="black", style="LightPink.TLabel")
view_angle_definitions_label.grid(row=1, column=0, columnspan=3, sticky=tk.W)
view_angle_definitions_label.configure(background="#FFD1DC")  # Set background color to light pink

# Select Video File
ttk.Label(frame, text="Select Video File:", style="LightPink.TLabel").grid(row=2, column=0, sticky=tk.W)
ttk.Entry(frame, textvariable=video_path, width=40, style="LightPink.TEntry").grid(row=2, column=1, sticky=tk.E)
ttk.Button(frame, text="Browse", command=select_video, style="LightPink.TButton").grid(row=2, column=2, sticky=tk.E)

# View angle selection
ttk.Label(frame, text="View Angle:", style="LightPink.TLabel").grid(row=3, column=0, sticky=tk.W)

# Create the OptionMenu widget for view angle selection
view_angle_menu = ttk.OptionMenu(frame, view_angle_var, "side", "side", "front", "45-degree front-left", "45-degree front-right", "45-degree behind-left", "45-degree behind-right")

# Configure the style for the dropdown menu options
style.configure("Dropdown.TMenubutton", foreground="black")

# Apply the custom style to each individual option in the dropdown menu
for child in view_angle_menu["menu"].winfo_children():
    style.configure(child, foreground="black")

# Grid the OptionMenu widget
view_angle_menu.grid(row=3, column=1, columnspan=2, sticky=tk.W)

# Analysis button
ttk.Button(frame, text="Analyze", command=process_selected_video, style="LightPink.TButton").grid(row=4, column=0, columnspan=3, pady=10)

# Feedback
ttk.Label(frame, text="Feedback:", style="LightPink.TLabel").grid(row=5, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=feedback_text, wraplength=400, style="LightPink.TLabel").grid(row=6, column=0, columnspan=3, sticky=tk.W)

app.mainloop()
