# import cv2
# import face_recognition
# import ollama
# import pyttsx3
# import os
# import numpy as np

# # --- Setup TTS ---
# engine = pyttsx3.init()
# engine.setProperty('rate', 160)

# def speak(text):
#     print(f"Assistant: {text}")
#     engine.say(text)
#     engine.runAndWait()

# # --- Load known faces (subfolders per person) ---
# known_encodings = []
# known_names = []

# # faces_dir = "Known_Faces"
# faces_dir = r"C:\Users\sudar\Documents\Face_nlp_app\Known_Faces"

# for person_name in os.listdir(faces_dir):
#     person_dir = os.path.join(faces_dir, person_name)
#     if not os.path.isdir(person_dir):
#         continue

#     person_encodings = []
#     for filename in os.listdir(person_dir):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#         img_path = os.path.join(person_dir, filename)
#         img = face_recognition.load_image_file(img_path)
#         encodings = face_recognition.face_encodings(img)
#         if encodings:
#             person_encodings.append(encodings[0])
#         else:
#             print(f"  ⚠️  No face found in {filename}, skipping")

#     if person_encodings:
#         avg_encoding = np.mean(person_encodings, axis=0)
#         known_encodings.append(avg_encoding)
#         known_names.append(person_name.capitalize())
#         print(f"✅ Loaded {person_name} ({len(person_encodings)} photos)")
#     else:
#         print(f"❌ No valid photos for {person_name}, skipping")

# print(f"\nReady with {len(known_names)} people: {known_names}\n")

# # --- LLM Response ---
# def get_response(name):
#     prompt = f"You are a friendly assistant. Greet {name} warmly and ask how you can help them today. Keep it to 1-2 sentences."
#     response = ollama.chat(
#         model="llama3.2",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response["message"]["content"]

# # --- Main Loop ---
# cap = cv2.VideoCapture(0)
# greeted = set()

# quit_flag = [False]

# def on_mouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if 10 <= x <= 120 and 10 <= y <= 50:
#             param[0] = True

# print("Starting camera... Press Q or ESC to quit.")

# cv2.namedWindow("Face Recognition Assistant")
# cv2.setMouseCallback("Face Recognition Assistant", on_mouse, quit_flag)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#     locations = face_recognition.face_locations(rgb)
#     encodings = face_recognition.face_encodings(rgb, locations)

#     for encoding, location in zip(encodings, locations):
#         matches = face_recognition.compare_faces(known_encodings, encoding)
#         name = "Unknown"

#         if True in matches:
#             distances = face_recognition.face_distance(known_encodings, encoding)
#             best = np.argmin(distances)
#             if matches[best]:
#                 name = known_names[best]

#         top, right, bottom, left = [v * 4 for v in location]
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         if name != "Unknown" and name not in greeted:
#             greeted.add(name)
#             response = get_response(name)
#             speak(response)

#     # Draw quit button
#     cv2.rectangle(frame, (10, 10), (120, 50), (0, 0, 255), -1)
#     cv2.putText(frame, "QUIT [Q]", (15, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     cv2.imshow("Face Recognition Assistant", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27 or quit_flag[0]:
#         break

# print("Closing application...")
# speak("Goodbye!")
# cap.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition
import ollama
import pyttsx3
import os
import numpy as np
import threading
from datetime import datetime

# --- Setup TTS (run in background thread to avoid freezing) ---
engine = pyttsx3.init()
engine.setProperty('rate', 160)
tts_lock = threading.Lock()

def get_time ():
    hour = datetime.now().hour
    if hour >=5 and hour <=12:
        return "Good Morning"
    elif hour >=12 and hour <17:
        return "Good Afternoon"
    elif hour>= 17:
        return "Good Evening"

def speak(text):
    print(f"Assistant: {text}")
    def run():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# --- Load known faces (subfolders per person) ---
known_encodings = []
known_names = []

# faces_dir = "Known_Faces"
# faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Known_Faces")
faces_dir = r"C:\Users\sudar\Documents\Face_nlp_app\Known_Faces"
for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    person_encodings = []
    for filename in os.listdir(person_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(person_dir, filename)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            person_encodings.append(encodings[0])
        else:
            print(f"  ⚠️  No face found in {filename}, skipping")

    if person_encodings:
        avg_encoding = np.mean(person_encodings, axis=0)
        known_encodings.append(avg_encoding)
        known_names.append(person_name.capitalize())
        print(f"✅ Loaded {person_name} ({len(person_encodings)} photos)")
    else:
        print(f"❌ No valid photos for {person_name}, skipping")

print(f"\nReady with {len(known_names)} people: {known_names}\n")

# --- LLM Response (run in background thread) ---
def get_response(name):
    # new code
    greeting = get_time()
    current_time = datetime.now().strftime("%I:%M %p")
    prompt = f"You are a friendly assistant. Greet '{greeting},{name}' warmly and ask how you can help them today. Keep it to 1-2 sentences."
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def greet_person(name):
    response = get_response(name)
    speak(response)

# --- Main Loop ---
cap = cv2.VideoCapture(0)
greeted = set()
quit_flag = [False]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 120 and 10 <= y <= 50:
            param[0] = True

print("Starting camera... Press Q or ESC to quit.")
cv2.namedWindow("Face Recognition Assistant")
cv2.setMouseCallback("Face Recognition Assistant", on_mouse, quit_flag)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every 3rd frame for performance
    frame_count += 1
    if frame_count % 3 == 0:
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        current_faces = []
        for encoding, location in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best = np.argmin(distances)
                if matches[best]:
                    name = known_names[best]

            current_faces.append((name, location))

            # Greet each new face in a background thread
            if name != "Unknown" and name not in greeted:
                greeted.add(name)
                threading.Thread(target=greet_person, args=(name,), daemon=True).start()

        # Store for drawing
        last_faces = current_faces if current_faces else []

    # Draw boxes for all detected faces
    for name, location in locals().get('last_faces', []):
        top, right, bottom, left = [v * 4 for v in location]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw quit button
    cv2.rectangle(frame, (10, 10), (120, 50), (0, 0, 255), -1)
    cv2.putText(frame, "QUIT [Q]", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Assistant", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 or quit_flag[0]:
        break

print("Closing application...")
speak("Goodbye!")
import time; time.sleep(2)  # Let goodbye finish speaking
cap.release()
cv2.destroyAllWindows()
print("All Good")