 
# import cv2
# import face_recognition
# import ollama
# import pyttsx3
# import os
# import numpy as np
# import threading
# import queue
# import threading
# import json
# import urllib.parse
# import urllib.request
# import sounddevice as sd
# from faster_whisper import WhisperModel



# from datetime import datetime
# from collections import deque

# #Settings#

# APP_DIR = r"C:\Users\sudar\Documents\Face_nlp_app"
# FACES_DIR = os.path.join(APP_DIR, "Known_Faces")



# # --- Setup TTS (run in background thread to avoid freezing) ---
# engine = pyttsx3.init()
# engine.setProperty('rate', 160)
# tts_lock = threading.Lock()

# def get_time ():
#     hour = datetime.now().hour
#     if hour >=5 and hour <=12:
#         return "Good Morning"
#     elif hour >=12 and hour <17:
#         return "Good Afternoon"
#     elif hour>= 17:
#         return "Good Evening"

# def speak(text):
#     print(f"Assistant: {text}")
#     def run():
#         with tts_lock:
#             engine.say(text)
#             engine.runAndWait()
#     threading.Thread(target=run, daemon=True).start()

# # --- Load known faces (subfolders per person) ---
# known_encodings = []
# known_names = []

# # faces_dir = "Known_Faces"
# # faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Known_Faces")
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

# # --- LLM Response (run in background thread) ---
# def get_response(name):
#     # new code
#     greeting = get_time()
#     current_time = datetime.now().strftime("%I:%M %p")
#     prompt = f"You are a friendly assistant. Greet '{greeting},{name}' warmly and ask how you can help them today. Keep it to 1-2 sentences."
#     response = ollama.chat(
#         model="llama3.2",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response["message"]["content"]

# def greet_person(name):
#     response = get_response(name)
#     speak(response)

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

# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Only process every 3rd frame for performance
#     frame_count += 1
#     if frame_count % 3 == 0:
#         small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#         locations = face_recognition.face_locations(rgb)
#         encodings = face_recognition.face_encodings(rgb, locations)

#         current_faces = []
#         for encoding, location in zip(encodings, locations):
#             matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
#             name = "Unknown"

#             if True in matches:
#                 distances = face_recognition.face_distance(known_encodings, encoding)
#                 best = np.argmin(distances)
#                 if matches[best]:
#                     name = known_names[best]

#             current_faces.append((name, location))

#             # Greet each new face in a background thread
#             if name != "Unknown" and name not in greeted:
#                 greeted.add(name)
#                 threading.Thread(target=greet_person, args=(name,), daemon=True).start()

#         # Store for drawing
#         last_faces = current_faces if current_faces else []

#     # Draw boxes for all detected faces
#     for name, location in locals().get('last_faces', []):
#         top, right, bottom, left = [v * 4 for v in location]
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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
# import time; time.sleep(2)  # Let goodbye finish speaking
# cap.release()
# cv2.destroyAllWindows()
# print("All Good")

"""
Face + Voice Assistant

- Camera: recognises known faces and greets them by time of day (local)
- Microphone: listens when the assistant isn't talking, transcribes what
  you say with faster-whisper (local), and answers with Llama 3.2 via Ollama
- Data: Llama can look things up to answer your question:
    * current time and date         (local)
    * weather for a city            (Open-Meteo, free, no API key)
    * facts about people/places/etc (Wikipedia)
- Speaker: replies are spoken with pyttsx3 (local)

Say "goodbye" or press Q / ESC (or the QUIT button) to exit.
"""
import json
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import ollama
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
APP_DIR = r"C:\Users\sudar\Documents\Face_nlp_app"
FACES_DIR = os.path.join(APP_DIR, "Known_Faces")

LLM_MODEL = "llama3.2"
WHISPER_SIZE = "base"          # "small" is more accurate but slower
WHISPER_LANGUAGE = "en"        # set to None to auto-detect language
DEFAULT_CITY = ""              # e.g. "Chennai": used when you ask "what's the weather?" with no city

SAMPLE_RATE = 16000
SILENCE_RMS = 0.01             # mic level counted as "speech"; raise in a noisy room
SILENCE_SECONDS = 1.0          # pause length that ends a sentence
MAX_UTTERANCE_SECONDS = 15
MIN_UTTERANCE_SECONDS = 0.6
FACE_TOLERANCE = 0.5
HTTP_TIMEOUT = 8

# ----------------------------------------------------------------------
# Shared state between threads
# ----------------------------------------------------------------------
stop_event = threading.Event()
speaking = threading.Event()
last_speech_end = [0.0]
tts_queue = queue.Queue()

faces_lock = threading.Lock()
faces_in_view = set()

SYSTEM_PROMPT = (
    "You are a friendly voice assistant running on a laptop with a camera. "
    "Your replies are spoken aloud, so keep them short (1-3 sentences), natural "
    "and conversational, with no lists, markdown, URLs or emojis. "
    "You have tools for the current date and time, the weather, and looking up "
    "facts on Wikipedia. Use a tool whenever the question needs real or current "
    "information instead of guessing, then answer from what the tool returned. "
    "If a tool fails or finds nothing, say so briefly. For simple chit-chat, "
    "just reply without tools. Messages may start with context in square brackets "
    "saying who is on camera; address people by name when you know it, but never "
    "mention the brackets."
)
history_lock = threading.Lock()
history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ----------------------------------------------------------------------
# Data tools (Llama decides when to call these)
# ----------------------------------------------------------------------
def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "FaceVoiceAssistant/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_current_datetime() -> str:
    """Get the current local date, day of the week and time.

    Returns:
        The current date and time as text.
    """
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def get_weather(city: str) -> str:
    """Get the current weather and today's forecast for a city.

    Args:
        city: Name of the city, for example "Mumbai" or "London".

    Returns:
        A short weather summary.
    """
    city = (city or DEFAULT_CITY).strip()
    if not city:
        return "No city given. Ask the user which city they mean."

    geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                    + urllib.parse.quote(city))
    if not geo.get("results"):
        return f"Could not find a place called {city}."
    place = geo["results"][0]

    params = urllib.parse.urlencode({
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto", "forecast_days": 1,
    })
    data = _get_json("https://api.open-meteo.com/v1/forecast?" + params)
    cur, day = data["current"], data["daily"]
    desc = WEATHER_CODES.get(cur["weather_code"], "unknown conditions")
    return (
        f"{place['name']}, {place.get('country', '')}: {desc}, "
        f"{cur['temperature_2m']}°C (feels like {cur['apparent_temperature']}°C), "
        f"humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} km/h. "
        f"Today: high {day['temperature_2m_max'][0]}°C, low {day['temperature_2m_min'][0]}°C, "
        f"{day['precipitation_probability_max'][0]}% chance of rain."
    )


def search_wikipedia(topic: str) -> str:
    """Look up facts about a person, place, thing or event on Wikipedia.

    Args:
        topic: What to look up, for example "Taj Mahal" or "APJ Abdul Kalam".

    Returns:
        A short summary of the topic.
    """
    search = _get_json(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1&srsearch="
        + urllib.parse.quote(topic))
    hits = search.get("query", {}).get("search", [])
    if not hits:
        return f"Wikipedia has nothing on {topic}."
    title = hits[0]["title"]
    summary = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                        + urllib.parse.quote(title.replace(" ", "_")))
    extract = summary.get("extract", "")
    return f"{title}: {extract[:1200]}" if extract else f"No summary found for {title}."


TOOLS = {
    "get_current_datetime": get_current_datetime,
    "get_weather": get_weather,
    "search_wikipedia": search_wikipedia,
}


# ----------------------------------------------------------------------
# Text to speech (one dedicated thread, so pyttsx3 is used safely)
# ----------------------------------------------------------------------
def tts_worker():
    try:
        import pythoncom  # comes with pywin32, needed for SAPI in a thread
        pythoncom.CoInitialize()
    except ImportError:
        pass

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    while True:
        text = tts_queue.get()
        if text is None:
            tts_queue.task_done()
            break
        speaking.set()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            last_speech_end[0] = time.time()
            speaking.clear()
            tts_queue.task_done()


def speak(text):
    print(f"Assistant: {text}")
    tts_queue.put(text)


def assistant_is_talking():
    return (speaking.is_set() or not tts_queue.empty()
            or time.time() - last_speech_end[0] < 0.6)


# ----------------------------------------------------------------------
# Faces
# ----------------------------------------------------------------------
def load_known_faces():
    encodings_out, names_out = [], []
    if not os.path.isdir(FACES_DIR):
        print(f"⚠️  Faces folder not found: {FACES_DIR}")
        return encodings_out, names_out

    for person_name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        person_encodings = []
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = face_recognition.load_image_file(os.path.join(person_dir, filename))
            encs = face_recognition.face_encodings(img)
            if encs:
                person_encodings.append(encs[0])
            else:
                print(f"  ⚠️  No face found in {filename}, skipping")

        if person_encodings:
            encodings_out.append(np.mean(person_encodings, axis=0))
            names_out.append(person_name.capitalize())
            print(f"✅ Face: {person_name} ({len(person_encodings)} photos)")
        else:
            print(f"❌ No valid photos for {person_name}, skipping")
    return encodings_out, names_out


# ----------------------------------------------------------------------
# Listening + speech to text
# ----------------------------------------------------------------------
def record_utterance():
    """Wait for speech, record until a pause. Returns float32 audio or None."""
    chunk = int(0.1 * SAMPLE_RATE)
    preroll = deque(maxlen=3)  # keep ~0.3 s before speech so the first word isn't clipped
    frames, started, silent, total = [], False, 0.0, 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while not stop_event.is_set():
            if assistant_is_talking():
                return None  # don't record our own voice
            data, _ = stream.read(chunk)
            data = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(data ** 2)))

            if not started:
                preroll.append(data)
                if rms > SILENCE_RMS:
                    started = True
                    frames.extend(preroll)
                continue

            frames.append(data)
            total += 0.1
            silent = silent + 0.1 if rms < SILENCE_RMS else 0.0
            if silent >= SILENCE_SECONDS or total >= MAX_UTTERANCE_SECONDS:
                break

    if not frames:
        return None
    audio = np.concatenate(frames)
    if len(audio) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
        return None
    return audio


def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio, language=WHISPER_LANGUAGE, beam_size=1)
    # Skip segments Whisper itself thinks are probably not speech (cuts hallucinations)
    text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6)
    return text.strip()


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------
def get_time_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    return "Hello"


def greet_person(name):
    greeting = get_time_greeting()
    prompt = (
        f"The current time is {datetime.now().strftime('%I:%M %p')}. "
        f"Begin your reply with exactly '{greeting}, {name}!' and then ask how "
        f"you can help them today. Keep it to 1-2 sentences."
    )
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    reply = response.message.content.strip()
    with history_lock:
        history.append({"role": "assistant", "content": reply})
    speak(reply)


def answer_question(text, seen):
    context = f"[On camera: {', '.join(seen) if seen else 'nobody recognised'}.]"
    with history_lock:
        history.append({"role": "user", "content": f"{context}\n{text}"})
        messages = [history[0]] + history[1:][-12:]  # system prompt + recent turns

    tool_list = list(TOOLS.values())
    response = ollama.chat(model=LLM_MODEL, messages=messages, tools=tool_list)

    # Let Llama call tools (up to 3 rounds), feeding results back each time
    for _ in range(3):
        calls = response.message.tool_calls or []
        if not calls:
            break
        messages.append(response.message)
        for call in calls:
            name, args = call.function.name, call.function.arguments or {}
            func = TOOLS.get(name)
            try:
                result = func(**args) if func else f"Unknown tool {name}."
            except Exception as e:
                result = f"The {name} lookup failed: {e}"
            print(f"🔎 {name}({args}) -> {str(result)[:120]}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=tool_list)

    reply = (response.message.content or "").strip() or "Sorry, I couldn't find an answer to that."
    with history_lock:
        history.append({"role": "assistant", "content": reply})
    return reply


def voice_loop():
    print("🎧 Listening for questions...")
    while not stop_event.is_set():
        if assistant_is_talking():
            time.sleep(0.1)
            continue
        try:
            audio = record_utterance()
            if audio is None:
                continue

            text = transcribe(audio)
            if len(text) < 2:
                continue
            print(f"🎤 You: {text}")

            if any(w in text.lower() for w in ("goodbye", "bye bye", "stop listening")):
                stop_event.set()
                break

            with faces_lock:
                seen = sorted(faces_in_view)
            speak(answer_question(text, seen))
        except Exception as e:
            print(f"⚠️  Voice error: {e}")
            time.sleep(1)


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------
threading.Thread(target=tts_worker, daemon=True).start()

print("Loading faces...")
known_face_encodings, known_face_names = load_known_faces()

print("Loading speech model (first run downloads it, then works offline)...")
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

print(f"\nReady. Faces: {known_face_names}\n")
threading.Thread(target=voice_loop, daemon=True).start()

# ----------------------------------------------------------------------
# Camera loop (main thread)
# ----------------------------------------------------------------------
cap = cv2.VideoCapture(0)
greeted = set()
quit_flag = [False]
last_faces = []
frame_count = 0


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 120 and 10 <= y <= 50:
        param[0] = True


WINDOW = "Face + Voice Assistant"
print("Starting camera... Press Q or ESC to quit, or say 'goodbye'.")
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse, quit_flag)

while not stop_event.is_set():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 == 0:  # process every 3rd frame for speed
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        current_faces = []
        for encoding, location in zip(encodings, locations):
            name = "Unknown"
            if known_face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, encoding, tolerance=FACE_TOLERANCE)
                if True in matches:
                    distances = face_recognition.face_distance(known_face_encodings, encoding)
                    best = int(np.argmin(distances))
                    if matches[best]:
                        name = known_face_names[best]
            current_faces.append((name, location))

            if name != "Unknown" and name not in greeted:
                greeted.add(name)
                threading.Thread(target=greet_person, args=(name,), daemon=True).start()

        last_faces = current_faces
        with faces_lock:
            faces_in_view.clear()
            faces_in_view.update(n for n, _ in current_faces if n != "Unknown")

    for name, location in last_faces:
        top, right, bottom, left = [v * 4 for v in location]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    status = "Speaking..." if speaking.is_set() else "Listening..."
    cv2.putText(frame, status, (140, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.rectangle(frame, (10, 10), (120, 50), (0, 0, 255), -1)
    cv2.putText(frame, "QUIT [Q]", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27 or quit_flag[0]:
        break

# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------
print("Closing application...")
stop_event.set()
cap.release()
cv2.destroyAllWindows()

goodbye = "Good night!" if datetime.now().hour >= 21 or datetime.now().hour < 5 else "Goodbye!"
speak(goodbye)
deadline = time.time() + 6
while (speaking.is_set() or not tts_queue.empty()) and time.time() < deadline:
    time.sleep(0.1)
tts_queue.put(None)
print("All Good")