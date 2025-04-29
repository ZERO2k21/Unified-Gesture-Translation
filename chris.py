import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import whisper
import pyaudio
import threading
import pyttsx3
import numpy as np

# Initialize modes
mode = "braille"  # Default mode

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Define gesture classes for each mode
braille_classes = {'1', '2', '3', '4', '5', '6', 'O'}
morse_classes = {'1', '2', 'O'}

# Morse and Braille mappings
morse_code = []
braille_code = list("000000")

# Braille character mapping (dots to characters)
braille_dict = {
    "100000": "A", "101000": "B", "110000": "C", "110100": "D",
    "100100": "E", "111000": "F", "111100": "G", "101100": "H",
    "011000": "I", "011100": "J", "100010": "K", "101010": "L",
    "110010": "M", "110110": "N", "100110": "O", "111010": "P",
    "111110": "Q", "101110": "R", "011010": "S", "011110": "T",
    "100011": "U", "101011": "V", "011101": "W", "110011": "X",
    "110111": "Y", "100111": "Z"
}

whisper_model = whisper.load_model("base")

# Create a GestureRecognizer object
gesture_model_path = os.path.abspath("gesture_recognizer.task")
gesture_recognizer = vision.GestureRecognizer.create_from_model_path(gesture_model_path)

# Initialize MediaPipe Hands for keypoint detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Initialize PyAudio for Vosk
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Variable to store the recognized command
recognized_command = None

# Variable to track the previous frame status
previously_detected_gesture = False

# Variable to store the last detected gesture
last_detected_gesture = None

def speak_text(text):
    """Function to speak text using TTS engine."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def recognize_speech():
    global recognized_command
    audio_buffer = bytes()
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        audio_buffer += data
        # Process audio every 5 seconds worth of data
        if len(audio_buffer) >= 16000 * 2 * 5:  # 16000 samples/sec, 2 bytes per sample, 5 seconds
            # Convert byte buffer to numpy array of float32 in range [-1, 1]
            audio_data = np.frombuffer(audio_buffer, np.int16).astype(np.float32) / 32768.0
            result = whisper_model.transcribe(audio_data)
            text = result.get("text", "").strip().lower()
            if text:
                recognized_command = text
                print(f"Recognized command: {text}")
            audio_buffer = bytes()

# Start the speech recognition in a separate thread
speech_thread = threading.Thread(target=recognize_speech)
speech_thread.daemon = True
speech_thread.start()

def process_morse_code():
    global morse_code
    morse_mapping = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
        "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
        "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
        "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
        "-.--": "Y", "--..": "Z"
    }
    morse_str = ''.join(morse_code)
    character = morse_mapping.get(morse_str, "?")
    print(f"Morse Code: {morse_str} -> Character: {character}")
    speak_text(f"Morse Code {morse_str} is {character}")
    morse_code = []  # Reset the morse code sequence

def process_braille_code():
    global braille_code
    braille_str = ''.join(braille_code)
    character = braille_dict.get(braille_str, "?")
    print(f"Braille Dots: {braille_str} -> Character: {character}")
    speak_text(f"Braille Dots {braille_str} is {character}")
    braille_code = list("000000")  # Reset the braille code sequence

def is_gesture_valid(gesture_name, mode):
    if mode == "braille":
        return gesture_name in braille_classes
    elif mode == "morse":
        return gesture_name in morse_classes
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe image object with the correct format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    detected_gesture = False  # Flag to check if any gesture is detected in the current frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gestures using the formatted image
            recognition_result = gesture_recognizer.recognize(mp_image)

            # Check if any gestures were recognized
            if recognition_result.gestures and recognition_result.gestures[0]:
                top_gesture = recognition_result.gestures[0][0]
                gesture_name = top_gesture.category_name

                detected_gesture = True  # Set the flag since a gesture is detected

                # Validate the gesture based on the current mode
                if is_gesture_valid(gesture_name, mode):
                    cv2.putText(frame, f'Gesture: {gesture_name} ({top_gesture.score:.2f})',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Update the last detected gesture
                    last_detected_gesture = gesture_name

    # If no gestures are detected and the previous frame had a detected gesture,
    # log the gesture and process the code sequence.
    if not detected_gesture and previously_detected_gesture and last_detected_gesture:
        if mode == "morse":
            if last_detected_gesture == "1":
                morse_code.append(".")
            elif last_detected_gesture == "2":
                morse_code.append("-")
            elif last_detected_gesture == "O":
                process_morse_code()

        elif mode == "braille":
            if last_detected_gesture in braille_classes:
                if last_detected_gesture != "O":
                    index = int(last_detected_gesture) - 1
                    braille_code[index] = "1"
                elif last_detected_gesture == "O":
                    process_braille_code()

        # Clear the last detected gesture
        last_detected_gesture = None

    # Update the previous frame's gesture detection status
    previously_detected_gesture = detected_gesture

    # Display the current mode on the frame
    cv2.putText(frame, f"Mode: {mode.capitalize()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check for the latest recognized command
    if recognized_command:
        if "switch please" in recognized_command:
            mode = "morse"
            speak_text("Switched to Morse mode")
            print("Switched to Morse mode")
        elif "please switch" in recognized_command:
            mode = "braille"
            speak_text("Switched to Braille mode")
            print("Switched to Braille mode")
        recognized_command = None  # Clear the command after processing

    # Display the processed frame
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
stream.stop_stream()
stream.close()
p.terminate()