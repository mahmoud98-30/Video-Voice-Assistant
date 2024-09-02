import os
import re
import threading
import time

import cv2
import pyperclip
import pyttsx3
from PIL import ImageGrab

from transcribe import whisper_model

web_cam = cv2.VideoCapture(0)


class ThreadedWebcam:
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.saved_frame_path = "webcam_capture.jpg"
        self.display_thread = None

    def start(self):
        if not self.webcam.isOpened():
            print("Error: Could not open webcam")
            return

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()

        self.saving_thread = threading.Thread(target=self._frame_saving_loop)
        self.saving_thread.start()

        print("Webcam started in background")

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.display_thread:
            self.display_thread.join()
        self.webcam.release()
        cv2.destroyAllWindows()
        print("Webcam stopped")

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.webcam.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                self.is_running = False
                break

            # Flip the frame horizontally
            # frame = cv2.flip(frame, 1)

            with self.lock:
                self.latest_frame = frame

    def _frame_saving_loop(self):
        while self.is_running:
            self.save_latest_frame(self.saved_frame_path)
            time.sleep(5)  # Save the frame every second

    def get_latest_frame(self):
        time.sleep(0.5)  # Short sleep to prevent busy-waiting
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def save_latest_frame(self, filename):
        frame = self.get_latest_frame()
        if frame is not None:
            # Delete the old file if it exists
            if os.path.exists(filename):
                os.remove(filename)

            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        else:
            print("No frame available to save")

    def display_stream(self):
        while self.is_running:
            frame = self.get_latest_frame()
            if frame is not None:
                cv2.imshow('Webcam Stream', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)  # Short sleep to prevent busy-waiting

        self.stop()

    def is_display_thread_alive(self):
        return self.display_thread is not None and self.display_thread.is_alive()


def take_screenshot():
    path = "screenshot.jpg"
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path)



def web_cam_capture(webcam: ThreadedWebcam):

    if webcam.is_display_thread_alive():
        print("Webcam is already running")
        return
    webcam.start()

    display_thread = threading.Thread(target=webcam.display_stream)
    display_thread.start()



def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("Error: Could not extract clipboard content")
        return None


def speak(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties before adding anything to speak
    # You can change the rate, volume, and voice here
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    # Speak the text
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()


def wav_to_text(audio_path):
    print(f'wav_to_text Transcribing audio from {audio_path}...')

    segments, info = whisper_model.transcribe(audio_path, language='de') # 'en' for English and 'de' for German

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = ''.join([seg.text for seg in segments])
    print(f' wav_to_text Transcribed Text: {text}')
    return text


def extract_prompt(transcribed_text, wake_word=None):
    if not wake_word:
        return transcribed_text
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        print(f'extract_prompt: {prompt}')
        return prompt
