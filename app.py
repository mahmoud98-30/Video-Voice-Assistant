import time
import speech_recognition as sr

from llm.llm import groq_llm, call_llm
from llm.vision_llm import vision_llm
from utils import wav_to_text, extract_prompt, take_screenshot, web_cam_capture, get_clipboard_text, speak, \
    ThreadedWebcam

r = sr.Recognizer()
m = sr.Microphone()

def callback(recognizer, audio):
    print('callback')
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text)

    if clean_prompt:
        print(f'User Prompt: {clean_prompt}')
        call = call_llm(clean_prompt)
        print(f'CALL {call}')
        if call == "take screenshot":
            print("Taking screenshot...")
            take_screenshot()
            vision_context = vision_llm(clean_prompt, "screenshot.jpg")
        elif call == "capture webcam":
            print("Capturing webcam...")
            webcam = ThreadedWebcam()
            web_cam_capture(webcam)
            image_path = r'D:/PycharmProjects/ChatBot/webcam_capture.jpg'
            vision_context = vision_llm(clean_prompt, webcam)
        elif call == "extract clipboard":
            print("Extracting clipboard text...")
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT {paste}'
            vision_context = None
        else:
            vision_context = None

        response = groq_llm(prompt=clean_prompt, img_context=vision_context)
        print(f'ASSISTANT {response}')
        speak(response)


def start_listening():
    with m as source:
        r.adjust_for_ambient_noise(source, duration=1)
    print(f"\nHow can i help you. \n")
    r.listen_in_background(source, callback)
    print('listen_in_background')

    while True:
        print("Listening ...")
        time.sleep(0.5)



if __name__ == "__main__":
    start_listening()