import os
import time
from typing import Tuple

from PIL import Image
import google.generativeai as genai

from utils import ThreadedWebcam

genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE',
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              safety_settings=safety_settings,
                              generation_config=generation_config, )




def vision_llm(prompt: str, webcam: ThreadedWebcam) -> Tuple[str, str, str]:
    """
    Capture an image from the webcam, perform vision analysis, and process the result.

    Args:
        prompt (str): The user's prompt for image analysis.
        webcam (ThreadedWebcam): An instance of the ThreadedWebcam class.

    Returns:
        Tuple[str, str, str]: A tuple containing the OCR result, summary, and translation.
    """
    # Ensure the webcam is running
    if not webcam.is_running:
        webcam.start()

    # Capture and save the latest frame
    image_path = 'vision_llm_capture.jpg'
    webcam.save_latest_frame(image_path)

    # Vision analysis
    img = Image.open(image_path)
    vision_prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        f'assistant who will respond to the user \n\nUser Prompt: {prompt}'
    )
    response = model.generate_content([vision_prompt, img])
    return response.text

