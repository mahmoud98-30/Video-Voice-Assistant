import os

from groq import Groq
from dotenv import load_dotenv

from llm.prompt import groq_convo, call_prompt_sys


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=groq_api_key)


def groq_llm(prompt, img_context):
    print(f'Prompt: {prompt}')
    print(f'Img Context: {img_context}')
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'
    groq_convo.append({'role': 'user', 'content': prompt})
    chat_complete = groq_client.chat.completions.create(messages=groq_convo, model='llama3-70b-8192')
    response = chat_complete.choices[0].message
    groq_convo.append({'role': 'system', 'content': response.content})
    return response.content


def call_llm(prompt):
    function_convo = [{'role': 'system', 'content': call_prompt_sys},
                      {'role': 'user', 'content': prompt}]
    chat_complete = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_complete.choices[0].message
    return response.content
