import datetime
import os

import google.generativeai as genai
import sounddevice as sd
from dotenv import load_dotenv
from scipy.io.wavfile import write

fs = 44100
seconds = 20
current_time = str(datetime.datetime.now())

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write("output" + current_time + ".wav", fs, recording)

with open("output" + current_time + ".wav", "rb") as audio_file:
    audio_data = audio_file.read()

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

response = chat_session.send_message(audio_data)

print(response.text)
print(chat_session.history)
