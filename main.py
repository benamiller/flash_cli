import os
import time

import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io.wavfile import read

from playsound import playsound

fs = 44100
seconds = 10

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write("output.wav", fs, recording)

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
)

audio_file = genai.upload_file(path="output.wav")

while audio_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    audio_file = genai.get_file(audio_file.name)

if audio_file.state.name == "FAILED":
    raise ValueError(audio_file.state.name)

response = model.generate_content(
    ["You are in conversation with an individual. Please respond to their speech, in any way they request", audio_file])

print(response.text)
tts = gTTS(text=response.text, lang='en')
tts.save("response.wav")

playsound("response.wav")
