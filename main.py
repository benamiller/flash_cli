import os
import time
import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS

import sounddevice as sd
from scipy.io.wavfile import write

from playsound import playsound

import numpy as np

fs = 44100
channels = 1
recording = []


def callback(indata, frames, time, status):
    if status:
        print(status)
    recording.append(indata.copy())


with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
    print("Recording...")
    while True:
        key = input("Press <space> to stop recording: ")
        if key == ' ':
            break

recording = np.concatenate(recording, axis=0)

current_date = str(datetime.datetime.now())

write("output_" + current_date + ".wav", fs, recording)

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
)

audio_file = genai.upload_file(path="output_" + current_date + ".wav")

while audio_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    audio_file = genai.get_file(audio_file.name)

if audio_file.state.name == "FAILED":
    raise ValueError(audio_file.state.name)

response = model.generate_content(
    ["You are in conversation with an individual. Please respond to their speech, in any way they request, in a somewhat concise manner", audio_file])

print(response.text)
tts = gTTS(text=response.text, lang='en')
tts.save("response.wav")

playsound("response.wav")
