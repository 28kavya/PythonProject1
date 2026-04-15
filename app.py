import os
import uuid
import numpy as np
import re

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager

# ✅ IMPORTANT: set cache BEFORE imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub_cache"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_hub as hub
import librosa
import whisper

app = FastAPI()

# 🔥 Lazy loaded models
yamnet_model = None
whisper_model = None
class_names = None

danger_keywords = [
    "scream", "shout", "yell", "cry", "help",
    "attack", "follow", "scared", "save me"
]

# ✅ Load models only when needed
def load_models():
    global yamnet_model, whisper_model, class_names

    if yamnet_model is None:
        print("Loading YAMNet...")
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        class_map_path = yamnet_model.class_map_path().numpy()
        class_names = [
            line.strip().split(",")[-1].strip().strip('"')
            for line in open(class_map_path)
        ]
        print("YAMNet loaded!")

    if whisper_model is None:
        print("Loading Whisper...")
        whisper_model = whisper.load_model("tiny", download_root="/tmp/whisper")
        print("Whisper loaded!")

# ✅ Keyword detection
def contains_danger_keyword(text, keywords):
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()

    for keyword in keywords:
        keyword_words = keyword.lower().split()
        for i in range(len(words) - len(keyword_words) + 1):
            if words[i:i + len(keyword_words)] == keyword_words:
                return True
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔥 Loading models at startup...")
    load_models()
    yield

# ✅ create app AFTER lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"status": "AI backend running 🚀"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print("🔥 /analyze API HIT")
    os.makedirs("uploads", exist_ok=True)  # ✅ ensure folder exists
    file_ext = file.filename.split(".")[-1]
    file_path = f"uploads/{uuid.uuid4()}.{file_ext}"

    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        load_models()

        waveform, sr = librosa.load(file_path, sr=16000)
        scores, embeddings, spectrogram = yamnet_model(waveform)

        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_indices = np.argsort(mean_scores)[-5:][::-1]

        ai_detected = False
        yamnet_labels = []

        for i in top_indices:
            label = class_names[i].lower()
            score = float(mean_scores[i])

            yamnet_labels.append({
                "label": label,
                "score": score
            })

            if any(word in label for word in ["scream", "shout", "yell", "cry"]) and score > 0.15:
                ai_detected = True

        amplitude = float(np.max(np.abs(waveform)))

        text = ""
        text_danger = False

        if ai_detected or amplitude > 0.2:
            result = whisper_model.transcribe(file_path)
            text = result["text"]
            text_danger = contains_danger_keyword(text, danger_keywords)

        if (ai_detected and amplitude > 0.25) or text_danger:
            overall_risk_level = "HIGH"
            alert_triggered = True
            alert_reason = "Suspicious audio detected"
        else:
            overall_risk_level = "LOW"
            alert_triggered = False
            alert_reason = "Normal"

        return {
            "alert_triggered": alert_triggered,
            "alert_reason": alert_reason,
            "overall_risk_level": overall_risk_level,
            "transcription": text,
            "yamnet_labels": yamnet_labels
        }

    except Exception as e:
        return {"error": "Processing failed", "details": str(e)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)