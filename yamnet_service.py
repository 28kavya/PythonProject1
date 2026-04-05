import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, jsonify
import tensorflow_hub as hub
import librosa
import numpy as np
import whisper
import uuid
import tensorflow as tf
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# 🔥 GLOBAL VARIABLES (lazy loading)
yamnet_model = None
class_names = None
whisper_model = None

danger_keywords = ["scream", "shout", "yell", "cry", "help", "attack", "follow", "scared", "save me"]


# ✅ Lazy load YAMNet
def load_yamnet():
    global yamnet_model, class_names
    if yamnet_model is None:
        print("Loading YAMNet...")
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = yamnet_model.class_map_path().numpy()
        class_names = [line.strip().split(",")[-1] for line in open(class_map_path)]
        print("YAMNet loaded!")
    return yamnet_model, class_names


# ✅ Lazy load Whisper
def load_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper...")
        whisper_model = whisper.load_model("base")
        print("Whisper loaded!")
    return whisper_model


@app.route("/analyze", methods=["POST"])
def analyze():

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join("uploads", filename)

    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        # 🔥 Load models ONLY when needed
        yamnet_model, class_names = load_yamnet()
        whisper_model = load_whisper()

        # 🔊 YAMNet processing
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

            yamnet_labels.append({"label": label, "score": score})

            if any(word in label for word in ["scream", "shout", "yell", "cry"]):
                if score > 0.15:
                    ai_detected = True

        amplitude = float(np.max(np.abs(waveform)))

        # 🧠 Whisper processing
        result = whisper_model.transcribe(file_path)
        text = result["text"].lower()

        text_danger = any(word in text for word in danger_keywords)

        # ⚖️ Decision logic
        if (ai_detected and amplitude > 0.25) or text_danger:
            alert_triggered = True
            alert_reason = "Suspicious audio detected"
            overall_risk_level = "HIGH"
        else:
            alert_triggered = False
            alert_reason = "Normal"
            overall_risk_level = "LOW"

        return jsonify({
            "alert_triggered": alert_triggered,
            "alert_reason": alert_reason,
            "overall_risk_level": overall_risk_level,
            "amplitude": amplitude,
            "text": text,
            "text_danger": text_danger,
            "yamnet_detected": ai_detected,
            "yamnet_predictions": yamnet_labels
        })

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ✅ Required for local run (Render uses Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
