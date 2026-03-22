from flask import Flask, request, jsonify
import tensorflow_hub as hub
import librosa
import numpy as np
import whisper
import os
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Load YAMNet
print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet_model.class_map_path().numpy()
class_names = [line.strip().split(",")[-1] for line in open(class_map_path)]
print("YAMNet loaded!")

# Load Whisper
print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper loaded!")

danger_keywords = ["scream", "shout", "yell", "cry", "help", "attack", "follow", "scared", "save me"]


@app.route("/analyze", methods=["POST"])
def analyze():

    # ✅ Check file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # ✅ Save with unique name (avoid overwrite)
    filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join("uploads", filename)

    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        # 🔊 YAMNet
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

        # 🧠 Whisper
        result = whisper_model.transcribe(file_path)
        text = result["text"].lower()

        # 🚨 Text danger detection
        text_danger = any(word in text for word in danger_keywords)
        print(text_danger)

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
            "overall_risk_level": overall_risk_level
        })

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

    # finally:
    #     # ✅ Cleanup file
    #     if os.path.exists(file_path):
    #         os.remove(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)