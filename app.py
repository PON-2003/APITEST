from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import base64
import os

app = Flask(__name__)
CORS(app)

# โหลดโมเดล
model_path = os.path.join("model", "trash_classifier_model.h5")
model = tf.keras.models.load_model(model_path)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ฟังก์ชันแปลง base64 -> cv2 image
def decode_image(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Image decoding failed: {e}")
        return None

# ฟังก์ชันทำนายผล
def predict_on_frame(frame):
    if frame is None:
        raise ValueError("Empty frame received for prediction.")
    
    h, w, _ = frame.shape
    box_size = 224
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    roi = frame[y1:y1+box_size, x1:x1+box_size]

    try:
        img = cv2.resize(roi, (150, 150)) / 255.0
    except Exception as e:
        raise ValueError(f"Failed to resize image: {e}")

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = float(np.max(prediction))

    cv2.rectangle(frame, (x1, y1), (x1+box_size, y1+box_size), (0, 255, 0), 2)
    cv2.putText(frame, f"{class_label} ({confidence:.2f})", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, class_label, confidence

# API สำหรับรับภาพ base64 และทำนายผล
@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    img_base64 = data.get("image")
    
    if not img_base64:
        return jsonify({"error": "No image provided"}), 400

    frame = decode_image(img_base64)
    if frame is None:
        return jsonify({"error": "Invalid or corrupted image base64"}), 400

    try:
        frame, label, confidence = predict_on_frame(frame)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    _, buffer = cv2.imencode('.jpg', frame)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "class": label,
        "confidence": round(confidence, 4),
        "image_base64": result_base64
    })

@app.route("/")
def home():
    return "<h1>Trash Classifier API</h1><p>Send POST to /predict with base64 image.</p>"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
