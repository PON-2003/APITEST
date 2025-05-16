from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
import time

app = Flask(__name__)
CORS(app)

interpreter = None
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# โหลดโมเดลตอนเริ่มแอป
def load_model():
    global interpreter
    try:
        print("[INFO] Loading model...")
        model_path = os.path.join("model", "trash_classifier_model.tflite")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("[INFO] Model loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        interpreter = None

load_model()

# แปลง base64 เป็น OpenCV image
def decode_image(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Image decoding failed: {e}")
        return None

# แปลง OpenCV image เป็น base64
def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# ฟังก์ชันทำนายพร้อมวาดกรอบและ label
def predict_on_frame(frame):
    h, w, _ = frame.shape
    box_size = 224
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # วาดกรอบสีเขียวรอบ ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]

    img = cv2.resize(roi, (target_width, target_height)) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    class_label = class_names[class_index] if confidence >= 0.70 else "No"

    # วาด label ลงบนภาพ
    label_text = f"{class_label} ({confidence:.2f})" if class_label != "No" else "Uncertain"
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return class_label, confidence, frame

# Endpoint ทำนาย
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if interpreter is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    img_base64 = data.get("image")

    if not img_base64:
        return jsonify({"error": "No image provided"}), 400

    frame = decode_image(img_base64)
    if frame is None:
        return jsonify({"error": "Invalid or corrupted image base64"}), 400

    try:
        label, confidence, framed_img = predict_on_frame(frame)
        framed_base64 = encode_image(framed_img)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

    elapsed = time.time() - start_time
    print(f"[INFO] Prediction completed in {elapsed:.2f}s")

    return jsonify({
        "class": label,
        "confidence": round(confidence, 4),
        "framed_image": framed_base64
    })

# หน้าหลัก
@app.route("/")
def home():
    return "<h1>Trash Classifier API</h1><p>Send POST to /predict with base64 image.</p>"

# เริ่มแอป
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
