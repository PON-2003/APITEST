from flask import Flask, jsonify, Response, request
from flask_cors import CORS  # เพิ่มการนำเข้า CORS
import cv2
import numpy as np
import tensorflow as tf
import base64
import os

app = Flask(__name__)
CORS(app)  # เปิดใช้งาน CORS สำหรับทุกเส้นทาง

model_path = os.path.join("model", "trash_classifier_model.h5")
model = tf.keras.models.load_model(model_path)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ฟังก์ชันที่ทำการแปลงภาพ base64 ที่รับมาจาก request
def decode_image(img_base64):
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# ฟังก์ชันในการทำนายจากภาพ
def predict_on_frame(frame):
    h, w, _ = frame.shape
    box_size = 224
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    roi = frame[y1:y1+box_size, x1:x1+box_size]

    img = cv2.resize(roi, (150, 150)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = float(np.max(prediction))

    cv2.rectangle(frame, (x1, y1), (x1+box_size, y1+box_size), (0, 255, 0), 2)
    cv2.putText(frame, f"{class_label} ({confidence:.2f})", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, class_label, confidence

# API สำหรับทำนายผลจากภาพที่ส่งมาผ่าน POST request
@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    img_base64 = data.get("image")
    if not img_base64:
        return jsonify({"error": "No image provided"}), 400
    
    frame = decode_image(img_base64)
    frame, label, confidence = predict_on_frame(frame)
    
    # แปลงภาพกลับเป็น base64 เพื่อส่งกลับไปยังผู้ใช้
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "class": label,
        "confidence": confidence,
        "image_base64": img_base64
    })

# API สำหรับส่งภาพวิดีโอเป็น live stream
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, _, _ = predict_on_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def home():
    return "<h1>Trash Classifier</h1><ul><li><a href='/predict'>Snapshot Predict</a></li><li><a href='/video_feed'>Live Video Feed</a></li></ul>"

import atexit
@atexit.register
def cleanup():
    cap.release()

if __name__ == "__main__":
    app.run(debug=True)
