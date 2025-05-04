# Trash Classifier API

<<<<<<< HEAD
API สำหรับทำนายประเภทขยะจากภาพถ่ายหรือวิดีโอสตรีมมิ่งแบบเรียลไทม์

## วิธีใช้งาน

- `/predict` – ส่งภาพถ่ายและรับผลการทำนาย
- `/video_feed` – สตรีมวิดีโอพร้อมผลการทำนายแบบเรียลไทม์

## Deploy บน Render

คลิกปุ่มด้านล่างเพื่อ Deploy โปรเจกต์นี้บน Render.com:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/PON-2003/APITEST)
=======
This is a Flask API to classify waste (cardboard, metal, glass, etc.) using a trained `.h5` model and OpenCV.

## Endpoints
- `/predict` – Snapshot + JSON prediction (with base64 image)
- `/video_feed` – Real-time MJPEG video stream with predictions

## Deploy to Render

Click below to deploy instantly on Render:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/trash-classifier-api)
>>>>>>> 43e9248
