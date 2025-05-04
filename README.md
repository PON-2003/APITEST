# Trash Classifier API

This is a Flask API to classify waste (cardboard, metal, glass, etc.) using a trained `.h5` model and OpenCV.

## Endpoints
- `/predict` – Snapshot + JSON prediction (with base64 image)
- `/video_feed` – Real-time MJPEG video stream with predictions

## Deploy to Render

Click below to deploy instantly on Render:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/trash-classifier-api)
