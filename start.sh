#!/bin/bash
# อัปเดต pip ให้เป็นเวอร์ชันล่าสุด
pip install --upgrade pip

# ติดตั้ง dependencies จาก requirements.txt
pip install -r requirements.txt

# เริ่มการทำงานของแอป
gunicorn APIBroo:app --timeout 90
