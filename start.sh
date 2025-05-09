#!/bin/bash
# อัปเดต pip ให้เป็นเวอร์ชันล่าสุด
python -m pip install --upgrade pip

# ติดตั้ง dependencies จาก requirements.txt
pip install -r requirements.txt

# เริ่มการทำงานของแอป Flask ด้วย gunicorn (timeout 90 วินาที)
exec gunicorn APIBroo:app --bind 0.0.0.0:$PORT --timeout 90
