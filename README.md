--สร้าง python env--
python -m venv .venv
--เปิดใช้งาน--
.\.venv\Scripts\activate
--ติดตั้ง PyTorch GPU--
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
--เช็คว่าใช้ GPU--
python -c "import torch; print(torch.cuda.is_available())"
--ติดตั้ง library อื่น ๆ--
pip install -r requirements.txt
--คำสั่งรันserver--
uvicorn main:app
--เข้าทดสอบ--
http://127.0.0.1:8000/


--คำสั่งเทรนโมเดล--
==Car Plate Detection Model==
yolo detect train model=yolov8s.pt data=dataset_car/data.yaml epochs=60 imgsz=640 batch=8 name=car_plate_v1

==Motorcycle Plate Detection Model==
yolo detect train model=yolov8s.pt data=dataset_motorcycle/data.yaml epochs=150 imgsz=768 batch=8 patience=30 workers=4 name=motorcycle_plate_v1

==OCR Model==
yolo detect train model=yolov8s.pt data=dataset_ocr/data.yaml epochs=30 imgsz=640 batch=8 name=ocr_v1

