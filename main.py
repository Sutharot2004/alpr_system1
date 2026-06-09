from fastapi import FastAPI, File, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from difflib import SequenceMatcher
import cv2
import numpy as np
import base64
import sqlite3
from datetime import datetime
import json
import asyncio
import math

app = FastAPI()
templates = Jinja2Templates(directory="templates")

plate_model = None
ocr_model = None
class_names = None

DB_NAME = "parking.db"
connected_clients = []
last_seen = {}
# =============================
# Mapping Dictionary
# =============================

CLASS_MAP = {
    # digits
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "A01": "ก",
    "A02": "ข",
    "A04": "ค",
    "A05": "ฅ",
    "A06": "ฆ",
    "A07": "ง",
    "A08": "จ",
    "A09": "ฉ",
    "A10": "ช",
    "A12": "ฌ",
    "A13": "ญ",
    "A14": "ฎ",
    "A16": "ฐ",
    "A18": "ฒ",
    "A19": "ณ",
    "A20": "ด",
    "A21": "ต",
    "A22": "ถ",
    "A23": "ท",
    "A24": "ธ",
    "A25": "น",
    "A26": "บ",
    "A28": "ผ",
    "A30": "พ",
    "A31": "ฟ",
    "A32": "ภ",
    "A33": "ม",
    "A34": "ย",
    "A35": "ร",
    "A36": "ล",
    "A37": "ว",
    "A38": "ศ",
    "A39": "ษ",
    "A40": "ส",
    "A41": "ห",
    "A42": "ฬ",
    "A43": "อ",
    "A44": "ฮ",
    # จังหวัด
    "ACR": "อำนาจเจริญ",
    "ATG": "อ่างทอง",
    "AYA": "พระนครศรีอยุธยา",
    "BKK": "กรุงเทพมหานคร",
    "BKN": "บึงกาฬ",
    "BRM": "บุรีรัมย์",
    "BTG": "ปทุมธานี",
    "CBI": "ชลบุรี",
    "CCO": "ฉะเชิงเทรา",
    "CMI": "เชียงใหม่",
    "CNT": "ชัยนาท",
    "CPM": "ชัยภูมิ",
    "CPN": "ชุมพร",
    "CRI": "เชียงราย",
    "CTI": "จันทบุรี",
    "KBI": "กระบี่",
    "KKN": "ขอนแก่น",
    "KPT": "กำแพงเพชร",
    "KRI": "กาญจนบุรี",
    "KSN": "กาฬสินธุ์",
    "LEI": "เลย",
    "LPG": "ลำปาง",
    "LPN": "ลำพูน",
    "LRI": "ลพบุรี",
    "MDH": "มุกดาหาร",
    "MKM": "มหาสารคาม",
    "MSN": "แม่ฮ่องสอน",
    "NAN": "น่าน",
    "NBI": "นนทบุรี",
    "NBP": "หนองบัวลำภู",
    "NKI": "หนองคาย",
    "NMA": "นครราชสีมา",
    "NPM": "นครพนม",
    "NPT": "นครปฐม",
    "NSN": "นครสวรรค์",
    "NST": "นครศรีธรรมราช",
    "NYK": "นครนายก",
    "PBI": "เพชรบุรี",
    "PCT": "พิจิตร",
    "PKN": "ประจวบคีรีขันธ์",
    "PKT": "ภูเก็ต",
    "PLG": "พัทลุง",
    "PLK": "พิษณุโลก",
    "PNA": "พังงา",
    "PNB": "ปราจีนบุรี",
    "PRE": "แพร่",
    "PRI": "ปัตตานี",
    "PTE": "พะเยา",
    "PTN": "ปทุมธานี",
    "PYO": "พะเยา",
    "RBR": "ราชบุรี",
    "RET": "ร้อยเอ็ด",
    "RNG": "ระนอง",
    "RYG": "ระยอง",
    "SBR": "สระบุรี",
    "SKA": "สงขลา",
    "SKM": "สมุทรสงคราม",
    "SKN": "สมุทรสาคร",
    "SKW": "สระแก้ว",
    "SNI": "สุราษฎร์ธานี",
    "SNK": "สกลนคร",
    "SPB": "สุพรรณบุรี",
    "SPK": "สมุทรปราการ",
    "SRI": "สิงห์บุรี",
    "SRN": "สุรินทร์",
    "SSK": "ศรีสะเกษ",
    "STI": "สุโขทัย",
    "TAK": "ตาก",
    "TRG": "ตรัง",
    "TRT": "ตราด",
    "UBN": "อุบลราชธานี",
    "UDN": "อุดรธานี",
    "UTI": "อุทัยธานี",
    "UTT": "อุตรดิตถ์",
    "YLA": "ยะลา",
    "YST": "ยโสธร",
}
# "A03": "",   # ฃ (งดใช้)
# "A11": "",   # ซ (เว้นไว้)
# "A15": "",   # ฏ (เว้นไว้)
# "A17": "",   # ฑ (เว้นไว้)
# "A27": "",   # ป (เว้นไว้)
# "A29": "",   # ฝ (เว้นไว้)


# =============================
@app.on_event("startup")
def startup():
    load_models()
    init_db()


def load_models():
    global car_model, motor_model, ocr_model, class_names
    car_model = YOLO("runs/detect/car_plate_v1/weights/best.pt")
    ocr_model = YOLO("runs/detect/ocr_v1/weights/best.pt")
    class_names = ocr_model.names


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def get_best_detection(results):
    best_conf = 0
    best_box = None

    for r in results:
        if len(r.boxes) == 0:
            continue

        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            if float(conf) > best_conf:
                best_conf = float(conf)
                best_box = box.cpu().numpy()

    return best_conf, best_box


def detect_plate_stable(img, attempts=2):
    best_conf = 0
    best_box = None
    for _ in range(attempts):
        results = car_model(img, conf=0.25, verbose=False)
        conf, box = get_best_detection(results)
        if box is not None and conf > best_conf:
            best_conf = conf
            best_box = box

    return best_conf, best_box


def deskew_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return four_point_transform(image, pts)

    return image


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def pca_align_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # ป้องกันหมุนแรงเกินไปจาก noise
    if abs(angle) > 20:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_plate_for_ocr(image):
    # upscale เพื่อเพิ่มความคมของตัวอักษรเล็ก
    h, w = image.shape[:2]
    scale = 2.0 if min(h, w) < 160 else 1.5
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # ลด noise โดยรักษาขอบ
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=55, sigmaSpace=55)

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # เพิ่ม local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # sharpen เบา ๆ
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.4, blurred, -0.4, 0)

    # OCR model รับภาพสี: แปลงกลับ BGR เพื่อไม่แตะ pipeline เดิม
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def decode_ocr_items(ocr_results):
    items = []

    for ocr_r in ocr_results:
        boxes = ocr_r.boxes.xyxy.cpu().numpy()
        classes = ocr_r.boxes.cls.cpu().numpy()
        confs = ocr_r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            x1c, y1c, x2c, y2c = box
            items.append(
                {
                    "x_center": float((x1c + x2c) / 2),
                    "y_center": float((y1c + y2c) / 2),
                    "box": box,
                    "class_id": int(cls_id),
                    "conf": float(conf),
                }
            )

    return items


def split_rows(items):
    if len(items) < 2:
        return items, []

    y_centers = np.array([[item["y_center"]] for item in items])

    # cluster เป็น 2 กลุ่ม
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(y_centers)

    row1 = []
    row2 = []

    for item, label in zip(items, labels):
        if label == 0:
            row1.append(item)
        else:
            row2.append(item)

    # row ที่อยู่สูงกว่าคือ top_row
    mean1 = np.mean([i["y_center"] for i in row1])
    mean2 = np.mean([i["y_center"] for i in row2])

    if mean1 < mean2:
        top_row = row1
        bottom_row = row2
    else:
        top_row = row2
        bottom_row = row1

    # sort left → right
    top_row = sorted(top_row, key=lambda x: x["x_center"])
    bottom_row = sorted(bottom_row, key=lambda x: x["x_center"])

    return top_row, bottom_row


def label_from_item(item):
    name = class_names[item["class_id"]]
    return CLASS_MAP.get(name, name)


def split_top_row_blocks(top_row):
    if len(top_row) <= 1:
        return top_row, []

    x_centers = [item["x_center"] for item in top_row]
    widths = [max(1.0, float(item["box"][2] - item["box"][0])) for item in top_row]
    gaps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]

    if not gaps:
        return top_row, []

    split_idx = int(np.argmax(gaps))
    median_width = float(np.median(widths))

    # ช่องว่างใหญ่ระหว่างหมวดอักษร (ซ้าย) และเลขทะเบียน (ขวา)
    if gaps[split_idx] > (0.8 * median_width):
        return top_row[: split_idx + 1], top_row[split_idx + 1 :]

    return top_row, []


def normalize_car_series_labels(left_labels):
    confusable = {
        "พ": "พ",
        "ป": "ป",
        "ต": "ต",
        "ฒ": "ฒ",
    }

    normalized = []
    for label in left_labels:
        if label in confusable:
            normalized.append(confusable[label])
        else:
            normalized.append(label)

    return normalized


def parse_car_top_row(top_row):
    left_block, right_block = split_top_row_blocks(top_row)

    left_labels = [label_from_item(item) for item in left_block]
    right_labels = [label_from_item(item) for item in right_block]

    left_labels = normalize_car_series_labels(left_labels)

    # ถ้า split ไม่ชัด ให้ fallback parsing แบบเดิม
    if not right_labels:
        series = []
        numbers = []

        i = 0
        n = len(left_labels)

        if i < n and left_labels[i].isdigit():
            series.append(left_labels[i])
            i += 1

        while i < n and not left_labels[i].isdigit():
            series.append(left_labels[i])
            i += 1

        while i < n:
            numbers.append(left_labels[i])
            i += 1

        return "".join(series) + "".join(numbers), len(left_labels)

    # โครงสร้างป้ายรถยนต์: ซ้าย (หมวดอักษร/เลขนำ) + ขวา (เลข 2-4 หลัก)
    series_text = "".join(left_labels)
    right_numbers = [ch for ch in right_labels if ch.isdigit()]
    number_text = "".join(right_numbers)

    if len(number_text) < 2:
        # เลขขวาหาย ให้ fallback เดิมเพื่อลด false correction
        flat_labels = [label_from_item(item) for item in top_row]
        series = []
        numbers = []

        i = 0
        n = len(flat_labels)

        if i < n and flat_labels[i].isdigit():
            series.append(flat_labels[i])
            i += 1

        while i < n and not flat_labels[i].isdigit():
            series.append(flat_labels[i])
            i += 1

        while i < n:
            numbers.append(flat_labels[i])
            i += 1

        return "".join(series) + "".join(numbers), len(flat_labels)
    return series_text + number_text, len(left_labels) + len(right_labels)


def run_ocr_with_fallback(ocr_input, plate_type):
    primary = ocr_model(ocr_input, conf=0.30)
    primary_items = decode_ocr_items(primary)
    primary_top, primary_bottom = split_rows(primary_items)

    avg_conf = np.mean([item["conf"] for item in primary_items]) if primary_items else 0

    if plate_type == "car" and avg_conf < 0.5 and len(primary_items) < 6:
        retry = ocr_model(ocr_input, conf=0.22)
        retry_items = decode_ocr_items(retry)
        retry_top, retry_bottom = split_rows(retry_items)

        if len(retry_top) > len(primary_top):
            return retry_items, retry_top, retry_bottom

    return primary_items, primary_top, primary_bottom


@app.post("/detect_plate")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # ลด resolution เพื่อให้เร็ว
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    img_small = cv2.resize(img, (int(w * scale), int(h * scale)))

    car_results = car_model(img_small, conf=0.30)
    car_conf, car_box = get_best_detection(car_results)

    if car_box is None:
        return {"detected": False}

    return {"detected": True, "type": "car", "confidence": car_conf}


def clean_items(items):
    # sort confidence สูงสุดก่อน
    items_sorted = sorted(items, key=lambda x: x["conf"], reverse=True)
    filtered = []
    for item in items_sorted:
        keep = True
        for f in filtered:
            if (
                abs(item["x_center"] - f["x_center"]) < 8
                and abs(item["y_center"] - f["y_center"]) < 8
            ):
                keep = False
                break

        if keep:
            filtered.append(item)

    return filtered


def robust_parse(items):
    # sort จากซ้ายไปขวา
    items_sorted = sorted(items, key=lambda x: x["x_center"])
    labels = [label_from_item(i) for i in items_sorted]
    province_labels = []
    main_labels = []
    for label in labels:

        # จังหวัด
        if len(label) > 2 and not label.isdigit():
            province_labels.append(label)

        else:
            main_labels.append(label)

    province = "".join(province_labels)

    if not main_labels:
        return "", province

    # -------------------------
    # แยก digit กับ letter
    # -------------------------
    digits = []
    letters = []

    for ch in main_labels:

        if ch.isdigit():
            digits.append(ch)

        else:
            letters.append(ch)

    # -------------------------
    # build plate string
    # -------------------------

    # กรณีมี digit นำหน้า เช่น 1กข
    prefix_digit = ""

    if len(digits) >= 1 and main_labels[0].isdigit():
        prefix_digit = digits[0]
        digits = digits[1:]

    # จำกัดตัวอักษรไม่เกิน 2 ตัว
    letters = letters[:2]

    # เลขท้ายไม่เกิน 4 ตัว
    digits = digits[:4]
    series_number = prefix_digit + "".join(letters) + "".join(digits)
    return series_number, province


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect plate (stable version)
    car_conf, car_box = detect_plate_stable(img)

    if car_box is None:
        return {"plates": []}

    selected_box = car_box
    plate_type = "car"

    # Crop plate
    x1, y1, x2, y2 = map(int, selected_box)
    h_img, w_img = img.shape[:2]

    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    plate_crop = img[y1:y2, x1:x2]
    # resize ก่อน
    plate_crop = cv2.resize(plate_crop, (320, 160))

    # Preprocess
    try:
        plate_crop = deskew_plate(plate_crop)
    except:
        pass

    try:
        plate_crop = pca_align_plate(plate_crop)
    except:
        pass

    try:
        plate_crop = enhance_plate_for_ocr(plate_crop)
    except:
        pass

    # OCR
    items, _, _ = run_ocr_with_fallback(plate_crop, plate_type)

    if not items:
        return {"plates": []}

    # Parsing
    items = clean_items(items)
    series_number, province = robust_parse(items)

    # Encode images
    _, buffer = cv2.imencode(".jpg", plate_crop)
    plate_img_base64 = base64.b64encode(buffer).decode("utf-8")
    _, full_buffer = cv2.imencode(".jpg", img)
    full_img_base64 = base64.b64encode(full_buffer).decode("utf-8")

    # Parking logic
    status, duration, fee = await process_parking(
        series_number, province, plate_img_base64, full_img_base64
    )

    # Return
    return {
        "plates": [
            {
                "type": "car",
                "series_number": series_number,
                "province": province,
                "status": status,
                "duration_minutes": duration,
                "fee": fee,
                "preview": plate_img_base64,
            }
        ]
    }


# ถึงนี่นะ


def calculate_fee(entry_time_str, exit_time_str):
    entry = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
    exit = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")

    duration = exit - entry
    minutes = int(duration.total_seconds() / 60)

    if minutes <= 30:
        fee = 0
    else:
        fee = minutes - 30  # นาทีเกิน คิดนาทีละ 1 บาท

    return minutes, fee


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast(message: dict):
    data = json.dumps(message, ensure_ascii=False)
    for client in connected_clients[:]:
        try:
            await client.send_text(data)
        except:
            connected_clients.remove(client)


async def process_parking(plate, province, plate_img_b64, full_img_b64):
    plate = normalize_plate(plate)
    now_ts = datetime.now().timestamp()
    # ป้องกันกล้องยิงซ้ำ
    if plate in last_seen and now_ts - last_seen[plate] < 4:
        return "SKIP", 0, 0
    last_seen[plate] = now_ts
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, plate, entry_time
        FROM parking_records
        WHERE exit_time IS NULL
        ORDER BY entry_time DESC
    """
    )

    rows = c.fetchall()
    best_match = None
    best_score = 0

    for r in rows:
        record_id, db_plate, entry_time = r
        score = similar(plate, db_plate)
        if score > best_score:
            best_score = score
            best_match = r

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ถ้า similarity สูง ถือว่าเป็นรถเดิม
    if best_match and best_score > 0.65:
        record_id, db_plate, entry_time = best_match
        minutes, fee = calculate_fee(entry_time, now)
        c.execute(
            """
            UPDATE parking_records
            SET exit_time=?, exit_image=?, fee=?, duration_minutes=?
            WHERE id=?
        """,
            (now, full_img_b64, fee, minutes, record_id),
        )

        conn.commit()
        conn.close()

        await broadcast(
            {
                "type": "event",
                "status": "EXIT",
                "plate": db_plate,
                "province": province,
                "fee": fee,
            }
        )

        await broadcast(get_summary())
        return "EXIT", minutes, fee

    else:

        c.execute(
            """
            INSERT INTO parking_records
            (plate, province, entry_time, entry_image, plate_image)
            VALUES (?, ?, ?, ?, ?)
        """,
            (plate, province, now, full_img_b64, plate_img_b64),
        )

        conn.commit()
        conn.close()

        await broadcast(
            {"type": "event", "status": "ENTRY", "plate": plate, "province": province}
        )

        await broadcast(get_summary())

        return "ENTRY", 0, 0


@app.get("/records")
def get_records():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        SELECT plate, province, entry_time, exit_time, duration_minutes, fee
        FROM parking_records
        ORDER BY entry_time DESC
    """
    )

    rows = c.fetchall()
    conn.close()

    records = []
    for r in rows:
        plate, province, entry_time, exit_time, duration, fee = r
        status = "ENTRY" if exit_time is None else "EXIT"
        records.append(
            {
                "plate": plate,
                "province": province,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration": duration,
                "fee": fee,
                "status": status,
            }
        )

    return {"records": records}


def get_summary():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute(
        "SELECT COUNT(*) FROM parking_records WHERE entry_time LIKE ?", (f"{today}%",)
    )
    entries = c.fetchone()[0]

    c.execute(
        "SELECT COUNT(*) FROM parking_records WHERE exit_time LIKE ?", (f"{today}%",)
    )
    exits = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM parking_records WHERE exit_time IS NULL")
    parked = c.fetchone()[0]

    c.execute(
        "SELECT SUM(fee) FROM parking_records WHERE exit_time LIKE ? AND fee IS NOT NULL",
        (f"{today}%",),
    )
    money = c.fetchone()[0] or 0

    conn.close()

    return {
        "type": "summary",
        "in": entries,
        "out": exits,
        "park": parked,
        "money": money,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        summary = get_summary()
        await websocket.send_text(json.dumps(summary, ensure_ascii=False))
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def normalize_plate(p: str):
    return p.replace(" ", "").strip()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()