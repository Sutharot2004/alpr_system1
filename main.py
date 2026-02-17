from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

plate_model = None
ocr_model = None
class_names = None

# =============================
# Mapping Dictionary
# =============================

CLASS_MAP = {
    # ตัวเลขไม่ต้อง map
    "A01": "ก", "A02": "ข", "A04": "ค", "A05": "ง",
    "A06": "จ", "A07": "ฉ", "A08": "ช", "A09": "ซ",
    "A10": "ฌ", "A12": "ญ", "A13": "ฎ", "A14": "ฏ",
    "A16": "ฐ", "A18": "ฑ", "A19": "ฒ", "A20": "ณ",
    "A21": "ด", "A22": "ต", "A23": "ถ", "A24": "ท",
    "A25": "ธ", "A26": "น", "A28": "บ", "A30": "ป",
    "A31": "ผ", "A32": "ฝ", "A33": "พ", "A34": "ฟ",
    "A35": "ภ", "A36": "ม", "A37": "ย", "A38": "ร",
    "A39": "ล", "A40": "ว", "A41": "ศ", "A42": "ษ",
    "A43": "ส", "A44": "ห",

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
    "PTN": "ปทุมธานี",  # หมายเหตุ: บางระบบใช้ BTG หรือ PTN
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
    "YST": "ยโสธร"
    # เพิ่มจังหวัดอื่นตามต้องการ
}

# =============================

@app.on_event("startup")
def load_models():
    global car_model, motor_model, ocr_model, class_names

    car_model = YOLO("runs/detect/car_plate_v1/weights/best.pt")
    motor_model = YOLO("runs/detect/motorcycle_plate_v1/weights/best.pt")
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


def deskew_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4,2)
            return four_point_transform(image, pts)

    return image


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")

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

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 🔥 Run both models
    car_results = car_model(img, conf=0.3)
    motor_results = motor_model(img, conf=0.3)

    car_conf, car_box = get_best_detection(car_results)
    motor_conf, motor_box = get_best_detection(motor_results)

    if car_box is None and motor_box is None:
        return {"plates": []}

    # เลือกโมเดลที่มั่นใจสูงกว่า
    if car_conf >= motor_conf:
        selected_box = car_box
        plate_type = "car"
    else:
        selected_box = motor_box
        plate_type = "motorcycle"

    x1, y1, x2, y2 = map(int, selected_box)

    # padding 5%
    h_img, w_img = img.shape[:2]
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    plate_crop = img[y1:y2, x1:x2]

    # 🔥 Document scanning
    try:
        plate_crop = deskew_plate(plate_crop)
    except:
        pass

    # 🔥 OCR
    ocr_results = ocr_model(plate_crop, conf=0.30)

    items = []

    for ocr_r in ocr_results:
        boxes = ocr_r.boxes.xyxy.cpu().numpy()
        classes = ocr_r.boxes.cls.cpu().numpy()

        for b, c in zip(boxes, classes):
            x1c,y1c,x2c,y2c = b
            x_center = (x1c + x2c) / 2
            y_center = (y1c + y2c) / 2
            items.append((x_center, y_center, int(c), b))

    if not items:
        return {"plates": []}

    # 🔥 Layout parsing

    y_values = [item[1] for item in items]
    y_mean = np.mean(y_values)

    top_row = []
    bottom_row = []

    for item in items:
        if item[1] < y_mean:
            top_row.append(item)
        else:
            bottom_row.append(item)

    top_row = sorted(top_row, key=lambda x: x[0])
    bottom_row = sorted(bottom_row, key=lambda x: x[0])

    # ===== Parsing =====

    top_labels = []
    for item in top_row:
        name = class_names[item[2]]
        mapped = CLASS_MAP.get(name, name)
        top_labels.append(mapped)

    series = []
    numbers = []

    i = 0
    n = len(top_labels)

    if i < n and top_labels[i].isdigit():
        series.append(top_labels[i])
        i += 1

    while i < n and not top_labels[i].isdigit():
        series.append(top_labels[i])
        i += 1

    while i < n:
        numbers.append(top_labels[i])
        i += 1

    series_number = "".join(series) + "".join(numbers)

    province_parts = []
    for item in bottom_row:
        name = class_names[item[2]]
        mapped = CLASS_MAP.get(name, name)
        province_parts.append(mapped)

    province = "".join(province_parts)

    # encode preview
    _, buffer = cv2.imencode('.jpg', plate_crop)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "plates": [{
            "type": plate_type,
            "series_number": series_number,
            "province": province,
            "preview": img_base64
        }]
    }


