import cv2
import numpy as np
import math

# ----------------------------------------------------------------------
# Вспомогательные функции геометрии
# ----------------------------------------------------------------------

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
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# ----------------------------------------------------------------------
# Логика поиска и улучшения
# ----------------------------------------------------------------------

def get_document_contour(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
    
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (width * height * 0.15):
            if cv2.isContourConvex(approx):
                return approx
    return None

def remove_shadows(image):
    """ Идеальное выравнивание белого фона (Division Normalization) """
    planes = cv2.split(image)
    result_planes = []
    for plane in planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        result_planes.append(norm)
    return cv2.merge(result_planes)

# ----------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ (ДЛЯ MAIN.PY)
# ----------------------------------------------------------------------

def process_document(image_bytes: bytes, scan_mode: str = "color", auto_crop: bool = True, enhance_level: str = "mild") -> tuple[np.ndarray, dict]:
    # Декодирование
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    working = image
    crop_applied = False
    
    # 1. Обрезка
    if auto_crop:
        contour = get_document_contour(image)
        if contour is not None:
            working = four_point_transform(image, contour.reshape(4, 2))
            crop_applied = True
        else:
            # Мягкая обрезка краев, если контур не найден
            h, w = image.shape[:2]
            working = image[int(h*0.02):int(h*0.98), int(w*0.02):int(w*0.98)]

    # 2. Улучшение качества
    if scan_mode == "bw":
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    else:
        # Убираем тени и выравниваем свет
        enhanced = remove_shadows(working)
        # Добавляем контраст
        alpha = 1.2 if enhance_level == "mild" else 1.5
        processed = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=-10)

    meta = {
        "auto_crop_applied": crop_applied,
        "mode": scan_mode,
        "size": [int(processed.shape[1]), int(processed.shape[0])]
    }
    
    return processed, meta
