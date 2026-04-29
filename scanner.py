import math
import cv2
import numpy as np

MAX_SIDE = 2000


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def resize_for_detection(image, max_side=MAX_SIDE):
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ----------------------------------------------------------------------
# Поиск документа: Canny + контуры
# ----------------------------------------------------------------------

def find_document_contour(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Сортируем по площади, берём самый крупный
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = img_area * 0.25   # документ должен занимать хотя бы 25% кадра

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        # Если не четырёхугольник – берём минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) >= min_area:
            return box.astype(np.float32)
    return None


# ----------------------------------------------------------------------
# Аффинное выравнивание (поворот + обрезка)
# ----------------------------------------------------------------------

def rotate_and_crop(image_bgr, contour):
    """
    Выравнивает документ по найденному контуру:
    - поворачивает так, чтобы документ встал горизонтально,
    - обрезает по границам документа после поворота.
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    # Нормализуем угол к диапазону [-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Расширяем холст, чтобы не обрезать углы после поворота
    h, w = image_bgr.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_corners = cv2.transform(corners.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
    x_min, y_min = np.min(new_corners, axis=0)
    x_max, y_max = np.max(new_corners, axis=0)
    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image_bgr, rot_mat, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    # На повёрнутом изображении ищем прямоугольник документа (бинаризация + контур)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(largest)
        # Добавляем небольшой отступ (2%)
        pad_x = int(rw * 0.02)
        pad_y = int(rh * 0.02)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        rw = min(rw + 2 * pad_x, rotated.shape[1] - x)
        rh = min(rh + 2 * pad_y, rotated.shape[0] - y)
        return rotated[y:y+rh, x:x+rw], True

    return rotated, True   # если не получилось обрезать, возвращаем хотя бы повёрнутое


# ----------------------------------------------------------------------
# Основная функция обрезки
# ----------------------------------------------------------------------

def crop_document_if_found(image_bgr):
    """
    Надёжная обрезка документа:
    - сначала ищем контур через Canny,
    - если не найден, пробуем найти через адаптивную бинаризацию (fallback).
    """
    # Основной метод
    contour = find_document_contour(image_bgr)
    if contour is not None:
        cropped, ok = rotate_and_crop(image_bgr, contour)
        if ok and cropped.shape[0] > 100 and cropped.shape[1] > 100:
            return cropped, True, "canny"

    # Fallback: ищем светлую область на тёмном фоне (для фото бумаги на столе)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > gray.shape[0] * gray.shape[1] * 0.25:
            x, y, w, h = cv2.boundingRect(largest)
            crop = image_bgr[y:y+h, x:x+w].copy()
            if crop.shape[0] > 100 and crop.shape[1] > 100:
                return crop, True, "otsu_fallback"

    # Ничего не найдено – возвращаем исходное
    return image_bgr, False, "none"


# ----------------------------------------------------------------------
# Функции улучшения (мягкие)
# ----------------------------------------------------------------------

def unsharp_mask(image, sigma=1.0, strength=0.8):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def normalize_illumination(gray):
    h, w = gray.shape[:2]
    kernel = max(31, (min(h, w) // 6) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.divide(gray, background, scale=255)


def enhance_color_scan(image_bgr, enhance_level="mild"):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    if enhance_level == "mild":
        clip_limit = 0.5
        d, sigma_c, sigma_s = 3, 8, 8
        us_sigma, us_str = 0.25, 0.15
    elif enhance_level == "normal":
        clip_limit = 1.0
        d, sigma_c, sigma_s = 5, 15, 15
        us_sigma, us_str = 0.4, 0.35
    else:  # strong
        clip_limit = 2.0
        d, sigma_c, sigma_s = 7, 25, 25
        us_sigma, us_str = 0.6, 0.7

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_chan = clahe.apply(norm)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s)
    enhanced = unsharp_mask(enhanced, sigma=us_sigma, strength=us_str)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255)).astype(np.float32) / 255.0
    paper_mask = cv2.GaussianBlur(paper_mask, (9, 9), 0)[..., None]
    white_bg = np.full_like(enhanced, 247, dtype=np.float32)
    mixed = enhanced.astype(np.float32) * (1.0 - 0.05 * paper_mask) + white_bg * (0.05 * paper_mask)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_clean_gray(image_bgr, enhance_level="mild"):
    color = enhance_color_scan(image_bgr, enhance_level)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h = 2 if enhance_level == "mild" else 4
    gray = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    strength = 0.1 if enhance_level == "mild" else 0.3
    return unsharp_mask(gray, sigma=0.5, strength=strength)


def enhance_bw(image_bgr, enhance_level="mild"):
    gray = enhance_clean_gray(image_bgr, enhance_level)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)


# ----------------------------------------------------------------------
# Главная функция API
# ----------------------------------------------------------------------

def process_document(image_bytes, scan_mode="color", auto_crop=False, enhance_level="mild"):
    if scan_mode == "mild_color":
        scan_mode = "color"
        enhance_level = "mild"

    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")

    original_h, original_w = image.shape[:2]
    working = image
    crop_applied = False
    crop_detector = "disabled"

    if auto_crop:
        working, crop_applied, crop_detector = crop_document_if_found(image)

    if scan_mode == "bw":
        scan = enhance_bw(working, enhance_level)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(working, enhance_level)
    else:
        scan = enhance_color_scan(working, enhance_level)

    meta = {
        "pipeline": "auto_crop_enhance" if auto_crop else "enhance_only",
        "scan_mode": scan_mode,
        "enhance_level": enhance_level,
        "auto_crop_requested": auto_crop,
        "auto_crop_applied": crop_applied,
        "crop_detector": crop_detector,
        "original_size": [original_w, original_h],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
