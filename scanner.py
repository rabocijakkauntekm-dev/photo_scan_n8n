import math
import cv2
import numpy as np

MAX_SIDE = 2000

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def resize_for_detection(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


# ----------------------------------------------------------------------
# Основная детекция документа
# ----------------------------------------------------------------------

def find_document_contour(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Комбинированный метод: Canny + контуры, если не найдено – Hough.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Canny – поиск по краям
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Сортируем по площади, берём самый крупный
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > image_bgr.shape[0] * image_bgr.shape[1] * 0.15:
                return approx.reshape(4, 2).astype(np.float32)
            # Если 4 угла не найдено, пробуем взять ограничивающий прямоугольник
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            if cv2.contourArea(box) > image_bgr.shape[0] * image_bgr.shape[1] * 0.15:
                return box.astype(np.float32)

    # 2. Hough Lines – если Canny не дал 4‑угольника
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=100, maxLineGap=40)
    if lines is not None and len(lines) >= 4:
        horizontals, verticals = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = angle % 180
            if angle < 10 or angle > 170:
                horizontals.append((x1, y1, x2, y2))
            elif 80 < angle < 100:
                verticals.append((x1, y1, x2, y2))
        if len(horizontals) >= 2 and len(verticals) >= 2:
            horizontals.sort(key=lambda l: (l[1] + l[3]) / 2)
            verticals.sort(key=lambda l: (l[0] + l[2]) / 2)
            top = horizontals[0]
            bottom = horizontals[-1]
            left = verticals[0]
            right = verticals[-1]
            tl = line_intersection(top, left)
            tr = line_intersection(top, right)
            br = line_intersection(bottom, right)
            bl = line_intersection(bottom, left)
            if all([tl is not None, tr is not None, br is not None, bl is not None]):
                pts = np.array([tl, tr, br, bl], dtype=np.float32)
                if cv2.contourArea(pts) > image_bgr.shape[0] * image_bgr.shape[1] * 0.15:
                    return pts

    # 3. Fallback: простая бинаризация по Оцу
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > image_bgr.shape[0] * image_bgr.shape[1] * 0.15:
            rect = cv2.minAreaRect(largest)
            return cv2.boxPoints(rect).astype(np.float32)

    return None


# ----------------------------------------------------------------------
# Выравнивание и обрезка
# ----------------------------------------------------------------------

def affine_straighten(image_bgr: np.ndarray, contour: np.ndarray, border_value=(255, 255, 255)):
    """
    Аффинное выравнивание: поворот на угол контура, обрезка по boundingRect.
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_bgr, rot_mat, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    # Находим документ на повёрнутом изображении
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        page = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(page)
        pad = int(min(rw, rh) * 0.03)
        x = max(0, x - pad)
        y = max(0, y - pad)
        rw = min(rw + 2 * pad, rotated.shape[1] - x)
        rh = min(rh + 2 * pad, rotated.shape[0] - y)
        rotated = rotated[y:y+rh, x:x+rw]
    return rotated


def four_point_transform(image: np.ndarray, pts: np.ndarray, border_value=(255, 255, 255)):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    if max_width < 50 or max_height < 50:
        return image
    dst = np.array([[0, 0], [max_width - 1, 0],
                   [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def is_perspective_needed(contour: np.ndarray, image_shape: tuple) -> bool:
    """
    Перспективная коррекция нужна, только если углы сильно отклоняются от 90°
    или противоположные стороны различаются больше чем на 5%.
    """
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    if w_top < 10 or w_bot < 10 or h_left < 10 or h_right < 10:
        return False
    ratio_w = w_top / w_bot if w_bot else 1
    ratio_h = h_left / h_right if h_right else 1
    if abs(ratio_w - 1.0) > 0.05 or abs(ratio_h - 1.0) > 0.05:
        return True
    angles = []
    def angle(v1, v2):
        dot = np.dot(v1, v2)
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 0 if n == 0 else math.degrees(math.acos(max(-1, min(1, dot / n))))
    angles.append(angle(tr - tl, bl - tl))
    angles.append(angle(tl - tr, br - tr))
    angles.append(angle(bl - br, tr - br))
    angles.append(angle(tl - bl, br - bl))
    max_dev = max(abs(a - 90) for a in angles)
    return max_dev > 3.0


def crop_document_if_found(image_bgr: np.ndarray):
    """
    Ищет документ, выравнивает его. Если ничего не найдено, возвращает исходник.
    """
    resized, scale = resize_for_detection(image_bgr)
    contour = find_document_contour(resized)
    if contour is None:
        return image_bgr, False, "none"

    contour = contour * scale

    # Если документ практически ровный – только поворот и обрезка
    if not is_perspective_needed(contour, image_bgr.shape):
        return affine_straighten(image_bgr, contour), True, "affine"
    # Иначе – перспективное выравнивание
    warped = four_point_transform(image_bgr, contour)
    oh, ow = image_bgr.shape[:2]
    wh, ww = warped.shape[:2]
    # Проверка, что результат не слишком искажён
    if wh < 200 or ww < 200 or (wh * ww) / (oh * ow) < 0.15:
        # fallback – аффинное выравнивание
        return affine_straighten(image_bgr, contour), True, "fallback_affine"
    return warped, True, "perspective"


# ----------------------------------------------------------------------
# Функции улучшения (по умолчанию mild – текст не искажается)
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
# Главная функция
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
