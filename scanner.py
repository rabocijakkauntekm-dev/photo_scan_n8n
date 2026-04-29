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
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray, border_value=(255, 255, 255)) -> np.ndarray:
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
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return warped


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


# ----------------------------------------------------------------------
# Детекция контура документа
# ----------------------------------------------------------------------

def find_document_contour_hough(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=150, maxLineGap=50)
    if lines is None or len(lines) < 4:
        return None
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        if abs(angle) <= 10:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) <= 10:
            vertical.append((x1, y1, x2, y2))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
    vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
    top_line = horizontal[0]
    bottom_line = horizontal[-1]
    left_line = vertical[0]
    right_line = vertical[-1]
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)
    if tl is None or tr is None or br is None or bl is None:
        return None
    points = np.array([tl, tr, br, bl], dtype=np.float32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if len(hull) != 4:
        return None
    if cv2.contourArea(hull) < image_bgr.shape[0] * image_bgr.shape[1] * 0.15:
        return None
    return points


def find_document_contour_canny(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > img_area * 0.15:
            return approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) > img_area * 0.15:
            return box.astype(np.float32)
    return None


def find_document_contour_brightness(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = img_area * 0.15
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box.astype(np.float32)) >= min_area:
            return box.astype(np.float32)
    return None


def is_reasonable_warp(original: np.ndarray, warped: np.ndarray) -> bool:
    oh, ow = original.shape[:2]
    wh, ww = warped.shape[:2]
    if wh < 300 or ww < 300:
        return False
    original_area = oh * ow
    warped_area = wh * ww
    area_ratio = warped_area / original_area
    if area_ratio < 0.15 or area_ratio > 0.95:
        return False
    aspect_orig = ow / oh
    aspect_warp = ww / wh
    if aspect_warp < 0.4 or aspect_warp > 2.5:
        return False
    if abs(aspect_warp - aspect_orig) > 1.5:
        return False
    return True


# ----------------------------------------------------------------------
# Надёжный fallback-обрезатель (всегда что-то возвращает)
# ----------------------------------------------------------------------

def simple_crop_by_content(image_bgr: np.ndarray) -> np.ndarray:
    """
    Если классические детекторы не нашли контур, эта функция всё равно обрежет
    по самому большому содержательному блоку (текст, изображение) с отступом.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Попытка 1: светлая область (бумага)
    mean_val = np.mean(blurred)
    _, light_mask = cv2.threshold(blurred, mean_val - 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    light_closed = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours_light, _ = cv2.findContours(light_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_light:
        largest = max(contours_light, key=cv2.contourArea)
        if cv2.contourArea(largest) > image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            x, y, w, h = cv2.boundingRect(largest)
            return _apply_padding(image_bgr, x, y, w, h)

    # Попытка 2: тёмный текст
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours_dark, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_dark:
        largest = max(contours_dark, key=cv2.contourArea)
        if cv2.contourArea(largest) > image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            x, y, w, h = cv2.boundingRect(largest)
            return _apply_padding(image_bgr, x, y, w, h)

    # Попытка 3: обрезаем всё, что не является полностью белым фоном
    non_white = np.where(gray < 250)
    if non_white[0].size > 0:
        y_min, y_max = non_white[0].min(), non_white[0].max()
        x_min, x_max = non_white[1].min(), non_white[1].max()
        return _apply_padding(image_bgr, x_min, y_min, x_max - x_min, y_max - y_min)

    # Если изображение абсолютно белое – возвращаем как есть
    return image_bgr


def _apply_padding(image, x, y, w, h, pad_ratio=0.02):
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(w + 2 * pad_x, image.shape[1] - x)
    h = min(h + 2 * pad_y, image.shape[0] - y)
    return image[y:y+h, x:x+w].copy()


# ----------------------------------------------------------------------
# Основная функция обрезки
# ----------------------------------------------------------------------

def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """
    Ищет документ и обрезает его (без поворота).
    Если контур найден – делает обычную/перспективную обрезку.
    Если нет – использует fallback по содержимому.
    """
    resized, scale = resize_for_detection(image_bgr)

    points = find_document_contour_hough(resized)
    detector = "hough"
    if points is None:
        points = find_document_contour_canny(resized)
        detector = "canny"
    if points is None:
        points = find_document_contour_brightness(resized)
        detector = "brightness" if points is not None else "none"

    if points is not None:
        points = points * scale
        rect = order_points(points)
        (tl, tr, br, bl) = rect
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_w = max(width_top, width_bottom)
        min_w = min(width_top, width_bottom)
        max_h = max(height_left, height_right)
        min_h = min(height_left, height_right)

        def angle_between(v1, v2):
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm == 0:
                return 0
            cos = dot / norm
            cos = max(-1.0, min(1.0, cos))
            return math.degrees(math.acos(cos))

        angle_tl = angle_between(tr - tl, bl - tl)
        angle_tr = angle_between(tl - tr, br - tr)
        angle_br = angle_between(bl - br, tr - br)
        angle_bl = angle_between(tl - bl, br - bl)
        max_angle_dev = max(abs(a - 90) for a in [angle_tl, angle_tr, angle_br, angle_bl])
        side_ratio_ok = (max_w / min_w < 1.02) and (max_h / min_h < 1.02)

        if max_angle_dev < 2.0 and side_ratio_ok:
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            cropped = _apply_padding(image_bgr, x, y, w, h)
            return cropped, True, f"{detector}_straight"

        warped = four_point_transform(image_bgr, points, border_value=(255, 255, 255))
        if not is_reasonable_warp(image_bgr, warped):
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            cropped = _apply_padding(image_bgr, x, y, w, h)
            return cropped, True, f"{detector}_fallback"
        return warped, True, f"{detector}_warp"

    # Контур не найден → fallback по содержимому
    fallback = simple_crop_by_content(image_bgr)
    return fallback, True, "fallback_content"


# ----------------------------------------------------------------------
# Функции улучшения (сохранены мягкие)
# ----------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    kernel = max(31, (min(h, w) // 6) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.divide(gray, background, scale=255)


def enhance_color_scan(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    if enhance_level == "mild":
        clip_limit, tile_size = 0.5, 8
        d, sigma_c, sigma_s = 3, 8, 8
        us_sigma, us_str = 0.25, 0.15
    elif enhance_level == "normal":
        clip_limit, tile_size = 1.0, 8
        d, sigma_c, sigma_s = 5, 15, 15
        us_sigma, us_str = 0.4, 0.35
    else:  # strong
        clip_limit, tile_size = 2.0, 8
        d, sigma_c, sigma_s = 7, 25, 25
        us_sigma, us_str = 0.6, 0.7

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
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


def enhance_clean_gray(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    color_enhanced = enhance_color_scan(image_bgr, enhance_level)
    gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
    h = 2 if enhance_level == "mild" else 4
    gray = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    strength = 0.1 if enhance_level == "mild" else 0.3
    return unsharp_mask(gray, sigma=0.5, strength=strength)


def enhance_bw(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    clean_gray = enhance_clean_gray(image_bgr, enhance_level)
    return cv2.adaptiveThreshold(clean_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

def process_document(image_bytes: bytes, scan_mode: str = "color", auto_crop: bool = False, enhance_level: str = "mild") -> tuple[np.ndarray, dict]:
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
