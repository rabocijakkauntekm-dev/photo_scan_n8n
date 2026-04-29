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
# Улучшенная детекция контура документа
# ----------------------------------------------------------------------

def find_document_contour_canny(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Сильное размытие для игнорирования текстуры дерева
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Адаптивные границы Canny
    edged = cv2.Canny(blurred, 30, 150)
    
    # Морфологическое закрытие для объединения разрывов в линиях
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Ищем четырехугольник с площадью не менее 10% от фото
        if len(approx) == 4 and cv2.contourArea(approx) > img_area * 0.1:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def find_document_contour_hough(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=150, maxLineGap=50)
    if lines is None or len(lines) < 4:
        return None
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < -90: angle += 180
        elif angle > 90: angle -= 180
        if abs(angle) <= 10: horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) <= 10: vertical.append((x1, y1, x2, y2))
    
    if len(horizontal) < 2 or len(vertical) < 2: return None
    horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
    vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
    
    tl = line_intersection(horizontal[0], vertical[0])
    tr = line_intersection(horizontal[0], vertical[-1])
    br = line_intersection(horizontal[-1], vertical[-1])
    bl = line_intersection(horizontal[-1], vertical[0])
    
    if any(p is None for p in [tl, tr, br, bl]): return None
    points = np.array([tl, tr, br, bl], dtype=np.float32)
    return points if cv2.contourArea(cv2.convexHull(points.reshape(-1, 1, 2))) > image_bgr.shape[0] * image_bgr.shape[1] * 0.15 else None

# ----------------------------------------------------------------------
# Логика обрезки и улучшения
# ----------------------------------------------------------------------

def is_reasonable_warp(original: np.ndarray, warped: np.ndarray) -> bool:
    oh, ow = original.shape[:2]
    wh, ww = warped.shape[:2]
    if wh < 300 or ww < 300: return False
    area_ratio = (wh * ww) / (oh * ow)
    if area_ratio < 0.10 or area_ratio > 0.95: return False
    aspect_warp = ww / wh
    return 0.4 <= aspect_warp <= 2.5

def _apply_padding(image, x, y, w, h, pad_ratio=0.02):
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x, y = max(0, x - pad_x), max(0, y - pad_y)
    w = min(w + 2 * pad_x, image.shape[1] - x)
    h = min(h + 2 * pad_y, image.shape[0] - y)
    return image[y:y+h, x:x+w].copy()

def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    resized, scale = resize_for_detection(image_bgr)
    points = find_document_contour_canny(resized)
    detector = "canny_advanced"
    
    if points is None:
        points = find_document_contour_hough(resized)
        detector = "hough"

    if points is not None:
        points *= scale
        warped = four_point_transform(image_bgr, points)
        if is_reasonable_warp(image_bgr, warped):
            return warped, True, f"{detector}_warp"
        
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        return _apply_padding(image_bgr, x, y, w, h), True, f"{detector}_bbox"

    return image_bgr, False, "none"

# ----------------------------------------------------------------------
# Улучшенная обработка цвета (сохранение печатей и удаление теней)
# ----------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

def enhance_color_scan(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    # Работа в пространстве LAB для выравнивания только яркости (L)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Оценка фона для удаления теней и выравнивания листа
    kernel_size = 201 
    dilated_l = cv2.dilate(l, np.ones((kernel_size, kernel_size), np.uint8))
    bg_l = cv2.medianBlur(dilated_l, kernel_size)
    
    # Вычитание фона: (Original / Background) * 255
    diff_l = 255 - cv2.absdiff(l, bg_l)
    norm_l = cv2.normalize(diff_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Повышение контраста через CLAHE
    clip = 1.2 if enhance_level == "mild" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_final = clahe.apply(norm_l)
    
    # Сборка каналов обратно (цвета a и b остаются нетронутыми — печати сохраняются)
    enhanced_lab = cv2.merge([l_final, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Финальная резкость
    return unsharp_mask(result, sigma=0.4, strength=0.3)

def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 31, 10)

# ----------------------------------------------------------------------
# Основной API
# ----------------------------------------------------------------------

def process_document(image_bytes: bytes, scan_mode: str = "color", auto_crop: bool = True, enhance_level: str = "mild") -> tuple[np.ndarray, dict]:
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Invalid image file")

    working = image
    crop_applied, detector = False, "disabled"

    if auto_crop:
        working, crop_applied, detector = crop_document_if_found(image)

    if scan_mode == "bw":
        scan = enhance_bw(working)
    else:
        scan = enhance_color_scan(working, enhance_level)

    meta = {
        "auto_crop_applied": crop_applied,
        "detector_used": detector,
        "result_size": [int(scan.shape[1]), int(scan.shape[0])]
    }
    return scan, meta
