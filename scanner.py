import math
import cv2
import numpy as np

MAX_SIDE = 2000
"""Максимальный размер стороны при уменьшении изображения для детекции контура."""

# ----------------------------------------------------------------------
# Вспомогательные функции для детекции и перспективы
# ----------------------------------------------------------------------

def resize_for_detection(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """
    Уменьшает изображение только для ускорения поиска контура.
    Возвращает (уменьшенное_изображение, коэффициент_возврата_к_оригиналу).
    """
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки в порядке: левый верхний, правый верхний,
    правый нижний, левый нижний.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Выполняет перспективное преобразование, выпрямляя четырёхугольник.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 50:
        return image  # слишком маленький – не применяем трансформацию

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def is_reasonable_warp(original: np.ndarray, warped: np.ndarray) -> bool:
    """
    Отбрасывает заведомо плохие результаты перспективного преобразования.
    """
    oh, ow = original.shape[:2]
    wh, ww = warped.shape[:2]

    if wh < 300 or ww < 300:
        return False

    original_area = oh * ow
    warped_area = wh * ww
    area_ratio = warped_area / original_area
    if area_ratio < 0.1 or area_ratio > 0.95:   # слишком маленький или почти весь кадр
        return False

    # Соотношение сторон итогового документа должно быть разумным
    aspect_orig = ow / oh
    aspect_warp = ww / wh
    if aspect_warp < 0.3 or aspect_warp > 3.5:
        return False
    if abs(aspect_warp - aspect_orig) > 2.0:    # слишком сильное отличие от оригинала
        return False

    return True


# ----------------------------------------------------------------------
# Вспомогательная функция: пересечение двух прямых
# ----------------------------------------------------------------------

def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    """
    Вычисляет точку пересечения двух отрезков, заданных координатами (x1,y1,x2,y2).
    Возвращает [x, y] или None, если отрезки параллельны.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


# ----------------------------------------------------------------------
# Детекция контура документа (три стратегии)
# ----------------------------------------------------------------------

def find_document_contour_hough(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Стратегия 0 (новая): ищет четыре прямые линии, образующие замкнутый контур,
    с помощью HoughLinesP. Отбирает по направлениям и находит углы.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=120, maxLineGap=60)
    if lines is None or len(lines) < 4:
        return None

    # Разделяем линии на горизонтальные и вертикальные по углу
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Приводим к диапазону (-180, 180]
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        if abs(angle) <= 15:          # горизонтальные (~0°)
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) <= 15:   # вертикальные (~90°)
            vertical.append((x1, y1, x2, y2))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # Сортируем горизонтальные линии по средней Y-координате (сверху вниз)
    horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
    top_line = horizontal[0]
    bottom_line = horizontal[-1]

    # Вертикальные линии сортируем по средней X-координате (слева направо)
    vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
    left_line = vertical[0]
    right_line = vertical[-1]

    # Находим четыре угла как пересечения
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)

    if tl is None or tr is None or br is None or bl is None:
        return None

    points = np.array([tl, tr, br, bl], dtype=np.float32)
    # Проверяем, что четырёхугольник выпуклый и имеет ненулевую площадь
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if len(hull) != 4:
        return None
    area = cv2.contourArea(hull)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    if area < img_area * 0.15:
        return None

    return points


def find_document_contour_canny(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Стратегия 1: поиск четырёхугольника через границы Canny.
    Возвращает 4x2 массив точек или None.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image_area * 0.15:
            return approx.reshape(4, 2).astype(np.float32)

    # Если нет четырёхугольника, пробуем взять минимальный прямоугольник самого большого контура
    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) > image_area * 0.15:
            return box.astype(np.float32)
    return None


def find_document_contour_brightness(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Стратегия 2: поиск яркой четырёхугольной области (светлая бумага на тёмном столе).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = image_area * 0.15

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        # Если аппроксимация дала не 4 точки, то берём минимальный повёрнутый прямоугольник
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box.astype(np.float32)) >= min_area:
            return box.astype(np.float32)
    return None


def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """
    Пытается найти документ и выпрямить его.
    Возвращает (обрезанное_изображение, успех_ли, имя_детектора).
    """
    resized, scale_to_original = resize_for_detection(image_bgr)

    # Пробуем три детектора по очереди
    points = find_document_contour_hough(resized)
    detector = "hough"
    if points is None:
        points = find_document_contour_canny(resized)
        detector = "canny"
    if points is None:
        points = find_document_contour_brightness(resized)
        detector = "brightness" if points is not None else "none"

    if points is None:
        return image_bgr, False, "none"

    points = points * scale_to_original
    warped = four_point_transform(image_bgr, points)

    # Проверяем, что преобразование дало разумный результат
    if not is_reasonable_warp(image_bgr, warped):
        return image_bgr, False, "rejected_warp"

    # Если warped совпадает с оригиналом (трансформация не применялась) – считаем неудачей
    if warped is image_bgr:
        return image_bgr, False, "invalid_warp"

    return warped, True, detector


# ----------------------------------------------------------------------
# Функции улучшения изображения
# ----------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    """Нечёткое маскирование для повышения резкости."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Подавляет тени и градиенты на странице путём деления на фон."""
    h, w = gray.shape[:2]
    kernel = max(31, (min(h, w) // 6) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.divide(gray, background, scale=255)


def enhance_color_scan(image_bgr: np.ndarray) -> np.ndarray:
    """
    Цветное улучшение: выравнивание освещения, CLAHE, билатеральный фильтр,
    повышение резкости и мягкое отбеливание бумаги (ослабленные настройки).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Снижена агрессивность CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_chan = clahe.apply(norm)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Более мягкий билатеральный фильтр
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)

    # Умеренное повышение резкости
    enhanced = unsharp_mask(enhanced, sigma=0.5, strength=0.6)

    # Лёгкое отбеливание бумажных участков (ослаблено)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 120), (180, 80, 255)).astype(np.float32) / 255.0
    paper_mask = cv2.GaussianBlur(paper_mask, (9, 9), 0)[..., None]
    white_bg = np.full_like(enhanced, 247, dtype=np.float32)
    mixed = enhanced.astype(np.float32) * (1.0 - 0.15 * paper_mask) + white_bg * (0.15 * paper_mask)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Чистый серый режим: цветное улучшение + шумоподавление + мягкое повышение резкости.
    """
    color_enhanced = enhance_color_scan(image_bgr)
    gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
    # Снижена сила шумоподавления
    gray = cv2.fastNlMeansDenoising(gray, None, h=4, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return unsharp_mask(gray, sigma=0.7, strength=0.45)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    """
    Чёрно-белый режим: адаптивная бинаризация на основе чистового серого.
    """
    clean_gray = enhance_clean_gray(image_bgr)
    return cv2.adaptiveThreshold(
        clean_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------

def process_document(
    image_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Основной конвейер обработки документа.

    Параметры:
        image_bytes: байты изображения (PNG, JPEG и т.д.)
        scan_mode: 'color', 'clean_gray' или 'bw'
        auto_crop: выполнять ли автоматическое кадрирование и выпрямление

    Возвращает:
        (обработанное_изображение, метаданные)
    """
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
        scan = enhance_bw(working)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(working)
    else:  # color
        scan = enhance_color_scan(working)

    meta = {
        "pipeline": "auto_crop_enhance" if auto_crop else "enhance_only",
        "scan_mode": scan_mode,
        "auto_crop_requested": auto_crop,
        "auto_crop_applied": crop_applied,
        "crop_detector": crop_detector,
        "original_size": [int(original_w), int(original_h)],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
