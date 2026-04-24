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
    if warped.shape[0] < 300 or warped.shape[1] < 300:
        return False
    original_area = original.shape[0] * original.shape[1]
    warped_area = warped.shape[0] * warped.shape[1]
    if warped_area < original_area * 0.08:
        return False
    return True


# ----------------------------------------------------------------------
# Детекция контура документа (две стратегии)
# ----------------------------------------------------------------------

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
        return cv2.boxPoints(rect).astype(np.float32)
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

    # Сначала пробуем детекцию через Canny
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
    повышение резкости и мягкое отбеливание бумаги.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    l_chan = clahe.apply(norm)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=35, sigmaSpace=35)
    enhanced = unsharp_mask(enhanced, sigma=0.8, strength=0.95)

    # Лёгкое отбеливание бумажных участков
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 120), (180, 80, 255)).astype(np.float32) / 255.0
    paper_mask = cv2.GaussianBlur(paper_mask, (9, 9), 0)[..., None]
    white_bg = np.full_like(enhanced, 247, dtype=np.float32)
    mixed = enhanced.astype(np.float32) * (1.0 - 0.22 * paper_mask) + white_bg * (0.22 * paper_mask)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Чистый серый режим: цветное улучшение + шумоподавление + мягкое повышение резкости.
    """
    color_enhanced = enhance_color_scan(image_bgr)
    gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return unsharp_mask(gray, sigma=0.7, strength=0.55)


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
