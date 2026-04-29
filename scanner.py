import math
import cv2
import numpy as np

MAX_SIDE = 2000
"""Максимальный размер стороны при уменьшении изображения для детекции контура."""


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
# Поиск контура целого листа (с морфологическим закрытием)
# ----------------------------------------------------------------------

def preprocess_for_whole_page(gray: np.ndarray) -> np.ndarray:
    """
    Сильно размывает изображение и применяет адаптивный порог,
    чтобы «склеить» текст, таблицы и фон в единую область листа.
    """
    # Сильное гауссово размытие, чтобы слить буквы и линии таблиц
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    # Адаптивная бинаризация: текст становится чёрным, промежутки — белыми
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 8)
    # Морфологическое закрытие, чтобы заполнить все дыры внутри листа
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed


def find_document_contour_whole_page(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Ищет контур, соответствующий всей странице документа.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Увеличиваем контраст для улучшения детекции
    gray = cv2.equalizeHist(gray)
    mask = preprocess_for_whole_page(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Сортируем по площади, берём самый большой контур
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    page_contour = contours[0]
    # Минимальная охватывающая площадь (хотя бы 30% от кадра)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    if cv2.contourArea(page_contour) < img_area * 0.25:
        return None
    # Аппроксимируем четырёхугольником
    peri = cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, 0.015 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    # Если не 4 угла – берём ограничивающий прямоугольник минимальной площади
    rect = cv2.minAreaRect(page_contour)
    return cv2.boxPoints(rect).astype(np.float32)


# ----------------------------------------------------------------------
# Коррекция геометрии
# ----------------------------------------------------------------------

def estimate_skew_angle(contour: np.ndarray) -> float:
    """
    Возвращает примерный угол поворота документа (в градусах).
    Используем минимальный охватывающий прямоугольник.
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return angle


def affine_straighten(image_bgr: np.ndarray, contour: np.ndarray, border_value=(255, 255, 255)) -> tuple[np.ndarray, np.ndarray]:
    """
    Поворачивает изображение так, чтобы документ стал горизонтальным,
    и обрезает по минимальному ограничивающему прямоугольнику.
    Возвращает (выровненное_изображение, матрица_преобразования).
    """
    rect = cv2.minAreaRect(contour)
    angle = estimate_skew_angle(contour)
    # Центр изображения и матрица поворота
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Поворачиваем
    rotated = cv2.warpAffine(image_bgr, rot_mat, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    # Находим контур документа на повёрнутом изображении
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        page = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(page)
        # Добавляем небольшой отступ
        pad = int(min(rw, rh) * 0.02)
        x = max(0, x - pad)
        y = max(0, y - pad)
        rw = min(rw + 2 * pad, rotated.shape[1] - x)
        rh = min(rh + 2 * pad, rotated.shape[0] - y)
        rotated = rotated[y:y+rh, x:x+rw]
    return rotated, rot_mat


def is_perspective_needed(contour: np.ndarray, image_shape: tuple) -> bool:
    """
    Решает, нужна ли перспективная коррекция, или достаточно аффинного поворота.
    Перспективу применяем только если углы сильно отличаются от 90° (> 3°).
    """
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    # Проверяем соотношение противоположных сторон
    if width_top < 10 or width_bottom < 10 or height_left < 10 or height_right < 10:
        return False  # слишком маленький документ
    ratio_w = width_top / width_bottom if width_bottom != 0 else 1
    ratio_h = height_left / height_right if height_right != 0 else 1
    # Допустимый перекос сторон: 5%
    if abs(ratio_w - 1.0) > 0.05 or abs(ratio_h - 1.0) > 0.05:
        return True
    # Проверяем углы
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
    dst = np.array([[0, 0], [max_width - 1, 0],
                   [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """
    Новый подход:
    1. Ищем контур всей страницы (морфология).
    2. Если документ почти ровный – аффинный поворот + обрезка.
    3. Если есть перспективные искажения – warp с белым фоном.
    """
    resized, scale = resize_for_detection(image_bgr)
    contour = find_document_contour_whole_page(resized)
    if contour is None:
        return image_bgr, False, "none"

    contour = contour * scale  # обратно к оригинальному масштабу

    if not is_perspective_needed(contour, image_bgr.shape):
        # Аффинное выравнивание
        straightened, _ = affine_straighten(image_bgr, contour, border_value=(255, 255, 255))
        return straightened, True, "affine"
    else:
        # Перспективное выравнивание
        warped = four_point_transform(image_bgr, contour, border_value=(255, 255, 255))
        # Проверяем, что результат не искажён
        oh, ow = image_bgr.shape[:2]
        wh, ww = warped.shape[:2]
        if wh < 200 or ww < 200:
            return image_bgr, False, "bad_warp"
        area_ratio = (wh * ww) / (oh * ow)
        if area_ratio < 0.15 or area_ratio > 0.9:
            # fallback на аффинный
            straightened, _ = affine_straighten(image_bgr, contour, border_value=(255, 255, 255))
            return straightened, True, "fallback_affine"
        return warped, True, "perspective"


# ----------------------------------------------------------------------
# Функции улучшения (максимально щадящие по умолчанию)
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
    """
    Улучшение цвета. Параметры подобраны для максимальной сохранности текста.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Настройки по уровням
    if enhance_level == "mild":
        clip_limit = 0.5
        tile_size = 8
        d = 3
        sigma_color = 8
        sigma_space = 8
        unsharp_sigma = 0.25
        unsharp_strength = 0.15
    elif enhance_level == "normal":
        clip_limit = 1.0
        tile_size = 8
        d = 5
        sigma_color = 15
        sigma_space = 15
        unsharp_sigma = 0.4
        unsharp_strength = 0.35
    else:  # strong
        clip_limit = 2.0
        tile_size = 8
        d = 7
        sigma_color = 25
        sigma_space = 25
        unsharp_sigma = 0.6
        unsharp_strength = 0.7

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_chan = clahe.apply(norm)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    enhanced = cv2.bilateralFilter(enhanced, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    enhanced = unsharp_mask(enhanced, sigma=unsharp_sigma, strength=unsharp_strength)

    # Мягкое отбеливание бумаги
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255)).astype(np.float32) / 255.0
    paper_mask = cv2.GaussianBlur(paper_mask, (9, 9), 0)[..., None]
    white_bg = np.full_like(enhanced, 247, dtype=np.float32)
    mixed = enhanced.astype(np.float32) * (1.0 - 0.05 * paper_mask) + white_bg * (0.05 * paper_mask)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_clean_gray(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    color_enhanced = enhance_color_scan(image_bgr, enhance_level=enhance_level)
    gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
    h = 2 if enhance_level == "mild" else 4
    gray = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    strength = 0.1 if enhance_level == "mild" else 0.3
    return unsharp_mask(gray, sigma=0.5, strength=strength)


def enhance_bw(image_bgr: np.ndarray, enhance_level: str = "mild") -> np.ndarray:
    clean_gray = enhance_clean_gray(image_bgr, enhance_level=enhance_level)
    return cv2.adaptiveThreshold(clean_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)


# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------

def process_document(
    image_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = False,
    enhance_level: str = "mild",
) -> tuple[np.ndarray, dict]:
    """
    Основной конвейер.
    """
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
        scan = enhance_bw(working, enhance_level=enhance_level)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(working, enhance_level=enhance_level)
    else:
        scan = enhance_color_scan(working, enhance_level=enhance_level)

    meta = {
        "pipeline": "auto_crop_enhance" if auto_crop else "enhance_only",
        "scan_mode": scan_mode,
        "enhance_level": enhance_level,
        "auto_crop_requested": auto_crop,
        "auto_crop_applied": crop_applied,
        "crop_detector": crop_detector,
        "original_size": [int(original_w), int(original_h)],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
