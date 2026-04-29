import math
import cv2
import numpy as np

MAX_SIDE = 2000


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def resize_for_detection(image, max_side=MAX_SIDE):
    """Уменьшает изображение для ускорения детекции."""
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


# ----------------------------------------------------------------------
# Оценка угла наклона документа
# ----------------------------------------------------------------------

def estimate_skew_angle(image_bgr):
    """
    Вычисляет угол поворота документа по прямым линиям (HoughLinesP).
    Возвращает средний угол горизонтальных линий в градусах.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # Утолщаем края
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=150, maxLineGap=40)
    if lines is None or len(lines) < 3:
        # Если линий мало, пробуем минимальный прямоугольник самого крупного контура
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            return rect[2]  # угол из minAreaRect
        return 0.0

    # Собираем углы горизонтальных линий (≈0° или 180°)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Нормализуем угол к диапазону [-45, 45]
        if angle < -45:
            angle += 180
        elif angle > 45:
            angle -= 180
        # Берём абсолютное значение, усредняем
        angles.append(abs(angle))

    if not angles:
        return 0.0
    return np.median(angles)  # устойчивее к выбросам


# ----------------------------------------------------------------------
# Поворот изображения
# ----------------------------------------------------------------------

def rotate_image(image, angle, border_value=(255, 255, 255)):
    """
    Поворачивает изображение на заданный угол (против часовой стрелки),
    дополняя пустые области белым.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return rotated


# ----------------------------------------------------------------------
# Поиск прямоугольника документа на ровном изображении
# ----------------------------------------------------------------------

def find_document_rect(image_bgr):
    """
    Находит ограничивающий прямоугольник документа на уже выровненном изображении.
    Использует адаптивную бинаризацию и морфологию для склеивания текста/таблиц.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Адаптивный порог, чтобы выделить бумагу
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 8)
    # Морфологическое закрытие, чтобы склеить буквы в строки, а строки в блоки
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Расширение, чтобы точно захватить края
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Берём самый большой по площади контур
    largest = max(contours, key=cv2.contourArea)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    if cv2.contourArea(largest) < img_area * 0.2:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    # Добавляем небольшой отступ (2% от размера, но не вылезаем за границы)
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(w + 2 * pad_x, image_bgr.shape[1] - x)
    h = min(h + 2 * pad_y, image_bgr.shape[0] - y)
    return (x, y, w, h)


# ----------------------------------------------------------------------
# Основная функция обрезки (поворот → обрезка)
# ----------------------------------------------------------------------

def crop_document_if_found(image_bgr):
    """
    Алгоритм:
    1. Определить угол поворота документа.
    2. Повернуть изображение так, чтобы документ встал ровно.
    3. Найти и обрезать по ограничивающему прямоугольнику.
    4. Если что-то пошло не так, вернуть исходное изображение.
    """
    # Уменьшаем копию для быстрого определения угла
    small, scale = resize_for_detection(image_bgr, max_side=800)
    angle = estimate_skew_angle(small)

    # Поворачиваем на вычисленный угол (если угол близок к 0, поворот не делаем)
    if abs(angle) < 0.5:
        rotated = image_bgr
    else:
        rotated = rotate_image(image_bgr, angle, border_value=(255, 255, 255))

    # Ищем прямоугольник документа на повёрнутом изображении
    rect = find_document_rect(rotated)
    if rect is not None:
        x, y, w, h = rect
        cropped = rotated[y:y+h, x:x+w].copy()
        # Проверяем, что результат не слишком мал
        if cropped.shape[0] > 200 and cropped.shape[1] > 200:
            return cropped, True, f"rotated_{angle:.1f}"

    # Fallback: если на повёрнутом не найдено, пробуем обрезать по исходному
    rect_orig = find_document_rect(image_bgr)
    if rect_orig is not None:
        x, y, w, h = rect_orig
        cropped = image_bgr[y:y+h, x:x+w].copy()
        if cropped.shape[0] > 200 and cropped.shape[1] > 200:
            return cropped, True, "fallback_orig"

    # Совсем не получилось — возвращаем исходник
    return image_bgr, False, "none"


# ----------------------------------------------------------------------
# Функции улучшения изображения (сохранены мягкие настройки)
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
