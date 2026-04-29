"""
scanner.py
Улучшенное сканирование документа: поиск четырёхугольника с верификацией,
качественное выпрямление, три режима улучшения (цвет, серая, ч/б).
Результат приближен к Adobe Scan.
"""

import math
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

MAX_SIDE = 2000

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def resize_for_detection(image: np.ndarray, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # левый-верхний
    rect[2] = pts[np.argmax(s)]   # правый-нижний
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # правый-верхний
    rect[3] = pts[np.argmax(diff)]  # левый-нижний
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, border_value=(255, 255, 255)) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))
    if max_width < 50 or max_height < 50:
        return image
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return warped

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    cos = dot / norm
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))

def is_valid_quad(pts: np.ndarray, img_shape: Tuple[int, int], min_area_ratio=0.05) -> bool:
    """Проверяет, является ли четырёхугольник разумным."""
    if pts is None or len(pts) != 4:
        return False
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Проверка на вырожденность
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    if min(w1, w2) < 30 or min(h1, h2) < 30:
        return False
    aspect_w = max(w1, w2) / max(h1, h2) if max(h1, h2) > 0 else 1
    if aspect_w > 4.0 or aspect_w < 0.25:
        return False
    # Проверка на перспективные искажения (разница между противоположными сторонами)
    if max(w1, w2) / min(w1, w2) > 1.8 or max(h1, h2) / min(h1, h2) > 1.8:
        return False
    # Площадь относительно изображения
    area = cv2.contourArea(pts.astype(np.float32))
    img_area = img_shape[0] * img_shape[1]
    if area / img_area < min_area_ratio:
        return False
    return True

# ----------------------------------------------------------------------
# Детекция документа (несколько методов)
# ----------------------------------------------------------------------

def find_quad_hough(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Метод HoughLinesP для поиска прямоугольника по линиям."""
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
        # normalisation angle to [0,180)
        angle = angle % 180
        if angle < 10 or angle > 170:
            horizontal.append((x1, y1, x2, y2))
        elif 80 < angle < 100:
            vertical.append((x1, y1, x2, y2))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    # Сортируем горизонтальные по y, вертикальные по x
    horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
    vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
    # Берём крайние
    top_line = horizontal[0]
    bottom_line = horizontal[-1]
    left_line = vertical[0]
    right_line = vertical[-1]

    def intersection(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=np.float32)

    tl = intersection(top_line, left_line)
    tr = intersection(top_line, right_line)
    br = intersection(bottom_line, right_line)
    bl = intersection(bottom_line, left_line)
    if None in (tl, tr, br, bl):
        return None
    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    # Проверка на выпуклость и порядок
    hull = cv2.convexHull(pts.reshape(-1,1,2))
    if len(hull) != 4:
        return None
    return order_points(pts)  # упорядочим

def find_quad_contour(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Метод поиска наибольшего четырёхугольного контура."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Используем адаптивный порог для лучшего разделения
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # Морфология закрытия для соединения разрывов
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    best_quad = None
    best_score = 0.0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            if is_valid_quad(quad, image_bgr.shape, min_area_ratio=0.05):
                area = cv2.contourArea(quad)
                score = area / img_area
                if score > best_score:
                    best_score = score
                    best_quad = quad
    return best_quad

def find_quad_brightness(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Метод на основе порога по яркости (для контрастных документов)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Порог Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7,7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    best_quad = None
    best_score = 0.0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area / img_area < 0.10:
            continue
        # Пытаемся аппроксимировать до 4 точек
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if is_valid_quad(quad, image_bgr.shape, min_area_ratio=0.10):
                score = area / img_area
                if score > best_score:
                    best_score = score
                    best_quad = quad
        else:
            # Пробуем ограничивающий прямоугольник
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            quad = order_points(box.astype(np.float32))
            if is_valid_quad(quad, image_bgr.shape, min_area_ratio=0.10):
                score = cv2.contourArea(box.astype(np.float32)) / img_area
                if score > best_score:
                    best_score = score
                    best_quad = quad
    return best_quad

def find_document_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Комбинированный поиск четырёхугольника документа."""
    # Пробуем методы по возрастанию стоимости
    quad = find_quad_hough(image_bgr)
    if quad is not None:
        return quad
    quad = find_quad_contour(image_bgr)
    if quad is not None:
        return quad
    quad = find_quad_brightness(image_bgr)
    return quad

# ----------------------------------------------------------------------
# Умный fallback – поиск главного текстового блока
# ----------------------------------------------------------------------

def smart_fallback_crop(image_bgr: np.ndarray) -> np.ndarray:
    """
    Если документ не найден, обрезаем по наибольшей связной области,
    где есть текст (быстрое решение для фотографий документа).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Повышаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    # Порог Otsu
    _, binary = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Убираем мелкий шум
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback – простой не белый фон
        non_white = np.where(gray < 240)
        if non_white[0].size > 0:
            y_min, y_max = non_white[0].min(), non_white[0].max()
            x_min, x_max = non_white[1].min(), non_white[1].max()
            h, w = image_bgr.shape[:2]
            x_min = max(0, x_min - 10)
            y_min = max(0, y_min - 10)
            x_max = min(w, x_max + 10)
            y_max = min(h, y_max + 10)
            return image_bgr[y_min:y_max, x_min:x_max]
        return image_bgr
    # Выбираем самый большой контур (он должен быть документом или его частью)
    largest = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest)
    # Добавляем небольшой отступ
    pad_x = int(w_box * 0.02)
    pad_y = int(h_box * 0.02)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w_box = min(w_box + 2*pad_x, image_bgr.shape[1] - x)
    h_box = min(h_box + 2*pad_y, image_bgr.shape[0] - y)
    return image_bgr[y:y+h_box, x:x+w_box].copy()

# ----------------------------------------------------------------------
# Основная функция обрезки и выпрямления
# ----------------------------------------------------------------------

def crop_and_warp_document(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Обрезает и выпрямляет документ. Возвращает (изображение, метаданные).
    """
    h, w = image_bgr.shape[:2]
    original_size = (w, h)
    # Работаем с уменьшенной копией для скорости
    small, scale = resize_for_detection(image_bgr)
    quad_small = find_document_quad(small)
    meta = {
        "quad_found": quad_small is not None,
        "warp_applied": False,
        "fallback_crop": False
    }
    if quad_small is None:
        # Не нашли четырёхугольник – используем умный обрез
        cropped = smart_fallback_crop(image_bgr)
        meta["fallback_crop"] = True
        return cropped, meta
    # Масштабируем точки обратно
    quad = quad_small * scale
    # Проверяем валидность на оригинальном размере
    if not is_valid_quad(quad, (h, w), min_area_ratio=0.03):
        # Если невалиден, используем fallback
        cropped = smart_fallback_crop(image_bgr)
        meta["fallback_crop"] = True
        return cropped, meta
    # Пытаемся сделать перспективное выпрямление
    try:
        warped = four_point_transform(image_bgr, quad, border_value=(255,255,255))
        # Проверяем результат: не слишком ли маленький?
        if warped.shape[0] < 100 or warped.shape[1] < 100:
            raise ValueError("warped too small")
        # Проверка соотношения сторон
        aspect = warped.shape[1] / warped.shape[0]
        if aspect < 0.3 or aspect > 3.5:
            # Слишком вытянуто – возможно ошибка, используем bounding rect вместо warp
            x, y, wb, hb = cv2.boundingRect(quad.astype(np.int32))
            cropped = image_bgr[y:y+hb, x:x+wb].copy()
            meta["warp_applied"] = False
            meta["bbox_crop"] = True
            return cropped, meta
        meta["warp_applied"] = True
        return warped, meta
    except Exception:
        # В случае ошибки – используем boundingRect
        x, y, wb, hb = cv2.boundingRect(quad.astype(np.int32))
        cropped = image_bgr[y:y+hb, x:x+wb].copy()
        meta["warp_applied"] = False
        meta["bbox_crop"] = True
        return cropped, meta

# ----------------------------------------------------------------------
# Улучшение качества (цвет, серая, ч/б)
# ----------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Коррекция неравномерной освещённости."""
    h, w = gray.shape[:2]
    kernel = max(31, (min(h, w) // 8) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # Избегаем деления на ноль
    background = np.clip(background, 5, 255)
    return cv2.divide(gray.astype(np.float32), background.astype(np.float32), scale=255).astype(np.uint8)

def enhance_color(image: np.ndarray, level='normal') -> np.ndarray:
    """Цветное улучшение (насыщенные цвета, контраст, резкость, выравнивание фона)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Сначала выравниваем освещение
    gray_norm = normalize_illumination(gray)
    # Применяем CLAHE к яркостному каналу
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Улучшаем L канал (можно использовать нормализованный gray_norm, но он одноканальный)
    # Лучше: смешиваем исходный L с нормализованным gray
    l_enh = cv2.addWeighted(l, 0.7, gray_norm, 0.3, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0 if level=='normal' else 3.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l_enh)
    lab_enh = cv2.merge([l_enh, a, b])
    result = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    # Немного повышаем насыщенность
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.addWeighted(hsv[:,:,1], 1.0, hsv[:,:,1], 0.15, 0)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Резкость
    result = unsharp_mask(result, sigma=0.8, strength=0.5)
    # Осветляем фон (делаем белее)
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Простая маска для светлых участков
    mask = cv2.inRange(gray_res, 200, 255)
    result[mask > 0] = cv2.addWeighted(result[mask > 0], 0.9, (255,255,255), 0.1, 0)
    return result

def enhance_clean_gray(image: np.ndarray, level='normal') -> np.ndarray:
    """Чистый серый скан (сохранение деталей)."""
    color_enh = enhance_color(image, level)
    gray = cv2.cvtColor(color_enh, cv2.COLOR_BGR2GRAY)
    # Сглаживание шумов, но сохранение резкости
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    # Лёгкое повышение контраста
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def enhance_bw(image: np.ndarray, level='normal') -> np.ndarray:
    """Чёрно-белый высококонтрастный скан (бинарный)."""
    gray = enhance_clean_gray(image, level)
    # Адаптивный порог для наилучшего результата
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 8)
    # Морфологическая чистка
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

# ----------------------------------------------------------------------
# Главная функция API
# ----------------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = True,
    enhance_level: str = "normal"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Основная функция обработки.
    :param file_bytes: байты изображения
    :param scan_mode: "color", "clean_gray", "bw"
    :param auto_crop: включено ли автообрезание/выпрямление
    :param enhance_level: "normal" или "mild"/"strong" (пока не используем)
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    original_h, original_w = img.shape[:2]
    meta = {
        "original_size": [original_w, original_h],
        "auto_crop": auto_crop,
        "scan_mode": scan_mode,
        "enhance_level": enhance_level,
    }

    if auto_crop:
        cropped, crop_meta = crop_and_warp_document(img)
        meta.update(crop_meta)
        working = cropped
    else:
        working = img
        meta["quad_found"] = False
        meta["warp_applied"] = False
        meta["fallback_crop"] = False

    # Улучшение в зависимости от режима
    if scan_mode == "bw":
        result = enhance_bw(working, enhance_level)
        # Для единообразия конвертируем в 3 канала (BGR)
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif scan_mode == "clean_gray":
        result = enhance_clean_gray(working, enhance_level)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:  # color
        result = enhance_color(working, enhance_level)

    meta["result_size"] = [result.shape[1], result.shape[0]]
    return result, meta
