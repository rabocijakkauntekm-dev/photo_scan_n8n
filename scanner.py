"""
scanner.py
Улучшенный сканнер документа с двумя методами поиска четырёхугольника
и качественной пост-обработкой (как Adobe Scan).
"""

import io
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

# ------------------------------------------------------------
# 1. Геометрические функции
# ------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки: левый-верхний, правый-верхний, правый-нижний, левый-нижний."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # левый-верхний
    rect[2] = pts[np.argmax(s)]   # правый-нижний
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # правый-верхний
    rect[3] = pts[np.argmax(diff)] # левый-нижний
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Перспективное выпрямление по четырём точкам."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Ширина и высота нового изображения (максимальные расстояния)
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = max(int(height_right), int(height_left))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

# ------------------------------------------------------------
# 2. Поиск документа (две стратегии)
# ------------------------------------------------------------

def find_document_contour_robust(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Основная функция поиска четырёхугольника.
    Возвращает 4 точки (np.ndarray shape (4,2)) или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Улучшаем контраст перед детектированием
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enh = clahe.apply(gray)

    # 2. Размытие для подавления шума
    blurred = cv2.GaussianBlur(gray_enh, (5, 5), 0)

    # 3. Адаптивный порог (лучше для неровного освещения)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. Морфологическое закрытие и открытие
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Поиск контуров
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Сортируем по площади (самый большой контур скорее всего документ)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = h * w

    best_quad = None
    best_score = 0

    for cnt in contours[:10]:  # проверяем первые 10 самых больших
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < 0.03 or area_ratio > 0.98:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            # Проверяем, что углы не слишком острые (не вырожденный четырёхугольник)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            # Ширина и высота не должны быть слишком маленькими
            w1 = np.linalg.norm(tr - tl)
            w2 = np.linalg.norm(br - bl)
            h1 = np.linalg.norm(bl - tl)
            h2 = np.linalg.norm(br - tr)
            if min(w1, w2) < 20 or min(h1, h2) < 20:
                continue
            aspect = max(w1, w2) / max(h1, h2) if max(h1, h2) > 0 else 1
            if aspect > 4.0 or aspect < 0.25:
                continue
            # Оценка: чем ближе площадь к площади изображения, тем лучше
            score = area / img_area
            if score > best_score:
                best_score = score
                best_quad = pts

    if best_quad is not None:
        return best_quad.astype(np.float32)
    return None

def find_document_lines_method(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Альтернативный метод: поиск через пересечение линий (если контур не найден).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Детектор границ Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)

    if lines is None or len(lines) < 4:
        return None

    # Объединяем близкие горизонтальные и вертикальные линии
    h_lines = []   # угол около 0° или 180°
    v_lines = []   # угол около 90°
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 30:
            h_lines.append((x1, y1, x2, y2))
        elif abs(abs(angle) - 90) < 30:
            v_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # Находим крайние точки
    all_points = []
    for h in h_lines[:5]:
        for v in v_lines[:5]:
            # Пересечение двух отрезков
            x1_h, y1_h, x2_h, y2_h = h
            x1_v, y1_v, x2_v, y2_v = v
            # Находим пересечение прямых
            A1 = y2_h - y1_h
            B1 = x1_h - x2_h
            C1 = A1*x1_h + B1*y1_h
            A2 = y2_v - y1_v
            B2 = x1_v - x2_v
            C2 = A2*x1_v + B2*y1_v
            det = A1*B2 - A2*B1
            if abs(det) > 1e-5:
                x = (B2*C1 - B1*C2) / det
                y = (A1*C2 - A2*C1) / det
                if 0 <= x < w and 0 <= y < h:
                    all_points.append((x, y))

    if len(all_points) < 4:
        return None

    # Ищем выпуклую оболочку и аппроксимируем до 4 точек
    points = np.array(all_points, dtype=np.float32)
    hull = cv2.convexHull(points)
    if len(hull) < 4:
        return None
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    # Если не вышло, берём 4 крайние точки
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)

def find_document_fallback(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Самый простой метод: взять весь кадр, но с небольшим отступом от краёв.
    Используется только если два предыдущих метода не сработали.
    """
    h, w = image.shape[:2]
    margin = int(min(h, w) * 0.05)
    return np.array([[margin, margin], [w-margin, margin], [w-margin, h-margin], [margin, h-margin]], dtype=np.float32)

# ------------------------------------------------------------
# 3. Улучшение качества выпрямленного изображения
# ------------------------------------------------------------

def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Нерезкое маскирование для повышения резкости."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
    sharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def denoise(image: np.ndarray) -> np.ndarray:
    """Удаление мелких шумов (быстрый медианный фильтр)."""
    return cv2.medianBlur(image, 3)

def color_enhance(image: np.ndarray) -> np.ndarray:
    """Цветное улучшение: CLAHE в LAB + резкость."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    bgr = sharpen_image(bgr)
    return denoise(bgr)

def clean_gray_enhance(image: np.ndarray) -> np.ndarray:
    """Ч/Б с сохранением оттенков серого."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (1, 1), 0)  # лёгкое сглаживание
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def bw_enhance(image: np.ndarray) -> np.ndarray:
    """
    Высококонтрастный бинарный (black-white) скан.
    Используем адаптивный порог + морфологию для чистоты.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Сначала повышаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray)

    # Бинаризация с помощью OTSU (глобальный порог) — для равномерного фона
    # Но лучше использовать адаптивный
    binary = cv2.adaptiveThreshold(gray_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    # Удаляем мелкие шумы
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def enhance_image(image: np.ndarray, scan_mode: str) -> np.ndarray:
    """Применяет выбранный режим улучшения."""
    if scan_mode == "color":
        return color_enhance(image)
    elif scan_mode == "clean_gray":
        return clean_gray_enhance(image)
    elif scan_mode == "bw":
        return bw_enhance(image)
    else:
        return image

# ------------------------------------------------------------
# 4. Главная функция
# ------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Основной вход для API.
    :param file_bytes: сырые байты изображения
    :param scan_mode: "color", "clean_gray", "bw"
    :param auto_crop: если True, пытается найти и выпрямить документ
    :return: (обработанное изображение BGR, мета-словарь)
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")

    meta = {
        "original_size": list(img.shape[:2]),
        "auto_crop_performed": False,
        "scan_mode": scan_mode,
    }

    if auto_crop:
        # Стратегия 1: поиск по контуру
        quad = find_document_contour_robust(img)
        if quad is None:
            # Стратегия 2: поиск по линиям
            quad = find_document_lines_method(img)
        if quad is None:
            # Стратегия 3: отступ от краёв
            quad = find_document_fallback(img)
            meta["fallback_used"] = True
        else:
            meta["auto_crop_performed"] = True

        if quad is not None:
            try:
                img = four_point_transform(img, quad)
                meta["warped_size"] = list(img.shape[:2])
            except Exception as e:
                meta["warp_error"] = str(e)
                # В случае ошибки оставляем исходное
        else:
            meta["no_quad_found"] = True

    # Повышение качества
    enhanced = enhance_image(img, scan_mode)

    return enhanced, meta
