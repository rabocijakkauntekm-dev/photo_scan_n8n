"""
scanner.py
Полноценный сканнер документа: поиск четырёхугольника,
коррекция перспективы, улучшение (color/grayscale/black-white).
"""

import io
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

# ------------------------------------------------------------
# 1. Вспомогательные геометрические функции
# ------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки в порядке: [левый-верхний, правый-верхний,
    правый-нижний, левый-нижний].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]          # левый-верхний (мин сумма)
    rect[2] = pts[np.argmax(s)]          # правый-нижний (макс сумма)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]       # правый-верхний
    rect[3] = pts[np.argmax(diff)]       # левый-нижний
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Выполняет перспективное преобразование, приводя четырёхугольник pts
    к прямоугольному виду.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Вычисляем ширину нового изображения
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    # Вычисляем высоту
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = max(int(height_right), int(height_left))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def find_document_contour(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    epsilon_factor: float = 0.02
) -> Optional[np.ndarray]:
    """
    Находит контур, который скорее всего является страницей документа.
    Возвращает 4 точки (np.ndarray shape (4,2)) или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Адаптивный порог даёт лучший результат для разных условий освещения
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Морфологическое закрытие для соединения близких областей
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image.shape[:2]
    img_area = h * w
    best_contour = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Аппроксимируем контур с уменьшенной точностью для поиска четырёхугольника
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)

        if len(approx) == 4:
            # Дополнительная проверка на выпуклость и вогнутость
            if cv2.isContourConvex(approx):
                # Чем больше площадь и чем ближе форма к прямоугольнику – тем лучше
                rect_area = cv2.contourArea(approx)
                score = rect_area / img_area
                # Бонус за меньшее количество искажений: проверка соотношения сторон
                pts = approx.reshape(4, 2)
                rect = order_points(pts)
                (tl, tr, br, bl) = rect
                w1 = np.linalg.norm(tr - tl)
                w2 = np.linalg.norm(br - bl)
                h1 = np.linalg.norm(bl - tl)
                h2 = np.linalg.norm(br - tr)
                aspect_ratio = max(w1, w2) / max(h1, h2) if max(h1, h2) > 0 else 1
                # Игнорируем слишком вытянутые "документы" (больше 3:1)
                if aspect_ratio > 3.0:
                    continue
                if score > best_score:
                    best_score = score
                    best_contour = approx

    if best_contour is not None:
        return best_contour.reshape(4, 2).astype(np.float32)
    return None


# ------------------------------------------------------------
# 2. Улучшение качества изображения
# ------------------------------------------------------------

def color_enhance(image: np.ndarray) -> np.ndarray:
    """Цветное улучшение: CLAHE + резкость (unsharp mask)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3.0)
    sharp = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clean_gray_enhance(image: np.ndarray) -> np.ndarray:
    """Мягкое ч/б улучшение: взвешенное преобразование в градации серого + CLAHE."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def bw_enhance(image: np.ndarray) -> np.ndarray:
    """Высококонтрастный бинарный (black-white) скан."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Адаптивный порог лучше передаёт детали
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_image(image: np.ndarray, scan_mode: str) -> np.ndarray:
    """Применяет выбранный тип улучшения."""
    if scan_mode == "color":
        return color_enhance(image)
    elif scan_mode == "clean_gray":
        return clean_gray_enhance(image)
    elif scan_mode == "bw":
        return bw_enhance(image)
    else:
        return image  # fallback


# ------------------------------------------------------------
# 3. Главная функция process_document
# ------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Основной вход для API.
    :param file_bytes: сырые байты изображения (JPEG/PNG/...)
    :param scan_mode: "color", "clean_gray", "bw"
    :param auto_crop: если True, пытается найти и выпрямить документ;
                      если False, только улучшает исходное изображение.
    :return: (обработанное изображение BGR в формате numpy array, мета-словарь)
    """
    # Декодируем изображение
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение. Поддерживаются JPEG/PNG.")

    meta = {
        "original_size": list(img.shape[:2]),
        "auto_crop_performed": False,
        "scan_mode": scan_mode,
    }

    # Шаг 1: автообрезка / перспектива (если включена)
    if auto_crop:
        contour = find_document_contour(img)
        if contour is not None:
            try:
                warped = four_point_transform(img, contour)
                img = warped
                meta["auto_crop_performed"] = True
                meta["warped_size"] = list(img.shape[:2])
            except Exception as e:
                # При ошибке перспективы используем оригинал и логируем
                meta["warp_error"] = str(e)

    # Шаг 2: улучшение качества
    enhanced = enhance_image(img, scan_mode)

    return enhanced, meta
