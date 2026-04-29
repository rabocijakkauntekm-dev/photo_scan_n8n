import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Расчет реальной геометрии
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def find_document(image):
    """ Улучшенный поиск: игнорируем текстуру дерева, ищем массу бумаги """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Убираем мелкие детали (текстуру дерева) через сильное размытие
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Адаптивный порог, чтобы выделить белый лист
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Морфология: закрываем дыры внутри листа и убираем мелкий шум снаружи
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (h * w * 0.1): # Если площадь меньше 10%, считаем что не нашли
        return None
        
    # Пытаемся найти именно 4 угла
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx
    
    # Если 4 угла не найдены четко, берем выпуклую оболочку и упрощаем её
    hull = cv2.convexHull(c)
    approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        return approx
        
    return None

def enhance_scan(image):
    """ Удаление теней и эффект 'сканера' """
    # Переходим в LAB для работы с яркостью
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Выравниваем освещение (Division Normalization)
    dilated_l = cv2.dilate(l, np.ones((15, 15), np.uint8))
    bg_l = cv2.medianBlur(dilated_l, 51)
    diff_l = 255 - cv2.absdiff(l, bg_l)
    norm_l = cv2.normalize(diff_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Склеиваем обратно, сохраняя цвета (печати)
    result = cv2.merge([norm_l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # Повышаем резкость
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(result, -1, kernel)

def process_document(image_bytes: bytes, scan_mode: str = "color", auto_crop: bool = True, enhance_level: str = "mild") -> tuple[np.ndarray, dict]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid Image")

    h_orig, w_orig = image.shape[:2]
    
    # Работаем с уменьшенной копией для скорости детекции
    scale = 1000.0 / max(h_orig, w_orig)
    resized = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))
    
    crop_applied = False
    if auto_crop:
        pts = find_document(resized)
        if pts is not None:
            # Возвращаем координаты к оригинальному размеру
            pts = pts.reshape(4, 2) * (1.0 / scale)
            image = four_point_transform(image, pts)
            crop_applied = True

    # Улучшение
    if scan_mode == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    else:
        processed = enhance_scan(image)

    return processed, {"auto_crop": crop_applied, "size": processed.shape[:2]}
