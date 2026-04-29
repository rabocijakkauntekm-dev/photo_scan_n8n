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
    
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")
    
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))

def find_document_contour(image):
    """ Ищем документ как самый крупный светлый объект, игнорируя текстуру стола """
    h, w = image.shape[:2]
    # Уходим в LAB, канал L (яркость) лучше всего отделяет бумагу от стола
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    
    # Размываем ОЧЕНЬ сильно, чтобы 'растворить' текстуру дерева
    blurred = cv2.GaussianBlur(l, (15, 15), 0)
    
    # Используем Otsu для выделения бумаги
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Убираем мелкие дырки внутри листа (текст) и мусор снаружи
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (h * w * 0.15): # Если меньше 15% кадра — это не документ
        return None
        
    # Пытаемся найти 4 угла
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx
    
    # Если 4 угла не найдены (например, лист немного закруглен), берем Convex Hull
    hull = cv2.convexHull(c)
    approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        return approx
        
    return None

def enhance_image(image):
    """ Эффект сканера: убираем тени, выравниваем фон """
    # Выравнивание освещения
    rgb_planes = cv2.split(image)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 31)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        result_planes.append(norm_img)
    
    result = cv2.merge(result_planes)
    
    # Финальный контраст
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=-10)
    return result

def process_document(image_bytes: bytes, scan_mode: str = "color", auto_crop: bool = True, enhance_level: str = "mild") -> tuple[np.ndarray, dict]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed")

    h_orig, w_orig = img.shape[:2]
    
    # Работаем с уменьшенной копией для поиска контура (быстрее и меньше шума)
    scale = 800.0 / max(h_orig, w_orig)
    resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    
    crop_applied = False
    working_img = img

    if auto_crop:
        cnt = find_document_contour(resized)
        if cnt is not None:
            # Масштабируем точки обратно
            pts = cnt.reshape(4, 2) * (1.0 / scale)
            working_img = four_point_transform(img, pts)
            crop_applied = True
        else:
            # Если не нашли — просто чуть-чуть обрезаем края (техническая рамка)
            margin = 0.02
            working_img = img[int(h_orig*margin):int(h_orig*(1-margin)), 
                              int(w_orig*margin):int(w_orig*(1-margin))]

    # Улучшение (ч/б или цвет)
    if scan_mode == "bw":
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 25, 10)
    else:
        final = enhance_image(working_img)

    meta = {
        "auto_crop": crop_applied,
        "width": int(final.shape[1]),
        "height": int(final.shape[0])
    }
    
    return final, meta
