import cv2
import numpy as np
import math

def order_points(pts):
    """ Упорядочивает точки: [top-left, top-right, bottom-right, bottom-left] """
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
    
    # Вычисляем реальную ширину и высоту
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_document_contour(image):
    """ Продвинутый поиск контура на сложном фоне """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Убираем шум, сохраняя края
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Морфологический градиент (выделяет границы объектов лучше, чем Canny)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
    
    # 3. Бинаризация и закрытие дыр
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Если нашли 4 точки и площадь больше 15% кадра
        if len(approx) == 4 and cv2.contourArea(c) > (width * height * 0.15):
            # Проверка на "выпуклость" (чтобы не было самопересечений)
            if cv2.isContourConvex(approx):
                return approx
    return None

def remove_shadows(image):
    """ Метод Division Normalization для идеального белого фона """
    # Разделяем на каналы
    rgb_planes = cv2.split(image)
    result_planes = []
    
    for plane in rgb_planes:
        # Размываем сильно, чтобы получить "карту освещенности"
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        
        # Делим исходник на карту освещенности
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        result_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    return result

def process_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    # 1. Поиск контура
    contour = get_document_contour(img)
    
    if contour is not None:
        # 2. Обрезка по точкам
        transformed = four_point_transform(img, contour.reshape(4, 2))
    else:
        # Если не нашли четкий лист, просто убираем лишнее по краям
        print(f"Контур не найден для {image_path}, использую дефолтную обрезку.")
        h, w = img.shape[:2]
        transformed = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
    
    # 3. Удаление теней и выравнивание фона
    enhanced = remove_shadows(transformed)
    
    # 4. Финальный контраст и резкость
    # Немного приподнимаем контраст, чтобы текст стал чернее
    alpha = 1.2 # Контраст (1.0-3.0)
    beta = -20   # Яркость (отрицательная, чтобы убрать серый налет)
    final = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    cv2.imwrite(save_path, final)

# Пример запуска
# process_image('5420638247186012262.jpg', 'scan_fixed_1.jpg')
