import cv2
import numpy as np

MAX_DETECT_SIDE = 1800


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Flatten paper lighting (shadows/gradients) via background estimation.
    """
    h, w = gray.shape[:2]
    kernel = _odd(max(31, min(h, w) // 6))
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
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

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def resize_for_detection(image: np.ndarray, max_side: int = MAX_DETECT_SIDE) -> tuple[np.ndarray, float]:
    """Resize for faster contour search, return image and scale-back factor."""
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def estimate_paper_mask(image_bgr: np.ndarray, gray_norm: np.ndarray) -> np.ndarray:
    """Estimate largest bright region as document sheet mask."""
    blur = cv2.GaussianBlur(gray_norm, (7, 7), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv[:, :, 1], 0, 70)
    high_val = cv2.inRange(hsv[:, :, 2], 130, 255)
    bright = cv2.bitwise_and(bright, cv2.bitwise_and(low_sat, high_val))

    kernel = np.ones((11, 11), dtype=np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray_norm.shape, 255, dtype=np.uint8)

    h, w = gray_norm.shape[:2]
    min_area = h * w * 0.1
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return np.full(gray_norm.shape, 255, dtype=np.uint8)

    mask = np.zeros_like(gray_norm)
    hull = cv2.convexHull(contour)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask


def detect_document_corners(image_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Detect 4 corners of the white document rectangle."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_illumination(gray)
    h, w = gray_norm.shape[:2]
    image_area = float(h * w)

    # Candidate mask for "white paper" regions.
    blur = cv2.GaussianBlur(gray_norm, (7, 7), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv[:, :, 1], 0, 90)
    high_val = cv2.inRange(hsv[:, :, 2], 120, 255)
    paper_like = cv2.bitwise_and(bright, cv2.bitwise_and(low_sat, high_val))
    paper_like = cv2.morphologyEx(
        paper_like,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2,
    )

    # Candidate mask for strong page borders.
    edges = cv2.Canny(blur, 60, 170)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge_regions = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )

    # Merge both clues: bright sheet + visible borders.
    candidate_mask = cv2.bitwise_or(paper_like, edge_regions)
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "none"

    border_margin = max(8, min(h, w) // 80)
    best_quad: np.ndarray | None = None
    best_score = -1.0
    best_detector = "none"

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15 or area > image_area * 0.97:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            detector = "approx_quad"
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)
            detector = "min_area_rect"

        # Reject full-frame-like boxes.
        near_left = np.sum(quad[:, 0] < border_margin)
        near_right = np.sum(quad[:, 0] > (w - border_margin))
        near_top = np.sum(quad[:, 1] < border_margin)
        near_bottom = np.sum(quad[:, 1] > (h - border_margin))
        if near_left + near_right + near_top + near_bottom >= 3:
            continue

        quad_area = abs(cv2.contourArea(quad))
        if quad_area < image_area * 0.12:
            continue

        # Prefer large and centered page-like candidates.
        cx = float(np.mean(quad[:, 0]))
        cy = float(np.mean(quad[:, 1]))
        center_penalty = abs(cx - (w / 2.0)) / w + abs(cy - (h / 2.0)) / h
        score = (quad_area / image_area) - (0.35 * center_penalty)

        if score > best_score:
            best_score = score
            best_quad = quad
            best_detector = detector

    if best_quad is not None:
        return best_quad, best_detector

    # Fallback pass: take the largest "paper-like" region and fit a rectangle.
    fallback_mask = estimate_paper_mask(image_bgr, gray_norm)
    fallback_hard = cv2.threshold(fallback_mask, 120, 255, cv2.THRESH_BINARY)[1]
    fb_contours, _ = cv2.findContours(fallback_hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not fb_contours:
        return None, "none"

    fb = max(fb_contours, key=cv2.contourArea)
    fb_area = cv2.contourArea(fb)
    if fb_area < image_area * 0.08:
        return None, "none"

    fb_peri = cv2.arcLength(fb, True)
    fb_approx = cv2.approxPolyDP(fb, 0.02 * fb_peri, True)
    if len(fb_approx) == 4:
        return fb_approx.reshape(4, 2).astype(np.float32), "fallback_approx_quad"

    fb_rect = cv2.minAreaRect(fb)
    fb_box = cv2.boxPoints(fb_rect).astype(np.float32)
    return fb_box, "fallback_min_area_rect"


def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """Try to crop and deskew document from input image."""
    resized, scale_to_original = resize_for_detection(image_bgr)
    corners, detector_used = detect_document_corners(resized)
    if corners is None:
        return image_bgr, False, detector_used

    corners = corners * scale_to_original
    warped = four_point_transform(image_bgr, corners)
    if warped is image_bgr:
        return image_bgr, False, "invalid_warp"

    h, w = image_bgr.shape[:2]
    wh, ww = warped.shape[:2]
    if wh * ww < h * w * 0.08:
        return image_bgr, False, "too_small_warp"

    return warped, True, detector_used


def estimate_content_mask(image_bgr: np.ndarray, gray_norm: np.ndarray) -> np.ndarray:
    """Keep text/grid/signatures as foreground content."""
    text_mask = cv2.adaptiveThreshold(
        gray_norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 35, 25), (150, 255, 255))

    content = cv2.bitwise_or(text_mask, blue_mask)
    content = cv2.dilate(content, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return content


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def sharpen_content_region(image_bgr: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Sharpen only document content regions."""
    sharp = unsharp_mask(image_bgr, sigma=0.7, strength=1.0).astype(np.float32)
    base = image_bgr.astype(np.float32)
    mask = cv2.GaussianBlur(content_mask, (5, 5), 0).astype(np.float32) / 255.0
    mask = mask[..., None]
    mixed = (mask * sharp) + ((1.0 - mask) * base)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_color_scan(image_bgr: np.ndarray) -> np.ndarray:
    """Color mode with cleaner paper and stronger legibility."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_illumination(gray)
    paper_mask = estimate_paper_mask(image_bgr, gray_norm)
    content_mask = estimate_content_mask(image_bgr, gray_norm)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = gray_norm
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_lab = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=30, sigmaSpace=30)
    sharp = unsharp_mask(denoised, sigma=0.8, strength=0.6)

    # Whiten paper while preserving text/signatures.
    paper_alpha = (paper_mask.astype(np.float32) / 255.0)[..., None]
    content_alpha = (content_mask.astype(np.float32) / 255.0)[..., None]
    white_bg = np.full_like(sharp, 248)
    whitened = (sharp.astype(np.float32) * 0.65) + (white_bg.astype(np.float32) * 0.35)
    l_float = l_chan.astype(np.float32)[..., None]
    gentle_zone = np.clip((l_float - 115.0) / 90.0, 0.0, 1.0)
    safe_whitened = (1.0 - gentle_zone) * sharp.astype(np.float32) + gentle_zone * whitened
    paper_clean = (1.0 - content_alpha) * safe_whitened + content_alpha * sharp.astype(np.float32)
    mixed = (paper_alpha * paper_clean) + ((1.0 - paper_alpha) * sharp.astype(np.float32))
    result = np.clip(mixed, 0, 255).astype(np.uint8)
    return sharpen_content_region(result, content_mask)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Natural grayscale scan with suppressed shadows."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_illumination(gray)
    paper_mask = estimate_paper_mask(image_bgr, gray)
    content_mask = estimate_content_mask(image_bgr, gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    paper_alpha = paper_mask.astype(np.float32) / 255.0
    content_alpha = content_mask.astype(np.float32) / 255.0
    white_bg = np.full_like(gray, 248)
    whitened = (gray.astype(np.float32) * 0.68) + (white_bg.astype(np.float32) * 0.32)
    gray_float = gray.astype(np.float32)
    gentle_zone = np.clip((gray_float - 115.0) / 90.0, 0.0, 1.0)
    safe_whitened = (1.0 - gentle_zone) * gray_float + gentle_zone * whitened
    paper_clean = (1.0 - content_alpha) * safe_whitened + content_alpha * gray_float
    mixed = (paper_alpha * paper_clean) + ((1.0 - paper_alpha) * gray.astype(np.float32))
    result = np.clip(mixed, 0, 255).astype(np.uint8)

    local_sharp = unsharp_mask(result, sigma=0.6, strength=0.9).astype(np.float32)
    mask = cv2.GaussianBlur(content_mask, (5, 5), 0).astype(np.float32) / 255.0
    merged = (mask * local_sharp) + ((1.0 - mask) * result.astype(np.float32))
    return np.clip(merged, 0, 255).astype(np.uint8)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    """High-contrast black/white mode for printing or OCR."""
    clean_gray = enhance_clean_gray(image_bgr)
    bw = cv2.adaptiveThreshold(
        clean_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        15,
    )
    kernel = np.ones((2, 2), dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw


def process_document(
    image_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = False,
) -> tuple[np.ndarray, dict]:
    """Process document image: optional auto-crop + enhancement."""
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
    else:
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
