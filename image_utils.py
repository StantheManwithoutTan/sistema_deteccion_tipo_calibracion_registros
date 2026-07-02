import cv2
import numpy as np


def sharpen_image(bgr_img, strength=1.0):
    """
    Aplica unsharp masking para mejorar la nitidez de la imagen.

    Parámetros
    ----------
    bgr_img  : imagen BGR (uint8)
    strength : intensidad del sharpening
               0.0 → sin cambio
               0.5 → suave  (para imágenes ya relativamente nítidas)
               1.0 → normal (recomendado)
               2.0 → fuerte (puede generar halos en bordes muy contrastados)

    Retorna imagen BGR sharpened (uint8).
    """
    if strength == 0:
        return bgr_img.copy()
    # Gaussian blur como máscara base
    blur = cv2.GaussianBlur(bgr_img, (0, 0), sigmaX=3)
    # Unsharp: img_sharp = (1 + s)*original − s*blur
    sharpened = cv2.addWeighted(bgr_img, 1.0 + strength, blur, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image(bgr_img):
    """LAB → CLAHE sobre canal L → devuelve imagen LAB mejorada."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    return cv2.merge((L, a, b))


def create_crosshair_template(size=101, ring_radius=40, ring_thickness=8,
                               cross_thickness=10, cross_length=90):
    """Genera plantilla sintética de marca de registro (círculo + cruz). Binaria blanco/negro."""
    template = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    half_cross = cross_length // 2
    cv2.line(template, (center, center - half_cross), (center, center + half_cross), 255, cross_thickness)
    cv2.line(template, (center - half_cross, center), (center + half_cross, center), 255, cross_thickness)
    cv2.circle(template, (center, center), ring_radius, 255, ring_thickness)
    return template


def multi_scale_template_match(gray, template, scales=None, threshold=0.5):
    """Template matching a múltiples escalas. Devuelve lista de (x, y, score, scale)."""
    if scales is None:
        scales = generar_escalas_log(0.3, 2.5, 20)
    detections = []
    th, tw = template.shape[:2]
    for s in scales:
        new_w = int(tw * s)
        new_h = int(th * s)
        if new_w < 12 or new_h < 12:  # antes era 20
            continue
        if new_w > gray.shape[1] or new_h > gray.shape[0]:
            continue
        resized_template = cv2.resize(template, (new_w, new_h))
        gray_inv = cv2.bitwise_not(gray)
        result = cv2.matchTemplate(gray_inv, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        for pt_y, pt_x in zip(*locations):
            cx = pt_x + new_w // 2
            cy = pt_y + new_h // 2
            score = result[pt_y, pt_x]
            detections.append((cx, cy, float(score), s))
    return detections



# En actualidad, podemos eliminar esta funcion porque solo necesitamos que devuelve la primera deteccion (la de mayor score) y no todas las detecciones no max supresion. Sin embargo, la dejamos por si en el futuro queremos obtener mas de una deteccion.
def non_max_suppression(detections, radius):
    """Elimina duplicados: si dos detecciones están a menos de `radius` px, queda la de mayor score."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for det in detections:
        cx, cy, score, scale = det
        is_duplicate = False
        for k in kept:
            dist = np.hypot(cx - k[0], cy - k[1])
            if dist < radius:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)
    return kept[:1]


# En image_utils.py, cambiar default y/o crear función auxiliar
def generar_escalas_log(start=0.15, stop=3.2, num=25):
    return np.logspace(np.log10(start), np.log10(stop), num=num)



def normalize_white_background(crop_bgr, l_threshold=150, min_tolerance=15, 
                               max_tolerance=50, sigma_factor=1.0):
    """
    Normaliza el fondo blanco de un crop detectando el color más prominente
    y reemplazando píxeles cercanos a ese color con blanco puro.
    
    Estrategia:
      1. Convertir a LAB
      2. Filtrar píxeles claros (L > l_threshold)
      3. Crear histograma de píxeles claros
      4. Encontrar color más frecuente (bin más poblado)
      5. Calcular media y desviación estándar de ese cluster
      6. Crear máscara de píxeles dentro de σ (con límites min/max)
      7. Reemplazar con blanco puro
    
    Parámetros
    ----------
    crop_bgr : imagen BGR (uint8)
    l_threshold : mínima luminancia para considerar píxel "claro" (0-255)
    min_tolerance : tolerancia mínima en RGB equivalente
    max_tolerance : tolerancia máxima en RGB equivalente
    sigma_factor : multiplicador de desviación estándar (1.0 = ±σ)
    
    Retorna
    -------
    crop_normalized : imagen BGR con fondo blanco normalizado
    """
    crop_h, crop_w = crop_bgr.shape[:2]
    
    # --- Convertir a LAB ---
    crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L_ch, a_ch, b_ch = cv2.split(crop_lab)
    
    # --- Filtrar píxeles claros (L > threshold) ---
    mask_bright = L_ch > l_threshold
    if not np.any(mask_bright):
        # Si no hay píxeles claros, retornar sin cambios
        return crop_bgr.copy()
    
    # --- Crear histograma de píxeles claros en espacio (a*, b*) ---
    # Usamos 32 bins en cada dimensión para a* y b*
    a_bright = a_ch[mask_bright]
    b_bright = b_ch[mask_bright]
    
    hist_2d, a_edges, b_edges = np.histogram2d(
        a_bright, b_bright, bins=[32, 32], range=[[0, 256], [0, 256]]
    )
    
    # --- Encontrar bin más frecuente ---
    max_bin_idx = np.unravel_index(np.argmax(hist_2d), hist_2d.shape)
    a_bin_idx, b_bin_idx = max_bin_idx
    
    # Centro del bin más frecuente (aproximadamente)
    a_center = (a_edges[a_bin_idx] + a_edges[a_bin_idx + 1]) / 2
    b_center = (b_edges[b_bin_idx] + b_edges[b_bin_idx + 1]) / 2
    
    # --- Calcular media y desviación estándar del cluster ---
    # Píxeles dentro de ±5 bins del centro
    a_tol = (a_edges[1] - a_edges[0]) * 2.5  # ~±2.5 bins = 20 unidades
    b_tol = (b_edges[1] - b_edges[0]) * 2.5
    
    cluster_mask = (
        (np.abs(a_ch.astype(np.float32) - a_center) < a_tol) &
        (np.abs(b_ch.astype(np.float32) - b_center) < b_tol) &
        (L_ch > l_threshold)
    )
    
    if not np.any(cluster_mask):
        return crop_bgr.copy()
    
    # Media del cluster
    a_mean = np.mean(a_ch[cluster_mask])
    b_mean = np.mean(b_ch[cluster_mask])
    
    # Desviación estándar
    a_std = np.std(a_ch[cluster_mask])
    b_std = np.std(b_ch[cluster_mask])
    L_std = np.std(L_ch[cluster_mask])
    
    # --- Convertir σ en LAB a tolerancia en RGB equivalente ---
    # Aproximación: σ_lab ≈ σ_rgb * factor
    # (usar promedios de las desviaciones)
    sigma_lab_avg = (a_std + b_std) / 2.0
    tolerance_rgb = sigma_lab_avg * sigma_factor
    
    # Aplicar límites
    tolerance_rgb = np.clip(tolerance_rgb, min_tolerance, max_tolerance)
    
    # --- Crear máscara: píxeles dentro de ±tolerance en LAB ---
    dist_to_mean = np.sqrt(
        ((a_ch.astype(np.float32) - a_mean) ** 2) +
        ((b_ch.astype(np.float32) - b_mean) ** 2)
    )
    
    mask_to_normalize = (dist_to_mean <= tolerance_rgb) & (L_ch > l_threshold)
    
    # --- Reemplazar con blanco puro ---
    crop_normalized = crop_bgr.copy()
    crop_normalized[mask_to_normalize] = [255, 255, 255]  # BGR: blanco
    
    return crop_normalized