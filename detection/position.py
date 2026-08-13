import cv2
import numpy as np

from config import (CLUSTERING_SIGMA)

from detection.color_analysis import weighted_median
from image_utils import multi_scale_template_match

def detectar_posicion(mask_near, img_isolated, template,
                      kcx, kcy, kscale,
                      rx1, ry1, k_local_cx, k_local_cy,
                      threshold=0.2, score_normalizer=300, px_count=0):
    """
    Detecta posición de una marca de color usando:
    1. Template matching sobre imagen aislada
    2. Fallback: centroide ponderado
    3. Refinamiento: mediana ponderada

    Retorna:
      (best_cx, best_cy)    : coordenadas globales
      best_score, best_scale: score y escala
      method_used           : string descriptivo
      xs, ys                : píxeles detectados (para radio)
    """
    best_cx, best_cy, best_score, best_scale = kcx, kcy, 0.05, kscale
    method_used = 'prediccion'

    if px_count > 30:
        isolated_gray = cv2.cvtColor(img_isolated, cv2.COLOR_BGR2GRAY)
        clahe2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        isolated_enhanced = clahe2.apply(isolated_gray)
        test_scales = np.arange(max(0.2, kscale - 0.2), kscale + 0.3, 0.05)
        local_detections = multi_scale_template_match(
            isolated_enhanced, template, scales=test_scales, threshold=threshold
        )
        if local_detections:
            local_detections.sort(key=lambda x: x[2], reverse=True)
            dcx, dcy, dscore, dscale = local_detections[0]
            best_cx, best_cy = dcx + rx1, dcy + ry1
            best_score, best_scale = dscore, dscale
            method_used = 'template_img_separada'

    # Extrae las coordenadas locales de todos los pixeles de la mascara de color
    ys, xs = np.where(mask_near > 0)
    if method_used == 'prediccion' and len(xs) > 0:
        # A: fallback invariante a K — mean-shift desde el centroide simple
        dists   = np.hypot(xs - k_local_cx, ys - k_local_cy)
        weights = np.exp(-dists / 20.0)
        best_cx = int(np.average(xs, weights=weights)) + rx1
        best_cy = int(np.average(ys, weights=weights)) + ry1
        best_score = round(min(px_count / score_normalizer, 1.0), 3)
        best_scale = kscale
        method_used = 'centroide_ponderado'

    if len(xs) > 0:
        # B: no sobrescribir el template — solo refinar local (sigma ~10px)
        dists_fin   = np.hypot(xs - k_local_cx, ys - k_local_cy)
        weights_fin = np.exp(-dists_fin / CLUSTERING_SIGMA)
        best_cx     = int(weighted_median(xs, weights_fin)) + rx1
        best_cy     = int(weighted_median(ys, weights_fin)) + ry1
        method_used += '+mediana_ponderada'
        
    return best_cx, best_cy, best_score, best_scale, method_used, xs, ys

def compute_detect_radius(xs, ys, best_cx, best_cy, best_scale, rx1, ry1):
    """
    Calcula el radio del círculo que representa la marca detectada,
    basado en el percentil 85 de distancias desde el centro.
    """
    if len(xs) > 3:
        xc = int(best_cx - rx1)
        yc = int(best_cy - ry1)
        dists = np.hypot(xs - xc, ys - yc)
        real_radius = int(np.percentile(dists, 85)) + 2
        detect_radius = max(int(40 * best_scale), real_radius)
    else:
        detect_radius = int(40 * best_scale)
    return detect_radius

