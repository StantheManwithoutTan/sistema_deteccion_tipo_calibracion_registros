import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (WHITE_BG_NORMALIZE, 
                    SIZE_ADAPTIVE_ENABLED, MIN_ROI_MARGIN,
MIN_SEARCH_RADIUS, PX_MIN_ACCEPT_LOW, SCORE_NORMALIZER_LOW, CMY_CROP_RANGES, WHITE_L_THRESHOLD, WHITE_SIGMA_FACTOR, WHITE_TOLERANCE_MAX, WHITE_TOLERANCE_MIN)

from detection import analyze_hue_range, crear_imagen_canal_color, get_representative_color
from detection.position import compute_detect_radius, detectar_posicion
from detection.visualization import crear_overlay_canal, plot_diagnostico_canal
from image_utils import multi_scale_template_match, non_max_suppression, normalize_white_background, preprocess_image


def detectar_canal_con_imagen_separada(img_bgr, ch_name, ch_info, k_marks,
                                        template, roi_margin=230,
                                        search_radius=110, threshold=0.2,
                                        px_min_accept=800, score_normalizer=300,
                                        show_plots=False):
    marks_canal = []
    diag_data_list = []
    img_h, img_w = img_bgr.shape[:2]

    for mark_idx, (kcx, kcy, kscore, kscale) in enumerate(k_marks[:1]):
        # 1. Crop
        rx1 = max(int(kcx) - roi_margin, 0)
        ry1 = max(int(kcy) - roi_margin, 0)
        rx2 = min(int(kcx) + roi_margin, img_w)
        ry2 = min(int(kcy) + roi_margin, img_h)
        crop_bgr = img_bgr[ry1:ry2, rx1:rx2].copy()
        k_local_cx, k_local_cy = int(kcx) - rx1, int(kcy) - ry1

        # 2. Normalizar + máscara de color
        if WHITE_BG_NORMALIZE:
            crop_bgr = normalize_white_background(
            crop_bgr,
            l_threshold=WHITE_L_THRESHOLD,
            min_tolerance=WHITE_TOLERANCE_MIN,
            max_tolerance=WHITE_TOLERANCE_MAX,
            sigma_factor=WHITE_SIGMA_FACTOR
        )
        img_isolated, mask_full, mask_near, crop_enhanced, diag_masks = \
            crear_imagen_canal_color(crop_bgr, ch_name, ch_info, k_local_cx, k_local_cy, search_radius)

        # 3. Análisis de color
        rgb_color, hsv_color, px_count = get_representative_color(mask_near, crop_bgr)
        hue_analysis = analyze_hue_range(cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2HSV), mask_near, ch_info)

        # 4. Detectar posición (EXTRACCIÓN 1)
        best_cx, best_cy, best_score, best_scale, method_used, xs, ys = \
            detectar_posicion(mask_near, img_isolated, template,
                              kcx, kcy, kscale, rx1, ry1, k_local_cx, k_local_cy,
                              threshold, score_normalizer, px_count)
        marks_canal.append((best_cx, best_cy, best_score, best_scale))

        # 5. Radio de detección (EXTRACCIÓN 2)
        detect_radius = compute_detect_radius(xs, ys, best_cx, best_cy, best_scale, rx1, ry1)

        # 6. Overlay (EXTRACCIÓN 3)
        draw_color_bgr = ch_info.get('color_display', (200, 200, 0))
        local_det_cx, local_det_cy = int(best_cx) - rx1, int(best_cy) - ry1
        overlay_near = crear_overlay_canal(crop_bgr, mask_near, px_count,
                                           k_local_cx, k_local_cy, search_radius,
                                           draw_color_bgr, local_det_cx, local_det_cy, detect_radius)

        # 7. Visualización (EXTRACCIÓN 4)
        plot_diagnostico_canal(crop_bgr, crop_enhanced, img_isolated, overlay_near,
                               mask_full, mask_near, diag_masks, ch_name, ch_info,
                               k_local_cx, k_local_cy, search_radius, px_count,
                               best_cx, best_cy, best_score, rgb_color, hsv_color,
                               hue_analysis, crop_bgr.shape[1], crop_bgr.shape[0],
                               mark_idx, show_plots)

        # 8. Guardar datos de diagnóstico
        diag_data_list.append({
        'crop_bgr':      crop_bgr,
        'crop_enhanced': crop_enhanced,
        'img_isolated':  img_isolated,
        'overlay_near':  overlay_near,
        'diag_masks':    diag_masks,
        'mask_full':     mask_full,
        'mask_near':     mask_near,
        'px_count':      px_count,
        'k_local_cx':    k_local_cx,
        'k_local_cy':    k_local_cy,
    })

    return marks_canal, diag_data_list




def procesar_imagen_completa(img_bgr, template, roi_margin=230, search_radius=110):
    """
    Procesa una imagen completa: detecta K, luego C/M/Y.
    Retorna: (cmyk_marks, diag_por_canal, k_marks)
    """
    # PASO 1: Detectar K
    lab_prep = preprocess_image(img_bgr)
    L_full, _, _ = cv2.split(lab_prep)
    k_detections = multi_scale_template_match(L_full, template, 
                                              scales=np.arange(0.15, 3.2, 0.1),
                                              threshold=0.35)
    nms_radius = int(110 * np.sqrt(k_detections[0][3])) if SIZE_ADAPTIVE_ENABLED and k_detections else 110
    k_marks = non_max_suppression(k_detections, radius=110)
    
    if len(k_marks) == 0:
        return None, None, k_marks

    if SIZE_ADAPTIVE_ENABLED:
        avg_k_score = np.mean([km[3] for km in k_marks])
        f = np.sqrt(avg_k_score)
        dyn_roi_margin    = max(MIN_ROI_MARGIN,    int(230 * f))
        dyn_search_radius    = max(MIN_SEARCH_RADIUS, int(110 * f))
        dyn_pixel_accept_amount    = max(PX_MIN_ACCEPT_LOW, int(800 * avg_k_score))
        dyn_normalizer_score  = max(SCORE_NORMALIZER_LOW, int(300 * avg_k_score))
    else:
        # valores defectos en el caso que la deteccion inicial no da una calificacion de K utilizable
        dyn_roi_margin, dyn_search_radius, dyn_pixel_accept_amount, dyn_normalizer_score = roi_margin, search_radius, 800, 300

    
    # PASO 2: Detectar CMY
    cmyk_marks = {'K': k_marks}
    diag_por_canal = {}
    
    for ch_name, ch_info in CMY_CROP_RANGES.items():
        marks_canal, diag_data = detectar_canal_con_imagen_separada(
            img_bgr, ch_name, ch_info, k_marks, template,
            roi_margin=dyn_roi_margin, search_radius=dyn_search_radius, threshold=0.1,
            px_min_accept=dyn_pixel_accept_amount, score_normalizer=dyn_normalizer_score,
        )
        
        # Filtrar por 1000 píxeles
        if len(diag_data) > 0 and diag_data[0].get('px_count', 0) >= dyn_pixel_accept_amount:
            cmyk_marks[ch_name] = marks_canal
            diag_por_canal[ch_name] = diag_data
        else:
            cmyk_marks[ch_name] = []
    
    return cmyk_marks, diag_por_canal, k_marks
