import cv2
import numpy as np



def crear_imagen_canal_color(crop_bgr, ch_name, ch_info, k_local_cx, k_local_cy,
                              search_radius=110):
    """
    Genera una imagen derivada exclusiva para aislar el canal ch_name (C, M o Y).

    Sistema híbrido (portado de v2):
      · C  → LAB/BGR (canal b* bajo + comparativa BGR) + HSV boost
      · M  → Solo HSV (con y sin boost de saturación)
      · Y  → HSV (con y sin boost) + LAB (canal b* alto y a* cálido)

    Retorna
    -------
    img_color_isolated : BGR imagen con solo el color detectado sobre fondo oscuro
    mask_full          : máscara binaria completa del color
    mask_near_k        : máscara restringida a la zona cercana a K
    crop_enhanced_bgr  : imagen preprocesada (LAB+CLAHE)
    diag_masks         : dict con máscaras intermedias para diagnóstico
                          {'hsv_sin_boost', 'hsv_con_boost', 'lab_bgr'}
    """
    crop_h, crop_w = crop_bgr.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # --- Preprocesamiento LAB + CLAHE ---
    crop_lab_color = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L_ch, a_ch, b_ch = cv2.split(crop_lab_color)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    L_eq = clahe.apply(L_ch)
    crop_enhanced_bgr = cv2.cvtColor(cv2.merge((L_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    crop_hsv = cv2.cvtColor(crop_enhanced_bgr, cv2.COLOR_BGR2HSV)

    B_i = crop_bgr[:, :, 0].astype(np.int16)
    G_i = crop_bgr[:, :, 1].astype(np.int16)
    R_i = crop_bgr[:, :, 2].astype(np.int16)

    # Máscaras intermedias (para diagnóstico)
    mask_hsv_sin_boost = np.zeros((crop_h, crop_w), dtype=np.uint8)
    mask_lab_bgr       = np.zeros((crop_h, crop_w), dtype=np.uint8)
    color_mask         = np.zeros((crop_h, crop_w), dtype=np.uint8)

    # ================================================================
    # ALGORITMO HÍBRIDO EXCLUSIVO POR CANAL
    # ================================================================
    if ch_name == 'C':  # CYAN: LAB/BGR + HSV boost
        # --- Máscara BGR: canal azul dominante ---
        bgr_mask = (
            (B_i > R_i + 3) & (B_i > G_i + 2) & (B_i > 25)
        ).astype(np.uint8) * 255
        # --- Máscara LAB: b* bajo (contenido azuloso/cian) ---
        lab_mask = cv2.bitwise_and(
            cv2.inRange(b_ch, np.array([0]),  np.array([124])),
            cv2.inRange(L_ch, np.array([15]), np.array([170]))
        )
        mask_lab_bgr = cv2.bitwise_or(bgr_mask, lab_mask)
        # --- Máscara HSV ---
        for (lower, upper) in ch_info['hsv_ranges']:
            mask_hsv_sin_boost = cv2.bitwise_or(mask_hsv_sin_boost,
                                                 cv2.inRange(crop_hsv,       lower, upper))
        #color_mask = mask_lab_bgr.copy()
        color_mask = cv2.bitwise_and(mask_hsv_sin_boost, mask_lab_bgr)

    elif ch_name == 'M':  # MAGENTA: HSV + LAB/BGR opcional (v3.2)
        for (lower, upper) in ch_info['hsv_ranges']:

            mask_hsv_sin_boost = cv2.bitwise_or(mask_hsv_sin_boost,
                                                 cv2.inRange(crop_hsv,       lower, upper))

        color_mask = mask_hsv_sin_boost.copy()

        # v3.2: componente LAB/BGR para magenta diluido
        """
        if ch_info.get('usar_lab_bgr', False):
            lab_m_mask = cv2.bitwise_and(
                cv2.inRange(a_ch, np.array([138]), np.array([255])),
                cv2.inRange(L_ch, np.array([15]),  np.array([200]))
            )
            
            bgr_m_mask = (
                (R_i > B_i + 5) & (R_i > G_i + 5) & (R_i > 30) &
                (np.abs(R_i - B_i) > 10)
            ).astype(np.uint8) * 255
            
            mask_lab_bgr = cv2.bitwise_or(lab_m_mask, bgr_m_mask)
            color_mask   = cv2.bitwise_or(color_mask, mask_lab_bgr)
            #color_mask = mask_lab_bgr.copy()
        """
        magenta_gray = np.clip(R_i - np.maximum(G_i, B_i), 0, 255).astype(np.uint8)
        gray_th = ch_info.get('gray_threshold', 30)
        _, mask_magenta_gray = cv2.threshold(magenta_gray, gray_th, 255, cv2.THRESH_BINARY)
        mask_lab_bgr = mask_magenta_gray.astype(np.uint8)
        color_mask   = cv2.bitwise_or(color_mask, mask_lab_bgr)

    elif ch_name == 'Y':  # AMARILLO: HSV + LAB (b* alto + a* cálido)
        for (lower, upper) in ch_info['hsv_ranges']:
            mask_hsv_sin_boost = cv2.bitwise_or(mask_hsv_sin_boost,
                                                 cv2.inRange(crop_hsv, lower, upper))
        mask_lab_bgr = cv2.bitwise_and(  # ← Asignar a mask_lab_bgr directamente
            cv2.inRange(b_ch, np.array([132]), np.array([255])),
            cv2.inRange(a_ch, np.array([110]), np.array([145])))

        color_mask = mask_lab_bgr.copy()

    # --- Limpieza morfológica ---
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- Restricción a zona cercana a K ---
    search_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.circle(search_mask, (k_local_cx, k_local_cy), search_radius, 255, -1)
    mask_near_k = cv2.bitwise_and(color_mask, search_mask)

    # --- Imagen aislada: píxeles del color sobre fondo muy oscuro ---
    background = (crop_bgr.astype(np.float32) * 0.15).astype(np.uint8)
    img_color_isolated = background.copy()
    img_color_isolated[color_mask > 0] = crop_bgr[color_mask > 0]

    diag_masks = {
        'hsv_sin_boost': mask_hsv_sin_boost,
        'lab_bgr':       mask_lab_bgr,
    }

    return img_color_isolated, color_mask, mask_near_k, crop_enhanced_bgr, diag_masks

