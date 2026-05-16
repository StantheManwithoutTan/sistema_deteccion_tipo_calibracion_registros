import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import CLUSTERING_SIGMA
from image_utils import multi_scale_template_match


def get_representative_color(mask, bgr_img):
    """
    Dado una máscara binaria y la imagen original BGR,
    calcula el color representativo (mediana) de los píxeles detectados.
    Devuelve (rgb_tuple, hsv_tuple, pixel_count).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, 0
    pixels_bgr = bgr_img[ys, xs]
    med_b = int(np.median(pixels_bgr[:, 0]))
    med_g = int(np.median(pixels_bgr[:, 1]))
    med_r = int(np.median(pixels_bgr[:, 2]))
    rgb = (med_r, med_g, med_b)
    pixel_bgr_sample = np.array([[[med_b, med_g, med_r]]], dtype=np.uint8)
    pixel_hsv = cv2.cvtColor(pixel_bgr_sample, cv2.COLOR_BGR2HSV)
    hsv = (int(pixel_hsv[0,0,0]), int(pixel_hsv[0,0,1]), int(pixel_hsv[0,0,2]))
    return rgb, hsv, len(xs)

    

def weighted_median(values, weights):
    """
    Calcula la mediana ponderada de un conjunto de valores.
    
    Retorna el valor cuya suma acumulada de pesos alcanza o supera el 50% del total.
    """
    if len(values) == 0:
        return 0
    
    # Ordenar valores y pesos según valores
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calcular suma acumulada de pesos
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]
    
    # Encontrar el índice donde cumsum >= 50% del total
    median_idx = np.searchsorted(cumsum_weights, total_weight / 2.0)
    median_idx = np.clip(median_idx, 0, len(sorted_values) - 1)
    
    return sorted_values[median_idx]



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


def detectar_canal_con_imagen_separada(img_bgr, ch_name, ch_info, k_marks,
                                        template, roi_margin=230,
                                        search_radius=110, threshold=0.2,
                                        show_plots=False):  
    """
    Para cada marca K detectada:
      1. Crea imagen derivada exclusiva para el canal ch_name.
      2. Muestra diagnóstico de máscaras (HSV sin boost | LAB/BGR | overlay).
      3. Detecta posición del canal con template matching + refinamiento por mediana.

    Retorna: (marks_canal, diag_data_list)
      marks_canal     : lista de (cx, cy, score, scale)
      diag_data_list  : lista de dicts con imágenes para el diagnóstico combinado
    """
    marks_canal    = []
    diag_data_list = []
    img_h, img_w   = img_bgr.shape[:2]

    print(f"\n{'='*65}")
    print(f"  CANAL {ch_name} ({ch_info['nombre']}) — Detección con imagen separada")
    print(f"{'='*65}")

    for mark_idx, (kcx, kcy, kscore, kscale) in enumerate(k_marks[:1]):  # v3.2: solo primera máscara
        print(f"\n  ► Marca K-{mark_idx} en ({int(kcx)},{int(kcy)}), scale={kscale:.2f}")

        rx1 = max(int(kcx) - roi_margin, 0)
        ry1 = max(int(kcy) - roi_margin, 0)
        rx2 = min(int(kcx) + roi_margin, img_w)
        ry2 = min(int(kcy) + roi_margin, img_h)

        crop_bgr    = img_bgr[ry1:ry2, rx1:rx2].copy()
        crop_h, crop_w = crop_bgr.shape[:2]
        k_local_cx  = int(kcx) - rx1
        k_local_cy  = int(kcy) - ry1

        # ── PASO 1: Imagen separada + máscaras diagnóstico ──
        img_isolated, mask_full, mask_near, crop_enhanced, diag_masks = crear_imagen_canal_color(
            crop_bgr, ch_name, ch_info, k_local_cx, k_local_cy, search_radius
        )

        # ── PASO 2: Color representativo ──
        rgb_color, hsv_color, px_count = get_representative_color(mask_near, crop_bgr)

        print(f"  ┌─ Píxeles detectados cerca de K: {px_count}")
        if rgb_color is not None:
            print(f"  │  Color representativo RGB : R={rgb_color[0]:3d}, G={rgb_color[1]:3d}, B={rgb_color[2]:3d}")
            print(f"  │  Color representativo HSV : H={hsv_color[0]:3d}°, S={hsv_color[1]:3d}, V={hsv_color[2]:3d}")
        else:
            print(f"  │  ⚠ No se detectaron píxeles del canal {ch_name} cerca de K.")

        # ── PASO 2.5: Análisis de rango H observado ──
        crop_hsv_local = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2HSV)
        hue_analysis = analyze_hue_range(crop_hsv_local, mask_near, ch_info)
            
        print(f"  ┌─ ANÁLISIS DE HUE:")
        if hue_analysis['pixel_count'] > 0:
            print(f"  │  Rango H observado: {hue_analysis['h_min']}–{hue_analysis['h_max']}°")
            print(f"  │  H promedio: {hue_analysis['h_mean']}° | moda: {hue_analysis['h_mode']}° | σ: {hue_analysis['h_std']}")
            print(f"  │  Saturación promedio: {hue_analysis['s_mean']} | Brillo promedio: {hue_analysis['v_mean']}")
            print(f"  │  Rangos configurados:")
            for i, (lower, upper) in enumerate(ch_info['hsv_ranges']):
                print(f"  │    [{i}] H: {lower[0]}–{upper[0]}, S: {lower[1]}–{upper[1]}, V: {lower[2]}–{upper[2]}")
            else:
                print(f"  │  (Sin píxeles detectados)")
            print(f"  └─")

        # ── PASO 3: Detección de posición (template + centroide + mediana) ──
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

        # Fallback centroide ponderado
        ys, xs = np.where(mask_near > 0)
        if method_used == 'prediccion' and len(xs) > 0:
            dists   = np.hypot(xs - k_local_cx, ys - k_local_cy)
            weights = np.exp(-dists / 20.0)
            best_cx = int(np.average(xs, weights=weights)) + rx1
            best_cy = int(np.average(ys, weights=weights)) + ry1
            best_score  = round(min(px_count / 300.0, 1.0), 3)
            best_scale  = kscale
            method_used = 'centroide_ponderado'

        # v3.2: refinamiento por mediana ponderada
        if len(xs) > 0:
            dists_fin   = np.hypot(xs - k_local_cx, ys - k_local_cy)
            weights_fin = np.exp(-dists_fin / CLUSTERING_SIGMA)
            best_cx     = int(weighted_median(xs, weights_fin)) + rx1
            best_cy     = int(weighted_median(ys, weights_fin)) + ry1
            method_used += '+mediana_ponderada'

        print(f"  │  Método: {method_used}")
        print(f"  └─ Posición global: ({int(best_cx)},{int(best_cy)})  score={best_score:.3f}")

        marks_canal.append((best_cx, best_cy, best_score, best_scale))

        # ── PASO 4: Overlay 'cerca de K' para diagnóstico ──
        draw_color_bgr = ch_info.get('color_display', (200, 200, 0))
        overlay_near_colored = crop_bgr.copy()
        if px_count > 0:
            overlay_near_colored[mask_near > 0] = draw_color_bgr
        cv2.circle(overlay_near_colored, (k_local_cx, k_local_cy), search_radius, (200,200,200), 1)
        cv2.drawMarker(overlay_near_colored, (k_local_cx, k_local_cy), (200,200,200),
                       cv2.MARKER_CROSS, 16, 1)
        local_det_cx = int(best_cx) - rx1
        local_det_cy = int(best_cy) - ry1
        cv2.circle(overlay_near_colored, (local_det_cx, local_det_cy),
                   int(40 * best_scale), draw_color_bgr, 2)
        cv2.circle(overlay_near_colored, (local_det_cx, local_det_cy), 5, draw_color_bgr, -1)

        # ── PASO 5: Visualización — Diagnóstico de máscaras (vista V2-style) ──
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))  # Cambio: de 2,4 a 2,5
        fig.suptitle(
            f"Diagnóstico de máscaras — Canal {ch_name} ({ch_info['nombre']}) | "
            f"Marca K-{mark_idx}\n"
            f"Posición: ({int(best_cx)},{int(best_cy)})  Score: {best_score:.3f}  "
            f"RGB: {rgb_color}  HSV: {hsv_color}  Píxeles cerca K: {px_count}",
            fontsize=10, fontweight='bold'
        )

        # Fila 0: imágenes de color
        panels_top = [
            (cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), f'Original crop ({crop_w}×{crop_h})'),
            (cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB), 'Preprocesada (LAB+CLAHE)'),
            (cv2.cvtColor(img_isolated, cv2.COLOR_BGR2RGB), f'Imagen aislada — canal {ch_name}'),
            (cv2.cvtColor(overlay_near_colored, cv2.COLOR_BGR2RGB),
             f'Overlay cerca K (r={search_radius}px, {px_count}px)'),
        ]
        
        # Fila 1: máscaras
        mask_full_3ch = cv2.cvtColor(mask_full,   cv2.COLOR_GRAY2BGR)
        mask_near_3ch = cv2.cvtColor(mask_near,   cv2.COLOR_GRAY2BGR)
        mask_hsv_sb   = cv2.cvtColor(diag_masks['hsv_sin_boost'], cv2.COLOR_GRAY2BGR)
        mask_lbgr     = cv2.cvtColor(diag_masks['lab_bgr'], cv2.COLOR_GRAY2BGR)
        
        panels_bot = [
            (cv2.cvtColor(mask_hsv_sb, cv2.COLOR_BGR2RGB),
            f'Máscara HSV ({cv2.countNonZero(diag_masks["hsv_sin_boost"])}px)'),
            (cv2.cvtColor(mask_lbgr, cv2.COLOR_BGR2RGB),
            f'Máscara LAB/BGR ({cv2.countNonZero(diag_masks["lab_bgr"])}px)'),
            (cv2.cvtColor(mask_near_3ch, cv2.COLOR_BGR2RGB),
            f'Máscara cerca K ({px_count}px)'),
            (cv2.cvtColor(mask_near_3ch, cv2.COLOR_BGR2RGB),
             f'Máscara usada en resultado ({px_count}px)'),
        ]

        # Rellenar primeras 4 columnas de fila superior
        for col, (img, title) in enumerate(panels_top):
            axes[0][col].imshow(img)
            axes[0][col].set_title(title, fontsize=8)
            axes[0][col].axis('off')
            axes[0][col].plot(k_local_cx, k_local_cy, '+', color='gray',
                              markersize=12, markeredgewidth=1.5)

        # Rellenar primeras 4 columnas de fila inferior
        for col, (img, title) in enumerate(panels_bot):
            axes[1][col].imshow(img)
            axes[1][col].set_title(title, fontsize=8)
            axes[1][col].axis('off')

        # ── Columna 5 (índice 4): Histograma de H + análisis ──
        ax_hue = axes[0][4]
        if hue_analysis['pixel_count'] > 0:
            ax_hue.hist(hue_analysis['h_values'], bins=180, range=(0, 180), 
                       color='lightblue', edgecolor='black', alpha=0.7)
            ax_hue.axvline(hue_analysis['h_min'], color='green', linestyle='--', 
                          linewidth=2, label=f"Min: {hue_analysis['h_min']}")
            ax_hue.axvline(hue_analysis['h_max'], color='red', linestyle='--', 
                          linewidth=2, label=f"Max: {hue_analysis['h_max']}")
            ax_hue.axvline(hue_analysis['h_mean'], color='blue', linestyle='-', 
                          linewidth=2, label=f"Mean: {hue_analysis['h_mean']}")
            
            # Overlay de rangos configurados como bandas
            for lower, upper in ch_info['hsv_ranges']:
                h_low = lower[0]
                h_high = upper[0]
                ax_hue.axvspan(h_low, h_high, alpha=0.1, color='orange')
            
            ax_hue.set_xlabel('Hue (°)', fontsize=8)
            ax_hue.set_ylabel('Frecuencia', fontsize=8)
            ax_hue.set_title('Distribución de Hue\n(bandas naranjas = rangos config)', fontsize=8)
            ax_hue.legend(fontsize=7)
            ax_hue.grid(True, alpha=0.3)
        else:
            ax_hue.text(0.5, 0.5, 'Sin datos\npara graficar', 
                       ha='center', va='center', transform=ax_hue.transAxes)
            ax_hue.set_title('Distribución de Hue', fontsize=8)
            ax_hue.axis('off')

        # Columna 5 fila 2: Tabla de resumen
        ax_text = axes[1][4]
        ax_text.axis('off')
        if hue_analysis['pixel_count'] > 0:
            text_summary = (
                f"H observado: {hue_analysis['h_min']}–{hue_analysis['h_max']}°\n"
                f"H media: {hue_analysis['h_mean']}° (σ={hue_analysis['h_std']})\n"
                f"H moda: {hue_analysis['h_mode']}°\n"
                f"─────────\n"
                f"S promedio: {hue_analysis['s_mean']}\n"
                f"V promedio: {hue_analysis['v_mean']}\n"
                f"Píxeles: {hue_analysis['pixel_count']}"
            )
            ax_text.text(0.1, 0.95, text_summary, transform=ax_text.transAxes,
                        fontsize=8, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax_text.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                        transform=ax_text.transAxes)

        plt.tight_layout(rect=[0.04, 0.0, 1.0, 0.95])
        if show_plots:
            plt.show()
        else:
            plt.close(fig)  

        diag_data_list.append({
            'crop_bgr':      crop_bgr,
            'crop_enhanced': crop_enhanced,
            'img_isolated':  img_isolated,
            'overlay_near':  overlay_near_colored,
            'diag_masks':    diag_masks,
            'mask_full':     mask_full,
            'mask_near':     mask_near,
            'px_count':      px_count,
            'k_local_cx':    k_local_cx,
            'k_local_cy':    k_local_cy,
        })

    return marks_canal, diag_data_list


def generar_diagnostico_mascaras(diag_por_canal, preprocesada_titulo='PREPROCESADA', show_plot=True):
    """
    Genera el diagnóstico combinado de máscaras para todos los canales CMY
    en una sola figura (3 filas × 4 columnas), similar a la vista de v2.

    diag_por_canal : dict {'C': diag_data_list, 'M': ..., 'Y': ...}
    """
    ch_order = ['C', 'M', 'Y']
    col_titles = [
        'Imagen (PREPROCESADA)',
        'Máscara HSV',
        'Máscara LAB/BGR',
        'Overlay cerca K',
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 12))
    fig.suptitle(f'Diagnóstico de máscaras — {preprocesada_titulo}',
                 fontsize=14, fontweight='bold')

    for row, ch_name in enumerate(ch_order):
        data_list = diag_por_canal.get(ch_name, [])
        if not data_list:
            for col in range(4):
                axes[row][col].set_visible(False)
            continue
        d = data_list[0]  # Usar primera marca K

        hsv_sb_cnt  = cv2.countNonZero(d['diag_masks']['hsv_sin_boost'])
        lab_cnt     = cv2.countNonZero(d['diag_masks']['lab_bgr'])

        panels = [
            cv2.cvtColor(d['crop_enhanced'],                   cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.cvtColor(d['diag_masks']['hsv_sin_boost'],
                                      cv2.COLOR_GRAY2BGR),      cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.cvtColor(d['diag_masks']['lab_bgr'],
                                      cv2.COLOR_GRAY2BGR),      cv2.COLOR_BGR2RGB),
            cv2.cvtColor(d['overlay_near'],                    cv2.COLOR_BGR2RGB),
        ]
        subtitles = [
            f'{ch_name} — Imagen ({d["crop_bgr"].shape[1]}×{d["crop_bgr"].shape[0]})',
            f'Máscara HSV\n ({hsv_sb_cnt}px)',
            f'Máscara LAB/BGR\n({lab_cnt}px)',
            f'Overlay cerca K\n({d["px_count"]}px en radio 80)',
        ]

        for col, (img, subtitle) in enumerate(zip(panels, subtitles)):
            axes[row][col].imshow(img)
            if row == 0:
                axes[row][col].set_title(col_titles[col], fontsize=9, fontweight='bold')
            axes[row][col].set_ylabel(ch_name, fontsize=11, fontweight='bold',
                                       rotation=0, labelpad=30)
            axes[row][col].set_xlabel(subtitle.replace('\n',' '), fontsize=7.5)
            axes[row][col].axis('off')
            axes[row][col].plot(d['k_local_cx'], d['k_local_cy'], '+',
                                 color='gray', markersize=12, markeredgewidth=1.5)

    plt.tight_layout(rect=[0.04, 0.0, 1.0, 0.95])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def analyze_hue_range(crop_hsv, mask_near, ch_info):
    """
    Analiza el rango de Hue (H) observado en los píxeles detectados.
    Compara contra los rangos configurados para el canal.
    
    Retorna dict con:
      - h_min, h_max: rango observado
      - h_mean, h_median, h_mode: estadísticas
      - s_mean: saturación promedio
      - v_mean: brillo promedio
      - config_ranges: los rangos de HSV configurados
    """
    ys, xs = np.where(mask_near > 0)
    if len(xs) == 0:
        return {
            'h_min': None, 'h_max': None, 'h_mean': None,
            'h_median': None, 'h_mode': None, 'h_std': None,
            's_mean': None, 'v_mean': None,
            'pixel_count': 0,
            'config_ranges': ch_info.get('hsv_ranges', [])
        }
    
    h_values = crop_hsv[ys, xs, 0]
    s_values = crop_hsv[ys, xs, 1]
    v_values = crop_hsv[ys, xs, 2]
    
    h_min = int(np.min(h_values))
    h_max = int(np.max(h_values))
    h_mean = int(np.mean(h_values))
    h_median = int(np.median(h_values))
    h_std = int(np.std(h_values))
    
    # Calcular moda (valor H más frecuente)
    h_hist, _ = np.histogram(h_values, bins=180, range=(0, 180))
    h_mode = int(np.argmax(h_hist))
    
    s_mean = int(np.mean(s_values))
    v_mean = int(np.mean(v_values))
    
    return {
        'h_min': h_min,
        'h_max': h_max,
        'h_mean': h_mean,
        'h_median': h_median,
        'h_mode': h_mode,
        'h_std': h_std,
        's_mean': s_mean,
        'v_mean': v_mean,
        'pixel_count': len(xs),
        'config_ranges': ch_info.get('hsv_ranges', []),
        'h_values': h_values  # para graficar histograma
    }


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
    k_marks = non_max_suppression(k_detections, radius=110)
    
    if len(k_marks) == 0:
        return None, None, k_marks
    
    # PASO 2: Detectar CMY
    cmyk_marks = {'K': k_marks}
    diag_por_canal = {}
    
    for ch_name, ch_info in CMY_CROP_RANGES.items():
        marks_canal, diag_data = detectar_canal_con_imagen_separada(
            img_bgr, ch_name, ch_info, k_marks,
            template, roi_margin=roi_margin,
            search_radius=search_radius, threshold=0.2
        )
        
        # Filtrar por 1000 píxeles
        if len(diag_data) > 0 and diag_data[0].get('px_count', 0) >= 1000:
            cmyk_marks[ch_name] = marks_canal
            diag_por_canal[ch_name] = diag_data
        else:
            cmyk_marks[ch_name] = []
    
    return cmyk_marks, diag_por_canal, k_marks
