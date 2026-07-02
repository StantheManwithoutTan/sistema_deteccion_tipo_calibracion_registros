import cv2
import numpy as np
import matplotlib.pyplot as plt

def crear_overlay_canal(crop_bgr, mask_near, px_count,
                        k_local_cx, k_local_cy, search_radius,
                        draw_color_bgr, local_det_cx, local_det_cy,
                        detect_radius):
    """
    Crea imagen overlay coloreando píxeles detectados y dibujando
    radio de búsqueda, cruz de K y círculo de detección.
    """
    overlay = crop_bgr.copy()
    if px_count > 0:
        overlay[mask_near > 0] = draw_color_bgr
    cv2.circle(overlay, (k_local_cx, k_local_cy), search_radius, (200, 200, 200), 1)
    cv2.drawMarker(overlay, (k_local_cx, k_local_cy), (200, 200, 200),
                   cv2.MARKER_CROSS, 16, 1)
    cv2.circle(overlay, (local_det_cx, local_det_cy), detect_radius, draw_color_bgr, 2)
    cv2.circle(overlay, (local_det_cx, local_det_cy), 5, draw_color_bgr, -1)
    return overlay


def plot_diagnostico_canal(crop_bgr, crop_enhanced, img_isolated,
                           overlay_near, mask_full, mask_near, diag_masks,
                           ch_name, ch_info, k_local_cx, k_local_cy,
                           search_radius, px_count,
                           best_cx, best_cy, best_score,
                           rgb_color, hsv_color, hue_analysis,
                           crop_w, crop_h, mark_idx, show_plots):
    """
    Genera figura matplotlib 2×5 con diagnóstico completo del canal.
    No retorna nada — muestra o cierra el plot según show_plots.
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
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
        (cv2.cvtColor(overlay_near, cv2.COLOR_BGR2RGB),
         f'Overlay cerca K (r={search_radius}px, {px_count}px)'),
    ]

    # Fila 1: máscaras
    mask_full_3ch = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
    mask_near_3ch = cv2.cvtColor(mask_near, cv2.COLOR_GRAY2BGR)
    mask_hsv_sb = cv2.cvtColor(diag_masks['hsv_sin_boost'], cv2.COLOR_GRAY2BGR)
    mask_lbgr = cv2.cvtColor(diag_masks['lab_bgr'], cv2.COLOR_GRAY2BGR)

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
