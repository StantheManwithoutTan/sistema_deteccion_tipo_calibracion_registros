"""
main_single.py
--------------
Ejecuta el pipeline de detección CMYK sobre una sola imagen O un lote de imágenes,
mostrando los diagnósticos visuales y guardando 3 archivos JPG por imagen.

Uso (una sola imagen — con visualización):
    python main_single.py
    python main_single.py --imagen ruta/a/imagen.jpg

Uso (lote de imágenes — sin visualización):
    python main_single.py --batch --input_dir ./fotos --output_dir ./resultados_v3
"""

import os
import glob
import cv2
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math

from config import (
    CMY_CROP_RANGES, COLORS_LABEL, offsets_label,
    distancia_camara_plano_mm, focal_mm, sensor_width_mm,
    APPLY_SHARPENING, SHARPENING_STRENGTH, SIZE_ADAPTIVE_ENABLED, MIN_ROI_MARGIN, MIN_SEARCH_RADIUS, FACTOR_CORRECION_MM,
)
from image_utils import (
    sharpen_image, preprocess_image,
    create_crosshair_template, multi_scale_template_match, non_max_suppression,
)
from detection import (
    detectar_canal_con_imagen_separada,
    generar_diagnostico_mascaras,
)

COLOR_KEY_DIST = {'C': 'cyan', 'M': 'magenta', 'Y': 'yellow'}


# ================================================================
# FUNCIONES DE GUARDADO (extraídas de batch.py)
# ================================================================

def guardar_imagen_mascaras(img_bgr, cmyk_marks, diag_por_canal, k_marks, 
                             output_dir, name_no_ext, roi_margin=230):
    """
    Genera y guarda el panel de máscaras: 3 filas (C/M/Y) × 2 columnas (imagen aislada + máscara cerca K).
    """
    ch_list  = ['C', 'M', 'Y']
    cell_h, cell_w = 230, 230
    n_marks  = len(k_marks)
    mask_panel = np.zeros((3 * cell_h, n_marks * 2 * cell_w + n_marks * 6, 3), dtype=np.uint8)

    for row, ch_name in enumerate(ch_list):
        draw_color_bgr = CMY_CROP_RANGES[ch_name].get('color_display', (200,200,0))
        
        # Obtener datos de diagnóstico para este canal
        diag_data_list = diag_por_canal.get(ch_name, [])
        if not diag_data_list:
            continue

        for mi in range(n_marks):
            if mi >= len(diag_data_list):
                continue
            
            d = diag_data_list[mi]
            img_iso = d['img_isolated']
            m_near = d['mask_near']
            klx = d['k_local_cx']
            kly = d['k_local_cy']
            kscale = k_marks[mi][3]

            # Dibujar posición detectada en imagen aislada
            iso_draw = img_iso.copy()
            if cmyk_marks.get(ch_name) and len(cmyk_marks[ch_name]) > mi:
                cx_det, cy_det = cmyk_marks[ch_name][mi][0], cmyk_marks[ch_name][mi][1]
                lx_det, ly_det = int(cx_det) - (int(k_marks[mi][0]) - klx), \
                                 int(cy_det) - (int(k_marks[mi][1]) - kly)
                cv2.circle(iso_draw, (lx_det, ly_det), int(40 * kscale), draw_color_bgr, 2)
                cv2.circle(iso_draw, (lx_det, ly_det), 4, draw_color_bgr, -1)
            
            cv2.drawMarker(iso_draw, (klx, kly), (180,180,180), cv2.MARKER_CROSS, 14, 1)

            # Máscara coloreada cerca de K
            near_colored = np.zeros_like(img_iso)
            near_colored[m_near > 0] = draw_color_bgr
            cv2.circle(near_colored, (klx, kly), 80, (80,80,80), 1)
            if cmyk_marks.get(ch_name) and len(cmyk_marks[ch_name]) > mi:
                cx_det, cy_det = cmyk_marks[ch_name][mi][0], cmyk_marks[ch_name][mi][1]
                lx_det, ly_det = int(cx_det) - (int(k_marks[mi][0]) - klx), \
                                 int(cy_det) - (int(k_marks[mi][1]) - kly)
                cv2.circle(near_colored, (lx_det, ly_det), int(40 * kscale), draw_color_bgr, 2)

            # Colocar ambas imágenes en el panel
            for col_panel, src in enumerate([iso_draw, near_colored]):
                resized = cv2.resize(src, (cell_w, cell_h))
                col_off = (mi * 2 + col_panel) * cell_w + mi * 6
                row_off = row * cell_h
                mask_panel[row_off:row_off+cell_h, col_off:col_off+cell_w] = resized

            cv2.putText(mask_panel,
                        f"{ch_name} ({CMY_CROP_RANGES[ch_name]['nombre']})",
                        (4, row * cell_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        draw_color_bgr, 1, cv2.LINE_AA)

    # Etiquetas de columnas
    for mi in range(n_marks):
        for ci, lbl in enumerate(['Imagen aislada', 'Mascara cerca K']):
            col_off = (mi * 2 + ci) * cell_w + mi * 6
            cv2.putText(mask_panel, f'K-{mi} {lbl}', (col_off + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

    masks_path = os.path.join(output_dir, f'{name_no_ext}_mascaras.jpg')
    cv2.imwrite(masks_path, mask_panel)
    print(f'  ✓ Máscaras guardadas: {os.path.basename(masks_path)}')


def guardar_imagen_resultado(img_bgr, cmyk_marks, k_marks, output_dir, 
                              name_no_ext, filename, roi_margin=230, gt_marks=None):
    """
    Genera y guarda la imagen de resultado: ROI original | ROI anotado con posiciones CMYK.
    """
    mcx, mcy, _, mscale = k_marks[0]
    rx1 = max(int(mcx) - roi_margin, 0)
    ry1 = max(int(mcy) - roi_margin, 0)
    rx2 = min(int(mcx) + roi_margin, img_bgr.shape[1])
    ry2 = min(int(mcy) + roi_margin, img_bgr.shape[0])

    roi_original = img_bgr[ry1:ry2, rx1:rx2].copy()
    cv2.putText(roi_original, 'Original', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    roi_final = img_bgr[ry1:ry2, rx1:rx2].copy()

    # Dibujar marcas detectadas
    for ch_name in ['C', 'M', 'Y', 'K']:
        color_bgr  = COLORS_LABEL[ch_name]
        marks_list = cmyk_marks.get(ch_name, [])[:1]
        for (cx, cy, score, scale) in marks_list:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                lx, ly = int(cx) - rx1, int(cy) - ry1
                r = int(40 * scale)
                cv2.circle(roi_final, (lx, ly), r, color_bgr, 2)
                cv2.circle(roi_final, (lx, ly), 5, color_bgr, -1)
                cv2.drawMarker(roi_final, (lx, ly), color_bgr, cv2.MARKER_CROSS, 18, 2)
                
                # Línea desde CMY a K
                if ch_name != 'K' and len(k_marks) > 0:
                    k_lx = int(k_marks[0][0]) - rx1
                    k_ly = int(k_marks[0][1]) - ry1
                    cv2.line(roi_final, (lx, ly), (k_lx, k_ly), color_bgr, 1, cv2.LINE_AA)                
                # Etiqueta
                ox, oy = offsets_label.get(ch_name, (10, -10))
                llx, lly = lx + ox, ly + oy
                label = f'{ch_name}  s={score:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(roi_final, (llx - 2, lly - th - 4),
                              (llx + tw + 2, lly + 4), (0, 0, 0), -1)
                cv2.putText(roi_final, label, (llx, lly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA)
            if gt_marks and ch_name in gt_marks:
                gx, gy = gt_marks[ch_name]
                if rx1 <= gx <= rx2 and ry1 <= gy <= ry2:
                    lgx, lgy = int(gx) - rx1, int(gy) - ry1
                    gr = int(40 * k_marks[0][3]) if k_marks else 20
                    cv2.circle(roi_final, (lgx, lgy), gr, (255, 255, 255), 2, cv2.LINE_4)  # punteado
                    cv2.putText(roi_final, f'{ch_name} GT', (lgx + 12, lgy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    n_detected = sum(1 for v in cmyk_marks.values() if len(v) > 0)
    cv2.putText(roi_final, f'Resultado final — {n_detected}/4 canales detectados',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    sep = np.full((roi_original.shape[0], 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([roi_original, sep, roi_final])
    res_path = os.path.join(output_dir, f'{name_no_ext}_resultado.jpg')
    cv2.imwrite(res_path, combined)
    print(f'  ✓ Resultado guardado: {os.path.basename(res_path)}')


def guardar_imagen_calculos(img_bgr, cmyk_marks, k_marks, output_dir,
                             name_no_ext, filename, roi_margin=230,
                             distancia_camara_plano_mm_custom=None, gt_posiciones=None):
    """
    Genera y guarda el panel de cálculos: desalineamientos respecto a K y distancias entre pares.
    """
    image_width_px  = img_bgr.shape[1]
    tamano_pixel_mm = sensor_width_mm / image_width_px

    dist_cam_plano = distancia_camara_plano_mm_custom if distancia_camara_plano_mm_custom is not None else distancia_camara_plano_mm
    mm_por_px = ((sensor_width_mm * dist_cam_plano) / (focal_mm * image_width_px)) * FACTOR_CORRECION_MM

    positions_mm = {}
    for ch in ['C', 'M', 'Y', 'K']:
        if cmyk_marks.get(ch) and len(cmyk_marks[ch]) > 0:
            positions_mm[ch] = (cmyk_marks[ch][0][0], cmyk_marks[ch][0][1])

    info_lines = [
        f'Archivo: {filename}',
        f'Factor optico: 1 px = {mm_por_px:.4f} mm  |  Dist. camara-plano: {dist_cam_plano} mm  |  Focal: {focal_mm} mm',
        '',
        '--- Desalineamiento respecto a K ---',
    ]

    # ── Desalineamiento de cada color respecto a K (no depende de GT) ──
    if len(k_marks) > 0:
        kx, ky = k_marks[0][0], k_marks[0][1]
        for ch in ['C', 'M', 'Y']:
            if positions_mm.get(ch):
                dx_px = positions_mm[ch][0] - kx
                dy_px = positions_mm[ch][1] - ky
                dist_px = np.hypot(dx_px, dy_px)
                #if dist_px <= 0:
                    #continue
                dx_mm   = ((dx_px * tamano_pixel_mm * dist_cam_plano) / focal_mm) * FACTOR_CORRECION_MM
                dy_mm   = ((dy_px * tamano_pixel_mm * dist_cam_plano) / focal_mm) * FACTOR_CORRECION_MM
                dist_mm = ((dist_px * tamano_pixel_mm * dist_cam_plano) / focal_mm) * FACTOR_CORRECION_MM
                info_lines.append(f'  {ch}-K:  Δx={dx_mm:+.3f} mm,  Δy={dy_mm:+.3f} mm,  dist={dist_mm:.3f} mm  ({dist_px:.1f} px)')
            else:
                info_lines.append(f'  {ch}: no disponible')

    # --- Comparación predicha vs real (GT) — solo si hay GT ──
    if gt_posiciones:
        info_lines += ['', '--- Comparación predicha vs real (GT) ---']
        for ch in ['C', 'M', 'Y']:
            if ch not in gt_posiciones:
                continue
            gx, gy = gt_posiciones[ch]
            if ch in positions_mm:
                pxc, pyc = positions_mm[ch]
                pred_err_px = np.hypot(pxc - gx, pyc - gy)
                pred_err_mm = pred_err_px * mm_por_px
                info_lines.append(f'  |{ch} pred - {ch} GT|: {pred_err_mm:.3f} mm ({pred_err_px:.1f} px)')
            else:
                info_lines.append(f'  |{ch} pred - {ch} GT|: no detectado')
            if 'K' in gt_posiciones:
                kx, ky = gt_posiciones['K']
                k_gt_px = np.hypot(gx - kx, gy - ky)
                k_gt_mm = k_gt_px * mm_por_px
                info_lines.append(f'  |NegGT - {ch} GT|: {k_gt_mm:.3f} mm ({k_gt_px:.1f} px)')

    # --- Distancias entre todos los pares ──
    info_lines += ['', '--- Distancias entre todos los pares ---']
    pnames = list(positions_mm.keys())
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            n1, n2 = pnames[i], pnames[j]
            dx = positions_mm[n1][0] - positions_mm[n2][0]
            dy = positions_mm[n1][1] - positions_mm[n2][1]
            dist_px = np.hypot(dx, dy)
            dist_mm = ((dist_px * tamano_pixel_mm * dist_cam_plano) / focal_mm) * FACTOR_CORRECION_MM
            info_lines.append(f'  {n1}-{n2}:  {dist_mm:.3f} mm  ({dist_px:.1f} px)')

    font_c  = cv2.FONT_HERSHEY_SIMPLEX
    lh, pad = 28, 14
    pw, ph  = 780, pad * 2 + lh * (len(info_lines) + 1)
    calc_panel = np.full((ph, pw, 3), 20, dtype=np.uint8)
    cv2.rectangle(calc_panel, (0, 0), (pw - 1, ph - 1), (60, 60, 60), 2)

    for idx, line in enumerate(info_lines):
        y = pad + (idx + 1) * lh
        color = (200, 200, 200)
        for ch, bgr in [('C', (255,255,0)), ('M', (255,0,255)), ('Y', (0,255,255)), ('K', (180,180,180))]:
            if line.strip().startswith(ch + '-') or line.strip().startswith(ch + ':'):
                color = bgr
                break
        cv2.putText(calc_panel, line, (pad, y), font_c, 0.52, color, 1, cv2.LINE_AA)

    mm_path = os.path.join(output_dir, f'{name_no_ext}_calculos_mm.jpg')
    cv2.imwrite(mm_path, calc_panel)
    print(f'  ✓ Cálculos mm guardados: {os.path.basename(mm_path)}')


# ================================================================
# FUNCIONES DE PROCESAMIENTO
# ================================================================

def procesar_y_guardar_imagen(img_path, template, output_dir, show_diagnostics=True,
                               calibration_method='distance',      
                               reference_size_mm=None,
                               distancia_camara_plano_mm_custom=None,
                               canales_a_procesar=['C', 'M', 'Y'],gt_marks=None,               
                               gt_posiciones=None):  
    """
    Procesa una imagen con método de calibración configurable.
    
    calibration_method:
      - 'distance': usa distancia cámara-plano
      - 'reference_size': usa tamaño de K conocido
    
    reference_size_mm: tamaño real de K en mm (ej: 10 para 1 cm)
    distancia_camara_plano_mm_custom: distancia cámara-plano en mm (si es None, usa config)

    canales_a_procesar: lista de canales a procesar (ej: ['C', 'M'] para solo Cyan y Magenta)
    """
    filename    = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]

    print(f"\n{'='*60}\nProcesando: {filename}\n{'='*60}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'  ⚠ No se pudo cargar {filename}')
        return

    # Aplicar sharpening si está activado
    if APPLY_SHARPENING:
        img_bgr = sharpen_image(img_bgr, strength=SHARPENING_STRENGTH)
        print(f'  ✓ Sharpening aplicado (strength={SHARPENING_STRENGTH})')

    image_width_px  = img_bgr.shape[1]
    tamano_pixel_mm = sensor_width_mm / image_width_px
    
    # Usar distancia personalizada o la de config
    dist_cam_plano = distancia_camara_plano_mm_custom if distancia_camara_plano_mm_custom is not None else distancia_camara_plano_mm
    
    # Primero detectar K para poder usarlo como referencia
    lab_prep = preprocess_image(img_bgr)
    L_full, _, _ = cv2.split(lab_prep)

    k_detections = multi_scale_template_match(
        L_full, template,
        scales=np.arange(0.15, 3.2, 0.1),
        threshold=0.35
    )


    # dyn_nms_radius ANTES de NMS
    if SIZE_ADAPTIVE_ENABLED and k_detections:
        dyn_nms_radius = int(110 * np.sqrt(k_detections[0][3]))
    else:
        dyn_nms_radius = 110
    k_marks = non_max_suppression(k_detections, radius=dyn_nms_radius)


    print(f'\nMarcas K encontradas: {len(k_marks)}')
    for i, (cx, cy, score, scale) in enumerate(k_marks):
        print(f'  K-{i}: ({int(cx)},{int(cy)})  score={score:.3f}  scale={scale:.2f}')

    if len(k_marks) == 0:
        print('⚠ No se detectaron marcas K.')
        return None

    # ✓ CALCULAR mm_por_px según el método elegido
    if calibration_method == 'distance':
        mm_por_px = ((tamano_pixel_mm * dist_cam_plano) / focal_mm) * FACTOR_CORRECION_MM
        calib_info = f'Dist. camara-plano: {dist_cam_plano} mm | Focal: {focal_mm} mm'
    
    elif calibration_method == 'reference_size':
        if reference_size_mm is None:
            print('  ⚠ ERROR: reference_size_mm es requerido para mode reference_size')
            return None
        
        # Calcular tamaño de K en píxeles (escala = tamaño detectado)
        k_scale_px = k_marks[0][3]  # El scale es aproximadamente el tamaño en píxeles
        # El template es de 101x101, así que:
        k_size_px = 101 * k_scale_px
        
        mm_por_px = reference_size_mm / k_size_px
        calib_info = f'Ref. tamaño K: {reference_size_mm} mm | Detectado: {k_size_px:.1f} px'
    
    else:
        print(f'  ⚠ Método de calibración inválido: {calibration_method}')
        return None

    print(f'Calibración: {calib_info}')
    print(f'Factor óptico: 1 px = {mm_por_px:.4f} mm')
    
    # =========================================================
    # PASO 1: Detectar marcas K
    # =========================================================
    print('=' * 60)
    print('  PASO 1: Detectar marcas K (negro) con template matching')
    print('=' * 60)

    print(f'\nMarcas K encontradas: {len(k_marks)}')
    for i, (cx, cy, score, scale) in enumerate(k_marks):
        print(f'  K-{i}: ({int(cx)},{int(cy)})  score={score:.3f}  scale={scale:.2f}')

    if len(k_marks) == 0:
        print('⚠ No se detectaron marcas K.')
        return

    if SIZE_ADAPTIVE_ENABLED:
        avg_kscale = np.mean([km[3] for km in k_marks])
        f = np.sqrt(avg_kscale)
        dyn_roi_margin    = max(MIN_ROI_MARGIN,    int(230 * f))
        dyn_search_radius = max(MIN_SEARCH_RADIUS, int(110 * f))
    else:
        dyn_roi_margin    = 230
        dyn_search_radius = 110

    # =========================================================
    # PASO 2: Detectar C, M, Y
    # =========================================================
    print('=' * 60)
    print('  PASO 2: Detectar CMY con imágenes separadas por canal')
    print('=' * 60)

    cmyk_marks    = {'K': k_marks}
    diag_por_canal = {}

    for ch_name, ch_info in CMY_CROP_RANGES.items():
        # ✓ NUEVA CONDICIÓN: saltar si el canal no está en la lista
        if ch_name not in canales_a_procesar:
            print(f"  ⊘ Canal {ch_name} omitido (no incluido en análisis)")
            cmyk_marks[ch_name] = []
            continue
            
        marks_canal, diag_data = detectar_canal_con_imagen_separada(
            img_bgr, ch_name, ch_info, k_marks,
            template, dyn_roi_margin,
            dyn_search_radius, threshold=0.2,
            show_plots=show_diagnostics 
        )
        
        # Filtrar: solo incluir si >= 800 píxeles
        if len(diag_data) > 0 and diag_data[0].get('px_count', 0) >= 400:
            cmyk_marks[ch_name] = marks_canal
            diag_por_canal[ch_name] = diag_data
        else:
            px_count = diag_data[0].get('px_count', 0) if len(diag_data) > 0 else 0
            print(f"  ⚠ Canal {ch_name} descartado: {px_count} < 800 píxeles")
            cmyk_marks[ch_name] = []

    # =========================================================
    # PASO 3: Guardar 3 archivos JPG
    # =========================================================
    os.makedirs(output_dir, exist_ok=True)

    guardar_imagen_mascaras(img_bgr, cmyk_marks, diag_por_canal, k_marks, 
                           output_dir, name_no_ext)
    guardar_imagen_resultado(img_bgr, cmyk_marks, k_marks, output_dir, name_no_ext, filename,
                             roi_margin=230, gt_marks=gt_marks)          # ← gt_marks
    guardar_imagen_calculos(img_bgr, cmyk_marks, k_marks, output_dir, name_no_ext, filename,
                            distancia_camara_plano_mm_custom=dist_cam_plano,
                            gt_posiciones=gt_posiciones)    # ✓ PASAR DISTANCIA

    n_detected = sum(1 for v in cmyk_marks.values() if len(v) > 0)
    print(f'  ✓ Canales detectados: {n_detected}/4')

    # =========================================================
    # PASO 4: Mostrar diagnósticos (solo si es single con visualización)
    # =========================================================
    # show_diagnostics = True
    if show_diagnostics and len(diag_por_canal) > 0:
        generar_diagnostico_mascaras(diag_por_canal, preprocesada_titulo='PREPROCESADA', show_plot=show_diagnostics)
        
        # Resumen de detecciones
        print('\n' + '=' * 60)
        print('RESUMEN DE DETECCIONES')
        print('=' * 60)
        for ch_name in ['C', 'M', 'Y', 'K']:
            marks = cmyk_marks.get(ch_name, [])
            if marks:
                for m in marks:
                    print(f'  {ch_name}: ({int(m[0])},{int(m[1])})  score={m[2]:.3f}  scale={m[3]:.2f}')
            else:
                print(f'  {ch_name}: no detectado')
        
        # Mostrar imagen final con anotaciones
        mcx, mcy, mscore, mscale = k_marks[0]
        roi_margin = 230
        rx1 = max(int(mcx) - roi_margin, 0)
        ry1 = max(int(mcy) - roi_margin, 0)
        rx2 = min(int(mcx) + roi_margin, img_bgr.shape[1])
        ry2 = min(int(mcy) + roi_margin, img_bgr.shape[0])

        roi_original = img_bgr[ry1:ry2, rx1:rx2].copy()
        cv2.putText(roi_original, 'Original', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        roi_final = img_bgr[ry1:ry2, rx1:rx2].copy()

        for ch_name in ['C', 'M', 'Y', 'K']:
            color_bgr  = COLORS_LABEL[ch_name]
            marks_list = cmyk_marks.get(ch_name, [])[:1]
            for (cx, cy, score, scale) in marks_list:
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    lx, ly = int(cx) - rx1, int(cy) - ry1
                    r = int(40 * scale)
                    cv2.circle(roi_final, (lx, ly), r, color_bgr, 2)
                    cv2.circle(roi_final, (lx, ly), 5, color_bgr, -1)
                    cv2.drawMarker(roi_final, (lx, ly), color_bgr, cv2.MARKER_CROSS, 18, 2)
                    if ch_name != 'K' and len(k_marks) > 0:
                        k_lx = int(k_marks[0][0]) - rx1
                        k_ly = int(k_marks[0][1]) - ry1
                        cv2.line(roi_final, (lx, ly), (k_lx, k_ly), color_bgr, 1, cv2.LINE_AA)
                    ox, oy = offsets_label.get(ch_name, (10, -10))
                    llx, lly = lx + ox, ly + oy
                    label = f'{ch_name}  s={score:.2f}'
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(roi_final, (llx - 2, lly - th - 4),
                                  (llx + tw + 2, lly + 4), (0, 0, 0), -1)
                    cv2.putText(roi_final, label, (llx, lly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA)

        cv2.putText(roi_final, f'Resultado final — {n_detected}/4 canales detectados',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(roi_final, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Resultado final — {n_detected}/4 canales', fontsize=12)
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
    return cmyk_marks


def procesar_lote(input_dir='.', output_dir='resultados_v3',
                  calibration_method='distance',
                  reference_size_mm=None,
                  distancia_camara_plano_mm_custom=None,
                  filenames=None,        # NUEVO: si se pasa, procesa solo estas
                  canales_a_procesar=['C', 'M', 'Y']):
    template = create_crosshair_template(
        size=101, ring_radius=40, ring_thickness=8,
        cross_thickness=10, cross_length=90
    )

    if filenames is not None:
        image_paths = [os.path.join(input_dir, f) for f in filenames]
    else:
        image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))  # ← cambia a png
        # o mantener .jpg + .png según dataset

    print(f'Imágenes a procesar: {len(image_paths)}')
    for img_path in image_paths:
        procesar_y_guardar_imagen(img_path, template, output_dir,
                                 show_diagnostics=False,        # ya headless
                                 calibration_method=calibration_method,
                                 reference_size_mm=reference_size_mm,
                                 distancia_camara_plano_mm_custom=distancia_camara_plano_mm_custom,
                                 canales_a_procesar=canales_a_procesar)
    plt.close('all')
    print(f'\n✓ Procesado lote completo. Resultados en: {output_dir}')


# Entra el CSV donde queda los resultados del analisis elegido y toma los 5 peores casos de cada color para ejecutar
def seleccionar_peores_por_color(csv_path, top_n=5):
    """
    Lee el CSV del dataset sintético y selecciona las top_n imágenes por canal
    (C/M/Y) con mayor '*_dist' (distancia en px de la marca al registro K).
    Devuelve lista de tuplas (filename, ch, dist_px) ordenada por color y desvío desc.
    """
    df = pd.read_csv(csv_path)
    df['black_cx'] = pd.to_numeric(df['black_cx'], errors='coerce')
    df = df.dropna(subset=['black_cx']).reset_index(drop=True)  # solo filas completas

    dist_cols = {'C': 'cyan', 'M': 'magenta', 'Y': 'yellow'}

    casos = []
    for ch, color in dist_cols.items():
        dist_col = f'{color}_dist'                              # cyan_dist / magenta_dist / yellow_dist
        sub = df[df[dist_col].notna()].nlargest(top_n, dist_col)
        for _, r in sub.iterrows():
            casos.append((r['filename'], ch,
              float(r[f'{color}_dist']),
              float(r[f'{color}_cx']), float(r[f'{color}_cy']),
              float(r['black_cx']), float(r['black_cy'])))
    return casos

# procesar los colores elegidos
def peores_por_color(csv_path, input_dir, output_dir, top_n=5,
                     calibration_method='distance',
                     distancia_camara_plano_mm_custom=None):
    template = create_crosshair_template(
        size=101, ring_radius=40, ring_thickness=8,
        cross_thickness=10, cross_length=90
    )
    casos = seleccionar_peores_por_color(csv_path, top_n)   # lista (filename, ch, dist_px)
    print(f'Casos a procesar (top {top_n} por color): {len(casos)}')

    for filename, ch, dist_gt, gtx, gty, k_gtx, k_gty in casos:
        img_path = os.path.join(input_dir, filename)
        print(f"\n[{ch}] {filename}  | dist GT = {dist_gt:.1f} px")
        if not os.path.exists(img_path):
            print('  ⚠ Imagen no encontrada')
            continue

        gt_marks      = {ch: (gtx, gty)}
        gt_posiciones = {ch: (gtx, gty), 'K': (k_gtx, k_gty)}

        cmyk = procesar_y_guardar_imagen(
            img_path, template, output_dir,
            show_diagnostics=False,
            calibration_method=calibration_method,
            distancia_camara_plano_mm_custom=distancia_camara_plano_mm_custom,
            canales_a_procesar=[ch],
            gt_marks=gt_marks,
            gt_posiciones=gt_posiciones,
        )

        pts = cmyk.get(ch) if cmyk else []         # lista de (cx, cy, score, scale)
        print(f'  Posición GT      : ({gtx:.1f}, {gty:.1f})')
        if pts:
            pxc, pyc = pts[0][0], pts[0][1]
            print(f'  Posición predicha: ({pxc:.1f}, {pyc:.1f})')
            print(f'  |Δ| = {math.hypot(pxc - gtx, pyc - gty):.1f} px')
        else:
            print('  Posición predicha: no detectada')

    plt.close('all')
    print(f'\n✓ Modo peores casos terminado. Resultados en: {output_dir}')

def main_single(imagen_path='20250925_142228.jpg',
                calibration_method='distance',
                reference_size_mm=None,
                distancia_camara_plano_mm_custom=None,
                canales_a_procesar=['C', 'M', 'Y']):  # ✓ NUEVO
    """
    Procesa una sola imagen con visualización de diagnósticos.
    """
    template = create_crosshair_template(
        size=101, ring_radius=40, ring_thickness=8,
        cross_thickness=10, cross_length=90
    )
    
    # Guarda en directorio actual si no existe
    output_dir = 'resultados_v5'
    os.makedirs(output_dir, exist_ok=True)
    
    procesar_y_guardar_imagen(imagen_path, template, output_dir, 
                             show_diagnostics=True,
                             calibration_method=calibration_method,
                             reference_size_mm=reference_size_mm,
                             distancia_camara_plano_mm_custom=distancia_camara_plano_mm_custom,
                             canales_a_procesar=canales_a_procesar)  # ✓ NUEVO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detección de marcas CMYK — imagen única o lote'
    )
    parser.add_argument('--imagen', default='20250925_142228.jpg',
                        help='Ruta a la imagen (modo single)')
    parser.add_argument('--batch', action='store_true',
                        help='Activar modo batch (procesar múltiples imágenes)')
    parser.add_argument('--input_dir', default='.',
                        help='Directorio con imágenes (modo batch)')
    parser.add_argument('--output_dir', default='resultados_v3',
                        help='Directorio de salida')
    
    # ✓ NUEVOS ARGUMENTOS PARA CALIBRACIÓN
    parser.add_argument('--calib_method', choices=['distance', 'reference_size'],
                        default='distance',
                        help='Método de calibración: distance (cámara-plano) o reference_size (tamaño de referencia)')
    parser.add_argument('--ref_size_mm', type=float, default=None,
                        help='Tamaño real de K en mm (ej: 10 para 1 cm)')
    parser.add_argument('--distancia_mm', type=float, default=None,
                        help='Distancia cámara-plano en mm (solo para --calib_method distance)')
    
    # ✓ NUEVO: argumento para seleccionar canales
    parser.add_argument('--canales', default='C,M,Y',
                        help='Canales a procesar (ej: C,M,Y / C,Y / M, etc.)')

    # Agrega opcion de analizar bote de resultados del analsis que salieron mal
    parser.add_argument('--peores_por_color', action='store_true',
                    help='Procesar y mostrar los top_n casos más imprecisos por color (CSV + carpeta)')
    parser.add_argument('--csv', default=r'C:\Users\cadet\Documents\tesis_reconocimiento_cara\CREACION_DATASET_SINTETICO_TESIS\registro_dataset_v4_2_a_10_4k.csv',
                        help='Ruta al CSV de ground truth (modo peores_por_color)')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Cantidad de imágenes por color (modo peores_por_color)')
    
    args = parser.parse_args()
    
    # Procesar string de canales
    canales_a_procesar = [ch.strip().upper() for ch in args.canales.split(',')]

    if args.peores_por_color:
        peores_por_color(args.csv, args.input_dir, args.output_dir,
                        top_n=args.top_n,
                        calibration_method=args.calib_method,
                        distancia_camara_plano_mm_custom=args.distancia_mm)
        sys.exit(0)
    
    if args.batch:
        # Modo batch: procesa todas las imágenes del directorio
        procesar_lote(args.input_dir, args.output_dir,
                      calibration_method=args.calib_method,
                      reference_size_mm=args.ref_size_mm,
                      distancia_camara_plano_mm_custom=args.distancia_mm,
                      canales_a_procesar=canales_a_procesar)  # ✓ NUEVO
    else:
        # Modo single: procesa una sola imagen con visualización
        main_single(args.imagen,
                    calibration_method=args.calib_method,
                    reference_size_mm=args.ref_size_mm,
                    distancia_camara_plano_mm_custom=args.distancia_mm,
                    canales_a_procesar=canales_a_procesar)  # ✓ NUEVO