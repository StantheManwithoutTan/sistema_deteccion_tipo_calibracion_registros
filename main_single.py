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
import matplotlib.pyplot as plt

from config import (
    CMY_CROP_RANGES, COLORS_LABEL, offsets_label,
    distancia_camara_plano_mm, focal_mm, sensor_width_mm,
    APPLY_SHARPENING, SHARPENING_STRENGTH,
)
from image_utils import (
    sharpen_image, preprocess_image,
    create_crosshair_template, multi_scale_template_match, non_max_suppression,
)
from detection import (
    detectar_canal_con_imagen_separada,
    get_representative_color,
    generar_diagnostico_mascaras,
)


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
                              name_no_ext, filename, roi_margin=230):
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

    n_detected = sum(1 for v in cmyk_marks.values() if len(v) > 0)
    cv2.putText(roi_final, f'Resultado final — {n_detected}/4 canales detectados',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    sep = np.full((roi_original.shape[0], 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([roi_original, sep, roi_final])
    res_path = os.path.join(output_dir, f'{name_no_ext}_resultado.jpg')
    cv2.imwrite(res_path, combined)
    print(f'  ✓ Resultado guardado: {os.path.basename(res_path)}')


def guardar_imagen_calculos(img_bgr, cmyk_marks, k_marks, output_dir, 
                             name_no_ext, filename, roi_margin=230):
    """
    Genera y guarda el panel de cálculos: desalineamientos respecto a K y distancias entre pares.
    """
    image_width_px  = img_bgr.shape[1]
    tamano_pixel_mm = sensor_width_mm / image_width_px
    mm_por_px = (sensor_width_mm * distancia_camara_plano_mm) / (focal_mm * image_width_px)

    positions_mm = {}
    for ch in ['C', 'M', 'Y', 'K']:
        if cmyk_marks.get(ch) and len(cmyk_marks[ch]) > 0:
            positions_mm[ch] = (cmyk_marks[ch][0][0], cmyk_marks[ch][0][1])

    info_lines = [
        f'Archivo: {filename}',
        f'Factor optico: 1 px = {mm_por_px:.4f} mm  |  Dist. camara-plano: {distancia_camara_plano_mm} mm  |  Focal: {focal_mm} mm',
        '',
        '--- Desalineamiento respecto a K ---',
    ]
    
    if len(k_marks) > 0:
        kx, ky = k_marks[0][0], k_marks[0][1]
        for ch in ['C', 'M', 'Y']:
            if positions_mm.get(ch):
                dx_px = positions_mm[ch][0] - kx
                dy_px = positions_mm[ch][1] - ky
                dist_px = np.hypot(dx_px, dy_px)
                dx_mm   = (dx_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                dy_mm   = (dy_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                dist_mm = (dist_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                info_lines.append(f'  {ch}-K:  Δx={dx_mm:+.3f} mm,  Δy={dy_mm:+.3f} mm,  dist={dist_mm:.3f} mm  ({dist_px:.1f} px)')
            else:
                info_lines.append(f'  {ch}: no disponible')

    info_lines += ['', '--- Distancias entre todos los pares ---']
    pnames = list(positions_mm.keys())
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            n1, n2 = pnames[i], pnames[j]
            dx = positions_mm[n1][0] - positions_mm[n2][0]
            dy = positions_mm[n1][1] - positions_mm[n2][1]
            dist_px = np.hypot(dx, dy)
            dist_mm = (dist_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
            info_lines.append(f'  {n1}-{n2}:  {dist_mm:.3f} mm  ({dist_px:.1f} px)')

    font_c  = cv2.FONT_HERSHEY_SIMPLEX
    lh, pad = 28, 14
    pw, ph  = 780, pad * 2 + lh * (len(info_lines) + 1)
    calc_panel = np.full((ph, pw, 3), 20, dtype=np.uint8)
    cv2.rectangle(calc_panel, (0, 0), (pw - 1, ph - 1), (60, 60, 60), 2)
    
    for idx, line in enumerate(info_lines):
        y = pad + (idx + 1) * lh
        color = (200, 200, 200)
        
        # Colorear líneas según canal
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
                               reference_size_mm=None):             # ✓ Solo esto
    """
    Procesa una imagen con método de calibración configurable.
    
    calibration_method:
      - 'distance': usa distancia cámara-plano
      - 'reference_size': usa tamaño de K conocido
    
    reference_size_mm: tamaño real de K en mm (ej: 10 para 1 cm)
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
    
    # Primero detectar K para poder usarlo como referencia
    lab_prep = preprocess_image(img_bgr)
    L_full, _, _ = cv2.split(lab_prep)

    k_detections = multi_scale_template_match(
        L_full, template,
        scales=np.arange(0.15, 3.2, 0.1),
        threshold=0.35
    )
    k_marks = non_max_suppression(k_detections, radius=110)

    print(f'\nMarcas K encontradas: {len(k_marks)}')
    for i, (cx, cy, score, scale) in enumerate(k_marks):
        print(f'  K-{i}: ({int(cx)},{int(cy)})  score={score:.3f}  scale={scale:.2f}')

    if len(k_marks) == 0:
        print('⚠ No se detectaron marcas K.')
        return

    # ✓ CALCULAR mm_por_px según el método elegido
    if calibration_method == 'distance':
        mm_por_px = (tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
        calib_info = f'Dist. camara-plano: {distancia_camara_plano_mm} mm | Focal: {focal_mm} mm'
    
    elif calibration_method == 'reference_size':
        if reference_size_mm is None:
            print('  ⚠ ERROR: reference_size_mm es requerido para mode reference_size')
            return
        
        # Calcular tamaño de K en píxeles (escala = tamaño detectado)
        k_scale_px = k_marks[0][3]  # El scale es aproximadamente el tamaño en píxeles
        # El template es de 101x101, así que:
        k_size_px = 101 * k_scale_px
        
        mm_por_px = reference_size_mm / k_size_px
        calib_info = f'Ref. tamaño K: {reference_size_mm} mm | Detectado: {k_size_px:.1f} px'
    
    else:
        print(f'  ⚠ Método de calibración inválido: {calibration_method}')
        return

    print(f'Calibración: {calib_info}')
    print(f'Factor óptico: 1 px = {mm_por_px:.4f} mm')
    
    # =========================================================
    # PASO 1: Detectar marcas K
    # =========================================================
    print('=' * 60)
    print('  PASO 1: Detectar marcas K (negro) con template matching')
    print('=' * 60)

    lab_prep = preprocess_image(img_bgr)
    L_full, _, _ = cv2.split(lab_prep)

    k_detections = multi_scale_template_match(
        L_full, template,
        scales=np.arange(0.15, 3.2, 0.1),
        threshold=0.35
    )
    k_marks = non_max_suppression(k_detections, radius=110)

    print(f'\nMarcas K encontradas: {len(k_marks)}')
    for i, (cx, cy, score, scale) in enumerate(k_marks):
        print(f'  K-{i}: ({int(cx)},{int(cy)})  score={score:.3f}  scale={scale:.2f}')

    if len(k_marks) == 0:
        print('⚠ No se detectaron marcas K.')
        return

    # =========================================================
    # PASO 2: Detectar C, M, Y
    # =========================================================
    print('=' * 60)
    print('  PASO 2: Detectar CMY con imágenes separadas por canal')
    print('=' * 60)

    cmyk_marks    = {'K': k_marks}
    diag_por_canal = {}

    for ch_name, ch_info in CMY_CROP_RANGES.items():
        marks_canal, diag_data = detectar_canal_con_imagen_separada(
            img_bgr, ch_name, ch_info, k_marks,
            template, roi_margin=230,
            search_radius=110, threshold=0.2,
            show_plots=show_diagnostics 
        )
        
        # Filtrar: solo incluir si >= 1000 píxeles
        if len(diag_data) > 0 and diag_data[0].get('px_count', 0) >= 1000:
            cmyk_marks[ch_name] = marks_canal
            diag_por_canal[ch_name] = diag_data
        else:
            px_count = diag_data[0].get('px_count', 0) if len(diag_data) > 0 else 0
            print(f"  ⚠ Canal {ch_name} descartado: {px_count} < 1000 píxeles")
            cmyk_marks[ch_name] = []

    # =========================================================
    # PASO 3: Guardar 3 archivos JPG
    # =========================================================
    os.makedirs(output_dir, exist_ok=True)

    guardar_imagen_mascaras(img_bgr, cmyk_marks, diag_por_canal, k_marks, 
                           output_dir, name_no_ext)
    guardar_imagen_resultado(img_bgr, cmyk_marks, k_marks, 
                            output_dir, name_no_ext, filename)
    guardar_imagen_calculos(img_bgr, cmyk_marks, k_marks, 
                           output_dir, name_no_ext, filename)

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


def procesar_lote(input_dir='.', output_dir='resultados_v3',
                  calibration_method='distance',
                  reference_size_mm=None):
    """
    Procesa TODAS las imágenes .jpg de un directorio.
    Guarda 3 archivos JPG por imagen sin mostrar visualizaciones.
    """
    template = create_crosshair_template(
        size=101, ring_radius=40, ring_thickness=8,
        cross_thickness=10, cross_length=90
    )
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    print(f'Imágenes encontradas: {len(image_paths)}')

    for img_path in image_paths:
        procesar_y_guardar_imagen(img_path, template, output_dir, 
                                 show_diagnostics=False,
                                 calibration_method=calibration_method,
                                 reference_size_mm=reference_size_mm)
    
    plt.close('all')
    print(f'\n✓ Procesado lote completo. Resultados en: {output_dir}')


def main_single(imagen_path='20250925_142228.jpg',
                calibration_method='distance',
                reference_size_mm=None):
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
                             reference_size_mm=reference_size_mm)

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
                        help='Tamaño real del registro en mm (ej: 10 para 1 cm)')
    parser.add_argument('--ref_size_px', type=float, default=None,
                        help='Píxeles que ocupa el registro en la imagen')
    
    args = parser.parse_args()
    
    if args.batch:
        # Modo batch: procesa todas las imágenes del directorio
        procesar_lote(args.input_dir, args.output_dir, 
                      calibration_method=args.calib_method,
                      reference_size_mm=args.ref_size_mm)
    else:
        # Modo single: procesa una sola imagen con visualización
        main_single(args.imagen,
                    calibration_method=args.calib_method,
                    reference_size_mm=args.ref_size_mm)