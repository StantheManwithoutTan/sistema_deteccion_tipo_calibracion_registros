import cv2
import numpy as np
import os

from config import CMY_CROP_RANGES, COLORS_LABEL, offsets_label
from image_utils import sharpen_image, preprocess_image, multi_scale_template_match, non_max_suppression
from detection import crear_imagen_canal_color


def process_single_image_v3(img_path, template, output_dir, roi_margin=230,
                              focal_mm=4.0, sensor_width_mm=5.6,
                              distancia_camara_plano_mm=110,
                              apply_sharpening=False, sharpening_strength=1.0,
                              calibration_method='distance',  # NUEVO
                              reference_size_mm=None,         # NUEVO
                              reference_size_px=None):        # NUEVO
    """
    Pipeline completo v3.2 con dos métodos de calibración.
    
    calibration_method: 'distance' o 'reference_size'
    """
    filename    = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]

    print(f"\n{'='*60}\nProcesando: {filename}\n{'='*60}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'  ⚠ No se pudo cargar {filename}, saltando...')
        return

    # Sharpening opcional
    if apply_sharpening:
        img_bgr = sharpen_image(img_bgr, strength=sharpening_strength)
        print(f'  ✓ Sharpening aplicado (strength={sharpening_strength})')

    image_width_px  = max(img_bgr.shape[0], img_bgr.shape[1])
    tamano_pixel_mm = sensor_width_mm / image_width_px
    
    # ✓ CALCULAR mm_por_px según el método elegido
    if calibration_method == 'distance':
        mm_por_px = (tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
        calib_info = f'Dist. camara-plano: {distancia_camara_plano_mm} mm | Focal: {focal_mm} mm'
    elif calibration_method == 'reference_size':
        if reference_size_mm is None or reference_size_px is None:
            print('  ⚠ ERROR: reference_size_mm y reference_size_px son requeridos')
            return
        mm_por_px = reference_size_mm / reference_size_px
        calib_info = f'Ref. tamaño: {reference_size_mm} mm / {reference_size_px} px'
    else:
        print(f'  ⚠ Método de calibración inválido: {calibration_method}')
        return

    print(f'  Calibración: {calib_info}')
    print(f'  Factor óptico: 1 px = {mm_por_px:.4f} mm')

    # ── PASO 1: Detectar K ──────────────────────────────────────────────
    lab_p = preprocess_image(img_bgr)
    L_f, _, _ = cv2.split(lab_p)
    # v3.2: threshold 0.35 + escalas ampliadas
    k_det  = multi_scale_template_match(L_f, template,
                                         scales=np.arange(0.15, 3.2, 0.1), threshold=0.35)
    k_marks = non_max_suppression(k_det, radius=110)
    print(f'  Marcas K: {len(k_marks)}')

    if len(k_marks) == 0:
        print('  ⚠ Sin marcas K.')
        out = img_bgr.copy()
        cv2.putText(out, 'Sin marcas K', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(output_dir, f'{name_no_ext}_resultado.jpg'), out)
        return

    # ── PASO 2: Detectar CMY con imagen separada ────────────────────────
    cmyk_marks    = {'K': k_marks}
    batch_iso     = {}   # {ch: img_isolated} para guardar máscaras
    batch_near    = {}   # {ch: mask_near}    para guardar máscaras

    for ch_name, ch_info in CMY_CROP_RANGES.items():
        marks_canal = []
        iso_imgs, near_masks = [], []

        for mark_idx, (kcx, kcy, kscore, kscale) in enumerate(k_marks[:1]):  # v3.2
            img_h, img_w = img_bgr.shape[:2]
            rx1 = max(int(kcx) - roi_margin, 0)
            ry1 = max(int(kcy) - roi_margin, 0)
            rx2 = min(int(kcx) + roi_margin, img_w)
            ry2 = min(int(kcy) + roi_margin, img_h)
            crop_bgr    = img_bgr[ry1:ry2, rx1:rx2].copy()
            k_local_cx  = int(kcx) - rx1
            k_local_cy  = int(kcy) - ry1

            img_isolated, mask_full, mask_near, _, _ = crear_imagen_canal_color(
                crop_bgr, ch_name, ch_info, k_local_cx, k_local_cy, search_radius=110
            )
            iso_imgs.append((img_isolated, rx1, ry1, k_local_cx, k_local_cy, kscale))
            near_masks.append(mask_near)

            px_count = cv2.countNonZero(mask_near)
            best_cx, best_cy, best_score, best_scale = kcx, kcy, 0.05, kscale
            method_used = 'prediccion'

            if px_count > 30:
                isolated_gray = cv2.cvtColor(img_isolated, cv2.COLOR_BGR2GRAY)
                clahe2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
                iso_enh = clahe2.apply(isolated_gray)
                test_scales = np.arange(max(0.2, kscale - 0.2), kscale + 0.3, 0.05)
                local_det = multi_scale_template_match(
                    iso_enh, template, scales=test_scales, threshold=0.2
                )
                if local_det:
                    local_det.sort(key=lambda x: x[2], reverse=True)
                    dcx, dcy, dscore, dscale = local_det[0]
                    best_cx, best_cy = dcx + rx1, dcy + ry1
                    best_score, best_scale = dscore, dscale
                    method_used = 'template'

            ys, xs = np.where(mask_near > 0)
            if len(xs) > 0:
                # v3.2: promedio ponderado (antes: mediana)
                dists_f   = np.hypot(xs - k_local_cx, ys - k_local_cy)
                weights_f = np.exp(-dists_f / 20.0)
                best_cx = int(np.average(xs, weights=weights_f)) + rx1
                best_cy = int(np.average(ys, weights=weights_f)) + ry1
                if method_used == 'prediccion':
                    best_score  = round(min(px_count / 300.0, 1.0), 3)
                    best_scale  = kscale
                    method_used = 'promedio_ponderado'
                else:
                    method_used += '+promedio_ponderado'

            marks_canal.append((best_cx, best_cy, best_score, best_scale))

        cmyk_marks[ch_name] = marks_canal
        batch_iso[ch_name]  = iso_imgs
        batch_near[ch_name] = near_masks

    # ── A. Guardar imagen de MÁSCARAS por canal ─────────────────────────
    ch_list  = ['C', 'M', 'Y']
    cell_h, cell_w = 230, 230
    n_marks  = len(k_marks)
    mask_panel = np.zeros((3 * cell_h, n_marks * 2 * cell_w + n_marks * 6, 3), dtype=np.uint8)

    for row, ch_name in enumerate(ch_list):
        draw_color_bgr = CMY_CROP_RANGES[ch_name].get('color_display', (200,200,0))
        for mi in range(n_marks):
            if mi >= len(batch_iso.get(ch_name, [])):
                continue
            img_iso, rx1, ry1, klx, kly, kscale = batch_iso[ch_name][mi]
            m_near = batch_near[ch_name][mi]

            iso_draw = img_iso.copy()
            cx_det, cy_det = cmyk_marks[ch_name][mi][0], cmyk_marks[ch_name][mi][1]
            lx_det, ly_det = int(cx_det) - rx1, int(cy_det) - ry1
            cv2.circle(iso_draw, (lx_det, ly_det), int(40 * kscale), draw_color_bgr, 2)
            cv2.circle(iso_draw, (lx_det, ly_det), 4, draw_color_bgr, -1)
            cv2.drawMarker(iso_draw, (klx, kly), (180,180,180), cv2.MARKER_CROSS, 14, 1)

            near_colored = np.zeros_like(img_iso)
            near_colored[m_near > 0] = draw_color_bgr
            cv2.circle(near_colored, (klx, kly), 80, (80,80,80), 1)
            cv2.circle(near_colored, (lx_det, ly_det), int(40 * kscale), draw_color_bgr, 2)

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

    for mi in range(n_marks):
        for ci, lbl in enumerate(['Imagen aislada', 'Mascara cerca K']):
            col_off = (mi * 2 + ci) * cell_w + mi * 6
            cv2.putText(mask_panel, f'K-{mi} {lbl}', (col_off + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

    masks_path = os.path.join(output_dir, f'{name_no_ext}_mascaras.jpg')
    cv2.imwrite(masks_path, mask_panel)
    print(f'  ✓ Máscaras guardadas: {os.path.basename(masks_path)}')

    # ── B. Guardar RESULTADO final (original | anotado) ─────────────────
    mcx, mcy, _, mscale = k_marks[0]
    rx1 = max(int(mcx) - roi_margin, 0); ry1 = max(int(mcy) - roi_margin, 0)
    rx2 = min(int(mcx) + roi_margin, img_bgr.shape[1])
    ry2 = min(int(mcy) + roi_margin, img_bgr.shape[0])

    roi_orig = img_bgr[ry1:ry2, rx1:rx2].copy()
    cv2.putText(roi_orig, filename, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    roi_ann  = img_bgr[ry1:ry2, rx1:rx2].copy()
    for ch_name in ['C', 'M', 'Y', 'K']:
        color_bgr = COLORS_LABEL[ch_name]
        for (cx, cy, score, scale) in cmyk_marks.get(ch_name, [])[:1]:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                lx, ly = int(cx) - rx1, int(cy) - ry1
                r = int(40 * scale)
                cv2.circle(roi_ann, (lx, ly), r, color_bgr, 2)
                cv2.circle(roi_ann, (lx, ly), 5, color_bgr, -1)
                cv2.drawMarker(roi_ann, (lx, ly), color_bgr, cv2.MARKER_CROSS, 16, 2)
                if ch_name != 'K':
                    k_lx = int(k_marks[0][0]) - rx1
                    k_ly = int(k_marks[0][1]) - ry1
                    cv2.line(roi_ann, (lx, ly), (k_lx, k_ly), color_bgr, 1, cv2.LINE_AA)
                ox, oy = offsets_label.get(ch_name, (10, -10))
                llx, lly = lx + ox, ly + oy
                lbl = f'{ch_name} s={score:.2f}'
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(roi_ann, (llx-2, lly-th-4), (llx+tw+2, lly+4), (0,0,0), -1)
                cv2.putText(roi_ann, lbl, (llx, lly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

    n_det = sum(1 for v in cmyk_marks.values() if len(v) > 0)
    cv2.putText(roi_ann, f'Analizada — {n_det}/4 canales', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    sep = np.full((roi_orig.shape[0], 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([roi_orig, sep, roi_ann])
    res_path = os.path.join(output_dir, f'{name_no_ext}_resultado.jpg')
    cv2.imwrite(res_path, combined)
    print(f'  ✓ Resultado guardado: {os.path.basename(res_path)}')

    # ── C. Guardar panel de CÁLCULOS EN MM ─────────────────────────────
    positions_mm = {}
    for ch in ['C', 'M', 'Y', 'K']:
        if cmyk_marks.get(ch) and len(cmyk_marks[ch]) > 0:
            positions_mm[ch] = (cmyk_marks[ch][0][0], cmyk_marks[ch][0][1])

    info_lines = [
        f'Archivo: {filename}',
        f'Factor optico: 1 px = {mm_por_px:.4f} mm  |  {calib_info}',
        '',
        '--- Desalineamiento respecto a K ---',
    ]
    if len(k_marks) > 0:
        kx, ky = k_marks[0][0], k_marks[0][1]
        for ch in ['C', 'M', 'Y']:
            if positions_mm.get(ch):
                cx, cy = positions_mm[ch]
                dx_px, dy_px = cx - kx, cy - ky
                dp = np.hypot(dx_px, dy_px)
                dx_mm = (dx_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                dy_mm = (dy_px * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                dm    = (dp    * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
                info_lines.append(
                    f'  {ch}-K:  Dx={dx_mm:+.3f} mm   Dy={dy_mm:+.3f} mm   dist={dm:.3f} mm  ({dp:.1f} px)'
                )
            else:
                info_lines.append(f'  {ch}: no detectado')

    info_lines += ['', '--- Distancias entre todos los pares ---']
    pnames = list(positions_mm.keys())
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            n1, n2 = pnames[i], pnames[j]
            dx = positions_mm[n1][0] - positions_mm[n2][0]
            dy = positions_mm[n1][1] - positions_mm[n2][1]
            dp = np.hypot(dx, dy)
            dm = (dp * tamano_pixel_mm * distancia_camara_plano_mm) / focal_mm
            info_lines.append(f'  {n1}-{n2}:  {dm:.3f} mm  ({dp:.1f} px)')

    font_c  = cv2.FONT_HERSHEY_SIMPLEX
    lh, pad = 28, 14
    pw, ph  = 780, pad * 2 + lh * (len(info_lines) + 1)
    calc_panel = np.full((ph, pw, 3), 20, dtype=np.uint8)
    cv2.rectangle(calc_panel, (0, 0), (pw - 1, ph - 1), (60, 60, 60), 2)
    for idx, line in enumerate(info_lines):
        y = pad + (idx + 1) * lh
        color = (200, 200, 200)
        for ch, bgr in [('C', (255,255,0)), ('M', (255,0,255)),
                         ('Y', (0,255,255)), ('K', (180,180,180))]:
            if line.strip().startswith(ch + '-') or line.strip().startswith(ch + ':'):
                color = bgr; break
        cv2.putText(calc_panel, line, (pad, y), font_c, 0.52, color, 1, cv2.LINE_AA)

    mm_path = os.path.join(output_dir, f'{name_no_ext}_calculos_mm.jpg')
    cv2.imwrite(mm_path, calc_panel)
    print(f'  ✓ Cálculos mm guardados: {os.path.basename(mm_path)}')
    print(f'  ✓ Canales detectados: {n_det}/4')