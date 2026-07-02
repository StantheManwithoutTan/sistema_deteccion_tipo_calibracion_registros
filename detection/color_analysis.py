import cv2
import numpy as np


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
