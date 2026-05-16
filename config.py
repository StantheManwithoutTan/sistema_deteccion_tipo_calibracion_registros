import numpy as np

# --- Rangos HSV extendidos para detección en crops (portados de v2) ---
CMY_CROP_RANGES = {
    'C': {
        'hsv_ranges': [
            (np.array([90,  30, 50]), np.array([125, 255, 220])),   # cian puro concentrado
            (np.array([88,  15, 30]), np.array([130,  80, 160])),   # cian oscuro/mezclado
        ],
        'label_color': (255, 255, 0),
        'nombre': 'Cyan',
        'color_display': (200, 200, 0),
    },
    'M': {
        'hsv_ranges': [
            (np.array([  0, 50, 30]), np.array([ 15, 255, 255])),  # rojo-magenta (ampliado v3.2)
            (np.array([160, 30, 30]), np.array([179, 255, 255])),  # magenta-rojo wrap (ampliado)
            (np.array([130, 25, 25]), np.array([165, 255, 255])),  # magenta puro (ampliado)
            (np.array([140, 15, 15]), np.array([179, 120, 200])),  # magenta diluido/muy desaturado
        ],
        'label_color': (255, 0, 255),
        'nombre': 'Magenta',
        'color_display': (200, 0, 200),
        'usar_lab_bgr': True,  # v3.2: activa componente LAB/BGR para magenta
    },
    'Y': {
        'hsv_ranges': [
            (np.array([10,  8, 20]), np.array([45, 255, 220])),    # amarillo puro a diluido
            (np.array([ 5,  5, 15]), np.array([55, 160, 180])),   # amarillo muy desaturado/oscuro
            (np.array([15, 15, 10]), np.array([38,  80, 140])),   # amarillo mezclado con K
        ],
        'label_color': (0, 255, 255),
        'nombre': 'Amarillo',
        'color_display': (0, 200, 200),
    },
}

COLORS_LABEL = {
    'C': (255, 255, 0),
    'M': (255, 0, 255),
    'Y': (0, 255, 255),
    'K': (180, 180, 180),
}
offsets_label = {'C': (-80, -30), 'M': (15, -30), 'Y': (-80, 40), 'K': (15, 40)}

# Parámetros ópticos
distancia_camara_plano_mm = 110
focal_mm = 4.0
sensor_width_mm = 5.6

# ── Sharpening ─────────────────────────────────────────────────────────────
# Activar/desactivar el filtro de sharpening al cargar cada imagen.
# strength: 0.0 = sin cambio | 0.5 = suave | 1.0 = normal | 2.0 = fuerte
APPLY_SHARPENING    = False   # True para habilitar
SHARPENING_STRENGTH = 1.0

# ── Método de clustering para posición final (v3.2) ────────────────────
# 'promedio_ponderado' : promedio ponderado por exp(-dist/sigma) — DEFAULT
# 'mediana'           : mediana clásica (legacy v3.1)
CLUSTERING_METHOD = 'promedio_ponderado'
CLUSTERING_SIGMA  = 40.0   # sigma en píxeles