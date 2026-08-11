# Detección de Marcas CMYK — v3.2

## Estructura del proyecto

```
deteccion_marcas/
│
├── config.py          # Parámetros globales (rangos HSV, óptica, sharpening)
├── image_utils.py     # Preprocesamiento, template matching, NMS, sharpening
├── detection.py       # Detección por canal de color, diagnóstico visual
├── batch.py           # Pipeline completo para procesamiento por lote
│
├── main_single.py     # ▶ Procesa UNA imagen con visualización (equivale al notebook)
├── main_batch.py      # ▶ Procesa TODAS las imágenes .jpg de un directorio
│
└── requirements.txt   # Dependencias
```

---

## Guía de configuración en Visual Studio Code

### 1. Instalar Python

Descarga Python 3.9 o superior desde https://www.python.org/downloads/  
Durante la instalación, marca la opción **"Add Python to PATH"**.

### 2. Instalar la extensión de Python en VS Code

1. Abre VS Code.
2. Ve a **Extensiones** (Ctrl+Shift+X).
3. Busca `Python` (publicado por Microsoft) e instálala.

### 3. Abrir el proyecto

1. En VS Code: **Archivo → Abrir Carpeta** y selecciona la carpeta `deteccion_marcas`.

### 4. Crear un entorno virtual (recomendado)

Abre la terminal integrada de VS Code con **Ctrl+`** y ejecuta:

```bash
# Crear entorno virtual (command prompt)
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (macOS / Linux)
source venv/bin/activate
```

### 5. Instalar dependencias

Con el entorno activado:

```bash
pip install -r requirements.txt
```

### 6. Seleccionar el intérprete de Python en VS Code

1. Presiona **Ctrl+Shift+P**.
2. Escribe `Python: Select Interpreter`.
3. Elige el que tenga `venv` en la ruta (por ejemplo: `./venv/Scripts/python.exe`).

---

## Cómo correr el proyecto

### Opción A — Procesar una sola imagen (con diagnósticos visuales)

Coloca tu imagen `.jpg` en la carpeta del proyecto y ejecuta:

```bash

foto calibrada
python main_single.py --imagen 20250925_142033.jpg

foto con un color no calibrado
python main_single.py --imagen 20251117_170440.jpg

foto con dos colores no calibrados
python main_single.py --imagen 20260217_162010.jpg


foto con todos los colores no calibrados
python main_single.py --imagen 20250925_142228.jpg



# Con tamaño de referencia (marca K = 1 cm = 10 mm)
python main_single.py --imagen 20250925_142228.jpg --calib_method reference_size --ref_size_mm 10

# Lote con referencia
python main_single.py --batch --input_dir . --output_dir resultados_v4 --calib_method reference_size --ref_size_mm 10



# Procesar todos (por defecto)
python main_single.py --imagen foto_real_empresa.jpg

# Con distancia personalizada (150 mm)
python main_single.py --imagen 20250925_142228.jpg --calib_method distance --distancia_mm 150
python main_single.py --imagen antes_calib_1.jpg --calib_method distance --distancia_mm 100

# Con tamaño de referencia (marca K = 10 mm)
python main_single.py --imagen 20250925_142228.jpg --calib_method reference_size --ref_size_mm 10

# Batch con distancia personalizada
python main_single.py --batch --input_dir ./fotos --calib_method distance --distancia_mm 120


# Procesar solo Cyan y Magenta
python main_single.py --imagen 20250925_142228.jpg --canales C,M

# Procesar solo Amarillo
python main_single.py --imagen 20250925_142228.jpg --canales Y


python main_single.py --imagen 20260315_235619.jpg --canales Y



python main_single.py --batch --peores_por_color --input_dir "...\registro_dataset_1_8_26_v4_2_a_10_4k" --output_dir "resultados_peores_4k" --top_n 5

python main_single.py --batch --input_dir "..\CREACION_DATASET_SINTETICO_TESIS\registro_dataset_7_8_26_colores_aleatorios" --output_dir "resultados_col_aleatorios_completos" 





python main_single.py --peores_por_color --input_dir "..\CREACION_DATASET_SINTETICO_TESIS\registro_dataset_1_8_26_v4_2_a_10_4k" --output_dir "resultados_peores_4k" --top_n 5

python main_single.py --peores_por_color --input_dir "..\CREACION_DATASET_SINTETICO_TESIS\registro_dataset_7_8_26_colores_aleatorios" --output_dir "resultados_peores_colores_aleatorios" --top_n 5


```

Esto abre las ventanas de matplotlib con los diagnósticos de máscaras,
igual que ejecutar el notebook celda por celda.

### Opción B — Procesar todas las imágenes en lote

```bash
python main_batch.py --input_dir . --output_dir resultados_v3

python main_single.py --batch --input_dir . --output_dir ./resultados_grayscale_magenta

python main_single.py --batch --input_dir "..\CREACION_DATASET_SINTETICO_TESIS\registro_dataset_1_8_26_v4_2_a_10_4k" --output_dir "resultados_total_4k"
```

Guarda en `resultados_v3/` tres archivos por imagen:
- `*_mascaras.jpg` — panel de máscaras C, M, Y
- `*_resultado.jpg` — crop original vs. anotado
- `*_calculos_mm.jpg` — tabla de desalineamientos en mm

### Opción C — Correr directamente desde VS Code

1. Abre `main_single.py` o `main_batch.py`.
2. Presiona **F5** o el botón ▶ (Run Python File).
3. Para pasar argumentos: ve a **Ejecutar → Agregar configuración** y edita `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Imagen única",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main_single.py",
            "args": ["--imagen", "20250925_142228.jpg"]
        },
        {
            "name": "Lote completo",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main_batch.py",
            "args": ["--input_dir", ".", "--output_dir", "resultados_v3"]
        }
    ]
}
```

---

## Ajustar parámetros

Todos los parámetros principales están en `config.py`:

| Variable               | Descripción                                      |
|------------------------|--------------------------------------------------|
| `APPLY_SHARPENING`     | `True` / `False` para activar el sharpening      |
| `SHARPENING_STRENGTH`  | Intensidad del sharpening (0.0 – 2.0)            |
| `distancia_camara_plano_mm` | Distancia cámara al plano de impresión      |
| `focal_mm`             | Focal de la cámara en mm                         |
| `sensor_width_mm`      | Ancho del sensor en mm                           |
| `CLUSTERING_SIGMA`     | Sigma del promedio ponderado para posición final |
| `CMY_CROP_RANGES`      | Rangos HSV por canal (C, M, Y)                   |# sistema_deteccion_tipo_calibracion_registros






Resumen del pipeline

El flujo real de detección es este:

1. Se preprocesa la imagen para mejorar el contraste.
2. Se detecta la marca negra de referencia K con template matching.
3. Con la posición de K se recorta una región de interés alrededor de cada marca.
4. Dentro de ese recorte se construyen máscaras por canal de color C, M y Y.
5. La detección se restringe a una zona cercana a K para evitar falsos positivos.
6. Se estima la posición final de cada canal combinando template matching y refinamiento geométrico.
7. Se calcula el radio visual de la marca y se genera salida diagnóstica.
8. Se convierten desplazamientos de píxeles a milímetros para evaluar la calibración.


Qué hace cada archivo y cada función

En color_analysis.py están las funciones que resumen el color y ayudan a refinar la posición.

get_representative_color(mask, bgr_img): 
    - calcula el color representativo de los píxeles detectados dentro de una máscara. Toma la mediana de B, G y R, y devuelve ese color en RGB, su equivalente HSV y el número de píxeles usados.
weighted_median(values, weights): 
    - calcula una mediana ponderada, dando más peso a los píxeles más cercanos al centro esperado. Se usa como refinamiento robusto de la posición.
    - analyze_hue_range(crop_hsv, mask_near, ch_info): analiza la distribución del tono Hue en los píxeles detectados. Devuelve mínimos, máximos, media, mediana, moda, desviación estándar y saturación/brillo medios para comparar lo observado con los rangos configurados.


En color_masks.py se construyen las máscaras de color para cada canal.

crear_imagen_canal_color(crop_bgr, ch_name, ch_info, k_local_cx, k_local_cy, search_radius=110): es una de las funciones más importantes del pipeline. Toma el recorte alrededor de K y genera:

- una imagen aislada del canal de color,
- una máscara completa del color,
- una máscara recortada alrededor de K,
- una imagen mejorada con LAB + CLAHE, máscaras intermedias de diagnóstico.

La lógica cambia según el canal:

    - Para C usa reglas en HSV + LAB + comparativas BGR.
    - Para M usa principalmente HSV y un refuerzo en escala de grises.
    - Para Y usa HSV y una validación adicional en LAB.


En position.py se estima la ubicación final de cada marca.

- detectar_posicion(mask_near, img_isolated, template, kcx, kcy, kscale, rx1, ry1, k_local_cx, k_local_cy, threshold=0.2, score_normalizer=300, px_count=0): intenta localizar la marca con tres niveles de decisión:

1. primero, template matching sobre la imagen aislada,
2. si no encuentra suficiente evidencia, usa un centroide ponderado,
3. finalmente refina la posición con mediana ponderada.
4. Devuelve la posición global, score, escala, método usado y los píxeles detectados.

- compute_detect_radius(xs, ys, best_cx, best_cy, best_scale, rx1, ry1): calcula el radio visual de la marca detectada usando el percentil 85 de las distancias de los píxeles al centro estimado.

En visualization.py se generan las salidas visuales de diagnóstico.

- crear_overlay_canal(crop_bgr, mask_near, px_count, k_local_cx, k_local_cy, search_radius, draw_color_bgr, local_det_cx, local_det_cy, detect_radius): crea un overlay donde se pintan los píxeles detectados, el radio de búsqueda, la cruz de K y el círculo de la detección final.
- plot_diagnostico_canal(...): arma una figura de diagnóstico por canal con el crop original, la versión preprocesada, la imagen aislada, el overlay, las máscaras y un histograma de Hue.
- generar_diagnostico_mascaras(diag_por_canal, preprocesada_titulo='PREPROCESADA', show_plot=True): genera una figura resumen para los canales C, M y Y en conjunto.


En pipeline.py está el flujo orquestador.

- detectar_canal_con_imagen_separada(...): procesa un canal específico dentro del crop alrededor de K. Hace el recorte, la normalización opcional del fondo blanco, crea la máscara del canal, analiza color, detecta posición, calcula radio y prepara todos los datos de diagnóstico.
- procesar_imagen_completa(img_bgr, template, roi_margin=230, search_radius=110): ejecuta el pipeline completo sobre una imagen: detecta K primero y luego procesa C, M y Y alrededor de esa referencia.

Además, __init__.py solo reexporta las funciones anteriores para que el resto del proyecto las importe desde un solo lugar.