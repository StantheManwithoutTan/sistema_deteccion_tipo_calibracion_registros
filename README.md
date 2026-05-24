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



# Con distancia personalizada (150 mm)
python main_single.py --imagen 20250925_142228.jpg --calib_method distance --distancia_mm 150

# Con tamaño de referencia (marca K = 10 mm)
python main_single.py --imagen 20250925_142228.jpg --calib_method reference_size --ref_size_mm 10

# Batch con distancia personalizada
python main_single.py --batch --input_dir ./fotos --calib_method distance --distancia_mm 120


```

Esto abre las ventanas de matplotlib con los diagnósticos de máscaras,
igual que ejecutar el notebook celda por celda.

### Opción B — Procesar todas las imágenes en lote

```bash
python main_batch.py --input_dir . --output_dir resultados_v3

python main_single.py --batch --input_dir . --output_dir ./resultados_v4
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
