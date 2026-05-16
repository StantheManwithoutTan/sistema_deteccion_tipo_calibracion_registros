"""
main_batch.py
-------------
Ejecuta el pipeline de detección CMYK sobre TODAS las imágenes .jpg
de un directorio, guardando los resultados como archivos JPG.

Uso:
    python main_batch.py
    python main_batch.py --input_dir ./fotos --output_dir ./resultados
"""

import os
import glob
import argparse
from config import APPLY_SHARPENING, SHARPENING_STRENGTH
from image_utils import create_crosshair_template
from main_single import procesar_lote  # Importar de main_single

def main(input_dir='.', output_dir='resultados_v3'):
    procesar_lote(input_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='.', help='Directorio con imágenes')
    parser.add_argument('--output_dir', default='resultados_v4', help='Directorio de salida')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)