#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera máscaras binarias a partir de detecciones YOLO sobre imágenes 1024x1024.
Entrada: carpeta con .bmp o .tif/.tiff
Salida: por cada imagen X, un archivo X_mask.tiff (8-bit, 1 canal)
        con fondo negro y detecciones en blanco.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm
import torch

# =========================
# CONFIGURACIÓN DEL USUARIO
# =========================
MODEL_PATH   = "/ruta/a/tu_modelo/best.pt"   # ruta a tu modelo YOLO
IMAGES_DIR   = "/ruta/a/imagenes_1024"       # carpeta con .bmp o .tif/.tiff
OUTPUT_DIR   = "/ruta/a/salida"              # carpeta de salida para las máscaras
CONF_THRESH  = 0.25                          # confianza mínima
DEVICE       = "0"                           # "cpu", "0" para GPU 0, etc.
# =========================


def ensure_gray(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img

def prepare_for_model(gray_img: Image.Image) -> np.ndarray:
    arr = np.array(gray_img)  # (H, W)
    arr3 = np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)
    return arr3

def draw_segmentation_masks(mask_canvas: Image.Image, result) -> bool:
    if getattr(result, "masks", None) is None:
        return False
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        return False

    h, w = masks.orig_shape
    data = masks.data
    if not isinstance(data, torch.Tensor):
        return False

    data_np = data.cpu().numpy()
    canvas_np = np.array(mask_canvas, dtype=np.uint8)
    drew_any = False

    for m in data_np:
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize((w, h), resample=Image.NEAREST)
        m_arr = np.array(m_img)
        canvas_np = np.maximum(canvas_np, (m_arr > 0).astype(np.uint8) * 255)
        drew_any = True

    mask_canvas.paste(Image.fromarray(canvas_np, mode="L"))
    return drew_any

def draw_boxes(mask_canvas: Image.Image, result) -> bool:
    if getattr(result, "boxes", None) is None or result.boxes is None or len(result.boxes) == 0:
        return False
    draw = ImageDraw.Draw(mask_canvas)
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    drew_any = False
    for x1, y1, x2, y2 in boxes_xyxy:
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        draw.rectangle([x1i, y1i, x2i, y2i], fill=255)
        drew_any = True
    return drew_any

def process_image(model: YOLO, img_path: Path, out_dir: Path):
    with Image.open(img_path) as im:
        im.load()
        gray = ensure_gray(im)

    arr3 = prepare_for_model(gray)
    results = model.predict(source=arr3, imgsz=1024, conf=CONF_THRESH, device=DEVICE, verbose=False)

    mask = Image.new("L", gray.size, color=0)

    if len(results) > 0:
        r = results[0]
        drew_masks = draw_segmentation_masks(mask, r)
        if not drew_masks:
            draw_boxes(mask, r)

    out_path = out_dir / f"{img_path.stem}_mask.tiff"
    mask.convert("L").save(out_path, format="TIFF", compression="raw")

def main():
    images_dir = Path(IMAGES_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    exts = ("*.bmp", "*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.BMP")
    files = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    files = sorted(files)

    if not files:
        print("No se encontraron imágenes .bmp o .tif/.tiff.")
        return

    for f in tqdm(files, desc="Procesando"):
        try:
            process_image(model, f, out_dir)
        except Exception as e:
            print(f"[ADVERTENCIA] Error procesando {f.name}: {e}")

    print(f"Listo. Máscaras guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
