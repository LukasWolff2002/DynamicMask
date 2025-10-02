#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera máscaras binarias a partir de detecciones YOLO sobre imágenes 1024x1024.
Entrada: carpeta con .bmp/.tif/.tiff (y .jpg opcional)
Salida:
 - Por cada imagen X, un archivo X_mask.tiff (8-bit, 1 canal) con fondo negro y detecciones en blanco, suavizadas.
 - Si OVERLAY_ORIGINAL=True, además X_overlay.tiff (RGBA) con la foto original de fondo y detecciones en blanco semitransparente.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2  # <-- Para suavizado morfológico

# =========================
# CONFIGURACIÓN DEL USUARIO
# =========================
MODEL_PATH   = "Weights/test_carbopol_region.pt"  # ruta a tu modelo YOLO
IMAGES_DIR   = "Photos"                # carpeta con imágenes
OUTPUT_DIR   = "Masks"                 # carpeta de salida para las máscaras
CONF_THRESH  = 0.25                    # confianza mínima
DEVICE       = "0"                     # "cpu", "0" para GPU 0, etc.

# --- Parámetros de suavizado (post-proceso) ---
APPLY_SMOOTHING   = True     # desactívalo si no quieres suavizado
KERNEL_SIZE       = 5        # tamaño del kernel (impar: 3,5,7,...)
MORPH_ITER        = 1        # iteraciones para morph
REMOVE_MIN_AREA   = 50       # px^2; elimina blobs más pequeños que esto (0 desactiva)
GAUSSIAN_BLUR_K   = 0        # impar >=3; 0 desactiva. p.ej. 3 o 5
FINAL_THRESHOLD   = 128      # umbral final tras blur (si se usa)

# --- Overlay de la imagen original ---
OVERLAY_ORIGINAL  = True     # True = guardar versión RGBA con detecciones encima
OVERLAY_ALPHA     = 128      # transparencia de las detecciones (0=transparente, 255=blanco sólido)
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

def postprocess_mask(mask_img: Image.Image) -> Image.Image:
    if not APPLY_SMOOTHING:
        return mask_img
    mask = np.array(mask_img, dtype=np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    k = max(3, KERNEL_SIZE | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
    if REMOVE_MIN_AREA > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(num_labels, dtype=bool)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            keep[i] = area >= REMOVE_MIN_AREA
        cleaned = np.zeros_like(mask)
        cleaned[np.isin(labels, np.where(keep)[0])] = 255
        mask = cleaned
    if GAUSSIAN_BLUR_K and GAUSSIAN_BLUR_K >= 3 and GAUSSIAN_BLUR_K % 2 == 1:
        mask = cv2.GaussianBlur(mask, (GAUSSIAN_BLUR_K, GAUSSIAN_BLUR_K), 0)
        _, mask = cv2.threshold(mask, FINAL_THRESHOLD, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask, mode="L")

def create_overlay(original_gray: Image.Image, mask: Image.Image, alpha: int) -> Image.Image:
    """Devuelve imagen RGBA con el original en gris de fondo y detecciones blancas con alpha."""
    base = original_gray.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    mask_arr = np.array(mask, dtype=np.uint8)
    overlay_arr = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
    overlay_arr[mask_arr > 0] = [255, 255, 255, alpha]  # blanco con transparencia alpha
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined

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

    mask = postprocess_mask(mask)

    # Guardar máscara binaria
    out_path = out_dir / f"{img_path.stem}_mask.tiff"
    mask.convert("L").save(out_path, format="TIFF", compression="raw")

    # Guardar overlay opcional
    if OVERLAY_ORIGINAL:
        overlay_img = create_overlay(gray, mask, OVERLAY_ALPHA)
        out_overlay = out_dir / f"{img_path.stem}_overlay.tiff"
        overlay_img.save(out_overlay, format="TIFF", compression="raw")

def main():
    images_dir = Path(IMAGES_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    exts = ("*.jpg", "*.bmp", "*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.BMP", "*.JPG", "*.jpeg", "*.JPEG")
    files = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    files = sorted(files)

    if not files:
        print("No se encontraron imágenes.")
        return

    for f in tqdm(files, desc="Procesando"):
        try:
            process_image(model, f, out_dir)
        except Exception as e:
            print(f"[ADVERTENCIA] Error procesando {f.name}: {e}")

    print(f"Listo. Máscaras guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
