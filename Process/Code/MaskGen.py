import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_PATH = "../sd3_t2i.png"

CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# Point
# POINT = None
POINT = [500, 300]      # [x, y]
POINT_LABEL = 1        

# Bbox
BOX = None              # [x1, y1, x2, y2]
# BOX = [100, 150, 600, 700]

OUTPUT_MASK = "../mask.png"
OUTPUT_OVERLAY = "../overlay.png"

# False: White remains, black changes
# True : Black remains, white changes
INVERT_MASK = True

# Save all 4 masks
SAVE_ALL_MASKS = True


def load_rgb_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image, np.array(image)


def save_binary_mask(mask, output_path, invert=False):
    if invert:
        mask = np.logical_not(mask)

    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_uint8)
    mask_img.save(output_path)


def save_overlay(image_pil, mask, output_path, point=None, box=None, invert=False):
    if invert:
        mask = np.logical_not(mask)

    image_rgba = image_pil.convert("RGBA")

    mask_uint8 = (mask.astype(np.uint8) * 160)
    color_layer = Image.new("RGBA", image_pil.size, (255, 0, 0, 0))
    alpha = Image.fromarray(mask_uint8)
    color_layer.putalpha(alpha)

    overlay = Image.alpha_composite(image_rgba, color_layer)
    draw = ImageDraw.Draw(overlay)

    if point is not None:
        x, y = point
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255))

    if box is not None:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0, 255), width=4)

    overlay.save(output_path)



def main():
    if POINT is None and BOX is None:
        raise ValueError("Please at least choose one from point or box")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sam2_model = build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    image_pil, image_np = load_rgb_image(IMAGE_PATH)

    if device == "cuda":
        context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        context = nullcontext()

    point_coords = None
    point_labels = None
    box_array = None

    if POINT is not None:
        point_coords = np.array([POINT], dtype=np.float32)
        point_labels = np.array([POINT_LABEL], dtype=np.int32)

    if BOX is not None:
        box_array = np.array(BOX, dtype=np.float32)

    with torch.inference_mode(), context:
        predictor.set_image(image_np)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            multimask_output=True,
        )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    print("Scores:", scores)
    print("Best mask index:", best_idx)
    print("Best score:", scores[best_idx])

    save_binary_mask(
        best_mask,
        OUTPUT_MASK,
        invert=INVERT_MASK,
    )

    save_overlay(
        image_pil,
        best_mask,
        OUTPUT_OVERLAY,
        point=POINT,
        box=BOX,
        invert=INVERT_MASK,
    )

    print("Saved mask to:", OUTPUT_MASK)
    print("Saved overlay to:", OUTPUT_OVERLAY)

    if SAVE_ALL_MASKS:
        base, ext = os.path.splitext(OUTPUT_MASK)
        for i, mask in enumerate(masks):
            out_path = f"{base}_{i}{ext}"
            save_binary_mask(mask, out_path, invert=INVERT_MASK)
            print("Saved candidate mask:", out_path)


if __name__ == "__main__":
    main()