from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


DATASET_CLASSES: Dict[str, List[str]] = {
    "gtsrb": ["speed_limit_30", "stop", "yield", "no_entry"],
    "tt100k": ["speed_limit", "turn_left", "turn_right", "no_parking"],
    "miotcd": ["car", "bus", "truck", "motorcycle"],
}


COLOR_BANK = [
    (225, 85, 70),
    (65, 150, 95),
    (70, 120, 225),
    (225, 180, 60),
    (160, 95, 225),
    (70, 210, 200),
]



def _draw_sign(draw: ImageDraw.ImageDraw, cls: str, size: int, color: tuple[int, int, int]) -> None:
    margin = size // 7
    if "stop" in cls:
        points = []
        for i in range(8):
            angle = math.pi / 8 + i * math.pi / 4
            x = size / 2 + (size / 2 - margin) * math.cos(angle)
            y = size / 2 + (size / 2 - margin) * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)
    elif "yield" in cls:
        draw.polygon(
            [(size / 2, margin), (size - margin, size - margin), (margin, size - margin)],
            fill=color,
        )
    elif "entry" in cls or "parking" in cls:
        draw.ellipse((margin, margin, size - margin, size - margin), fill=color)
    elif "turn" in cls:
        draw.rectangle((margin, margin, size - margin, size - margin), fill=(245, 245, 245), outline=color, width=4)
        if "left" in cls:
            draw.polygon(
                [(margin + 10, size / 2), (size / 2, margin + 10), (size / 2, size / 3), (size - margin - 10, size / 3), (size - margin - 10, 2 * size / 3), (size / 2, 2 * size / 3), (size / 2, size - margin - 10)],
                fill=color,
            )
        else:
            draw.polygon(
                [(size - margin - 10, size / 2), (size / 2, margin + 10), (size / 2, size / 3), (margin + 10, size / 3), (margin + 10, 2 * size / 3), (size / 2, 2 * size / 3), (size / 2, size - margin - 10)],
                fill=color,
            )
    else:
        draw.circle = getattr(draw, "ellipse")
        draw.ellipse((margin, margin, size - margin, size - margin), fill=(245, 245, 245), outline=color, width=5)
        draw.text((size // 3, size // 3), str(len(cls) % 9), fill=color)



def _draw_vehicle(draw: ImageDraw.ImageDraw, cls: str, size: int, color: tuple[int, int, int]) -> None:
    base_y = int(size * 0.62)
    if cls == "bus":
        draw.rounded_rectangle((10, base_y - 26, size - 10, base_y + 6), radius=8, fill=color)
        draw.rectangle((20, base_y - 20, size - 20, base_y - 5), fill=(220, 235, 250))
    elif cls == "truck":
        draw.rounded_rectangle((8, base_y - 24, int(size * 0.68), base_y + 6), radius=6, fill=color)
        draw.rounded_rectangle((int(size * 0.68), base_y - 18, size - 10, base_y + 6), radius=6, fill=(min(color[0] + 20, 255), min(color[1] + 20, 255), min(color[2] + 20, 255)))
    elif cls == "motorcycle":
        draw.line((15, base_y, size // 2, base_y - 18, size - 18, base_y - 5), fill=color, width=6)
        draw.line((size // 2, base_y - 18, size // 2 + 10, base_y - 30), fill=color, width=5)
    else:
        draw.rounded_rectangle((12, base_y - 22, size - 12, base_y + 4), radius=10, fill=color)
        draw.polygon([(26, base_y - 22), (42, base_y - 38), (size - 42, base_y - 38), (size - 26, base_y - 22)], fill=color)
    draw.ellipse((18, base_y - 2, 34, base_y + 14), fill=(35, 35, 35))
    draw.ellipse((size - 34, base_y - 2, size - 18, base_y + 14), fill=(35, 35, 35))



def make_sample(dataset: str, cls: str, out_path: Path, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    background = np.zeros((size, size, 3), dtype=np.uint8)
    base = np.array([rng.integers(160, 245), rng.integers(160, 245), rng.integers(160, 245)], dtype=np.uint8)
    background[:] = base
    noise = rng.normal(0, 8, size=background.shape).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    color = COLOR_BANK[seed % len(COLOR_BANK)]

    if dataset in {"gtsrb", "tt100k"}:
        _draw_sign(draw, cls, size, color)
    else:
        _draw_vehicle(draw, cls, size, color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny mock traffic datasets for TAFDG smoke tests.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CLASSES.keys()), required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=48)
    args = parser.parse_args()

    root = Path(args.output_dir) / "train"
    classes = DATASET_CLASSES[args.dataset]
    for class_idx, cls in enumerate(classes):
        for sample_idx in range(args.samples_per_class):
            out_path = root / cls / f"{sample_idx:04d}.png"
            make_sample(
                dataset=args.dataset,
                cls=cls,
                out_path=out_path,
                size=args.image_size,
                seed=class_idx * 1000 + sample_idx,
            )
    print(f"Created mock dataset at: {root}")


if __name__ == "__main__":
    main()
