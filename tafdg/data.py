from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset

from .config import TAFDGConfig


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


UrlSpec = Union[str, Sequence[str]]


PRESET_DATASETS: Dict[str, Dict[str, object]] = {
    "gtsrb": {
        "default_domains": ["day", "night", "fog", "motion"],
        "description": "German Traffic Sign Recognition Benchmark with synthetic domainization.",
        "urls": {
            # The legacy benchmark.ini.rub.de archive links now redirect to a page that returns 404.
            # The current official dataset page points to the sid.erda.dk public archive instead.
            "train": [
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
                "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
            ],
            "test": [
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
                "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
            ],
            "test_gt": [
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
                "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip",
            ],
        },
    },
    "tt100k": {
        "default_domains": ["day", "night", "rain", "fog"],
        "description": "Tsinghua-Tencent 100K traffic-sign dataset converted to classification crops with synthetic domainization.",
        "urls": {
            "data": ["http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip"],
            "code": ["http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/code.zip"],
        },
    },
    "miotcd": {
        "default_domains": ["day", "night", "fog", "jpeg"],
        "description": "MIO-TCD vehicle classification dataset with synthetic domainization.",
        "urls": {
            "classification": ["https://tcd.miovision.com/static/dataset/MIO-TCD-Classification.tar"],
            "classification_code": ["https://tcd.miovision.com/static/dataset/MIO-TCD-Classification-Code.tar"],
        },
    },
}


@dataclass(frozen=True)
class BasePresetSample:
    path: str
    label_name: str
    bbox: Optional[Tuple[float, float, float, float]] = None


Sample5 = Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]


class BasicImageTransform:
    def __init__(self, image_size: int, train: bool, allow_hflip: bool = False) -> None:
        self.image_size = image_size
        self.train = train
        self.allow_hflip = allow_hflip

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.train and self.allow_hflip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        return tensor


class ImageFolderDomainDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, str]], transform: BasicImageTransform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, domain = self.samples[idx]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, label, domain


class StyledVisionDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]],
        transform: BasicImageTransform,
        base_seed: int,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, domain, style, bbox = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if bbox is not None:
                image = crop_with_context(image, bbox)
            image = apply_domain_style(image, style=style, seed=_stable_style_seed(path, style, self.base_seed))
            tensor = self.transform(image)
        return tensor, label, domain


class TensorDomainDataset(Dataset):
    def __init__(self, tensors: List[torch.Tensor], labels: List[int], domains: List[str]) -> None:
        self.tensors = tensors
        self.labels = labels
        self.domains = domains

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        return self.tensors[idx], self.labels[idx], self.domains[idx]


class ClientSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y, _ = self.base_dataset[self.indices[idx]]
        return x, y


@dataclass
class FederatedBenchmark:
    client_datasets: List[ClientSubsetDataset]
    test_dataset: Dataset
    source_domains: List[str]
    holdout_domain: str
    class_names: List[str]
    val_dataset: Optional[Dataset] = None


def _stable_style_seed(path: str, style: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{path}|{style}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def _list_image_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]


def _resolve_classification_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    candidate_names = [
        "train",
        "Train",
        "training",
        "Training",
        "images",
        "Images",
        "Final_Training",
        "Final_Training/Images",
        "Final_Training/Final_Training/Images",
        "GTSRB/Final_Training/Images",
    ]
    for name in candidate_names:
        candidate = root / name
        if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
            return candidate
    if any(child.is_dir() for child in root.iterdir()):
        return root
    raise ValueError(f"No class folders were found under {root}")


def _scan_classification_root(data_root: str | Path) -> Tuple[Path, List[str], Dict[str, List[str]]]:
    class_root = _resolve_classification_root(data_root)
    class_names = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found under {class_root}")

    class_to_paths: Dict[str, List[str]] = {}
    for class_name in class_names:
        paths = sorted(str(p) for p in _list_image_files(class_root / class_name))
        if paths:
            class_to_paths[class_name] = paths
    class_names = [name for name in class_names if name in class_to_paths]
    if not class_names:
        raise ValueError(f"No readable images found under {class_root}")
    return class_root, class_names, class_to_paths


def _scan_imagefolder_domains(data_root: str, domains: Optional[List[str]]) -> Tuple[List[str], Dict[str, List[Tuple[str, str]]], List[str]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    available_domains = sorted([p.name for p in root.iterdir() if p.is_dir()])
    use_domains = domains if domains else available_domains
    if not use_domains:
        raise ValueError(f"No domains found under: {data_root}")

    class_names_set = set()
    domain_samples: Dict[str, List[Tuple[str, str]]] = {}
    for domain in use_domains:
        domain_dir = root / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Missing domain directory: {domain_dir}")
        samples: List[Tuple[str, str]] = []
        for class_dir in sorted([p for p in domain_dir.iterdir() if p.is_dir()]):
            class_names_set.add(class_dir.name)
            for image_path in _list_image_files(class_dir):
                samples.append((str(image_path), class_dir.name))
        if not samples:
            raise ValueError(f"No images found in domain: {domain}")
        domain_samples[domain] = samples
    class_names = sorted(class_names_set)
    return use_domains, domain_samples, class_names



def _split_per_domain_per_class(
    samples: List[Sample5],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample5], List[Sample5]]:
    """
    Split source-domain samples into train / val in a stratified way per (domain, class).
    This keeps the FedDG protocol closer to the paper: source domains provide train/val,
    while the holdout domain is used only for test.
    """
    if val_ratio <= 0.0:
        return list(samples), []

    rng = np.random.default_rng(seed)
    grouped: Dict[Tuple[str, int], List[int]] = {}
    for idx, (_path, label, domain, _style, _bbox) in enumerate(samples):
        grouped.setdefault((str(domain), int(label)), []).append(idx)

    val_indices = set()
    for _, idxs in grouped.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) <= 1:
            continue
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        n_val = min(n_val, len(idxs) - 1)
        val_indices.update(idxs[:n_val])

    train_samples = [sample for i, sample in enumerate(samples) if i not in val_indices]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_indices]
    return train_samples, val_samples


def _audit_and_raise(
    train_samples: Sequence[Sample5],
    val_samples: Sequence[Sample5],
    test_samples: Sequence[Sample5],
    class_names: Sequence[str],
    source_domains: Sequence[str],
    holdout_domain: str,
) -> None:
    """
    Hard-stop on obviously broken benchmarks so we do not silently produce fake 100% results.
    """
    if len(class_names) <= 1:
        raise ValueError(
            'benchmark build failed: only one class was parsed. '
            'Usually this means the dataset root was wrong or labels were mis-read.'
        )
    if not train_samples:
        raise ValueError('benchmark build failed: source train split is empty.')
    if not test_samples:
        raise ValueError('benchmark build failed: holdout test split is empty.')

    train_labels = {int(x[1]) for x in train_samples}
    test_labels = {int(x[1]) for x in test_samples}
    val_labels = {int(x[1]) for x in val_samples}

    if len(train_labels) <= 1:
        raise ValueError('benchmark build failed: train split collapsed to one class.')
    if len(test_labels) <= 1:
        raise ValueError('benchmark build failed: test split collapsed to one class.')

    max_label = len(class_names) - 1
    for split_name, label_set in {
        'train': train_labels,
        'val': val_labels,
        'test': test_labels,
    }.items():
        if not label_set:
            continue
        if min(label_set) < 0 or max(label_set) > max_label:
            raise ValueError(
                f'benchmark build failed: {split_name} labels fall outside [0, {max_label}].'
            )

    source_domain_set = set(source_domains)
    if holdout_domain in source_domain_set:
        raise ValueError('benchmark build failed: holdout domain leaked into source domain list.')

    train_domains = {x[2] for x in train_samples}
    val_domains = {x[2] for x in val_samples}
    test_domains = {x[2] for x in test_samples}

    if holdout_domain in train_domains or holdout_domain in val_domains:
        raise ValueError('benchmark build failed: holdout domain leaked into train/val splits.')

    if test_domains != {holdout_domain}:
        raise ValueError('benchmark build failed: test split contains non-holdout domains.')


def _build_imagefolder_benchmark(cfg: TAFDGConfig) -> FederatedBenchmark:
    domains, domain_samples, class_names = _scan_imagefolder_domains(cfg.data_root, cfg.domains)
    holdout_domain = cfg.holdout_domain or domains[cfg.holdout_index]
    if holdout_domain not in domains:
        raise ValueError(f'Holdout domain {holdout_domain} not found in {domains}')

    label_map = {name: idx for idx, name in enumerate(class_names)}
    source_domains = [d for d in domains if d != holdout_domain]

    train_transform = BasicImageTransform(cfg.image_size, train=True, allow_hflip=cfg.random_hflip)
    eval_transform = BasicImageTransform(cfg.image_size, train=False, allow_hflip=False)

    source_samples: List[Sample5] = []
    test_samples: List[Sample5] = []

    for domain in domains:
        packed = [
            (path, label_map[label_name], domain, domain, None)
            for path, label_name in domain_samples[domain]
        ]
        if domain == holdout_domain:
            test_samples.extend(packed)
        else:
            source_samples.extend(packed)

    train_samples, val_samples = _split_per_domain_per_class(source_samples, cfg.val_ratio, cfg.seed)
    _audit_and_raise(train_samples, val_samples, test_samples, class_names, source_domains, holdout_domain)

    train_dataset = StyledVisionDataset(train_samples, transform=train_transform, base_seed=cfg.seed)
    val_dataset = (
        StyledVisionDataset(val_samples, transform=eval_transform, base_seed=cfg.seed + 123)
        if val_samples else None
    )
    test_dataset = StyledVisionDataset(test_samples, transform=eval_transform, base_seed=cfg.seed + 997)

    client_datasets = _build_client_subsets_from_train_samples(
        train_dataset=train_dataset,
        train_samples=train_samples,
        source_domains=source_domains,
        cfg=cfg,
    )

    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=class_names,
    )


def _image_to_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return array


def _array_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _add_fog(array: np.ndarray, strength: float = 0.22) -> np.ndarray:
    h, w, _ = array.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    center_x = w / 2.0
    center_y = h / 2.0
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius = radius / max(radius.max(), 1.0)
    haze = (1.0 - 0.35 * radius)[..., None]
    fog = 255.0 * np.ones_like(array)
    return array * (1.0 - strength * haze) + fog * (strength * haze)


def _add_rain(image: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    n_lines = max(12, int((width * height) / 900))
    for _ in range(n_lines):
        x0 = int(rng.integers(0, max(width, 1)))
        y0 = int(rng.integers(0, max(height, 1)))
        length = int(rng.integers(max(8, height // 18), max(12, height // 8 + 1)))
        dx = int(rng.integers(4, 10))
        draw.line((x0, y0, min(width - 1, x0 + dx), min(height - 1, y0 + length)), fill=(220, 230, 255, 95), width=1)
    merged = Image.alpha_composite(image, overlay)
    return merged.convert("RGB")


def _apply_jpeg_artifact(image: Image.Image, quality: int = 28) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_domain_style(image: Image.Image, style: str, seed: int) -> Image.Image:
    style = style.lower()
    image = image.convert("RGB")

    if style in {"clean", "identity", "base"}:
        return image
    if style in {"day", "bright"}:
        image = ImageEnhance.Brightness(image).enhance(1.12)
        image = ImageEnhance.Contrast(image).enhance(1.05)
        return image
    if style in {"night", "dark"}:
        array = _image_to_array(image)
        tint = np.array([0.85, 0.90, 1.10], dtype=np.float32).reshape(1, 1, 3)
        array = array * 0.50 * tint
        image = _array_to_image(array)
        image = ImageEnhance.Contrast(image).enhance(0.95)
        return image
    if style in {"fog", "haze"}:
        array = _image_to_array(image)
        array = _add_fog(array, strength=0.22)
        image = _array_to_image(array)
        image = ImageEnhance.Contrast(image).enhance(0.90)
        return image
    if style in {"motion", "blur", "gaussian_blur"}:
        return image.filter(ImageFilter.GaussianBlur(radius=1.2))
    if style == "rain":
        image = _add_rain(image, seed)
        image = ImageEnhance.Contrast(image).enhance(0.95)
        return image
    if style == "jpeg":
        return _apply_jpeg_artifact(image, quality=28)
    raise ValueError(f"Unsupported generated domain style: {style}")


def crop_with_context(image: Image.Image, bbox: Tuple[float, float, float, float], pad_ratio: float = 0.12) -> Image.Image:
    width, height = image.size
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    xmin = max(0.0, min(xmin, width - 1))
    ymin = max(0.0, min(ymin, height - 1))
    xmax = max(xmin + 1.0, min(xmax, width))
    ymax = max(ymin + 1.0, min(ymax, height))
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio
    left = int(max(0.0, math.floor(xmin - pad_x)))
    top = int(max(0.0, math.floor(ymin - pad_y)))
    right = int(min(float(width), math.ceil(xmax + pad_x)))
    bottom = int(min(float(height), math.ceil(ymax + pad_y)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return image.crop((left, top, right, bottom))


def _as_url_candidates(value: UrlSpec) -> List[str]:
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value if str(x).strip()]


def _print_download_banner(name: str, urls: Dict[str, UrlSpec], target_root: Path) -> None:
    print(f"[TAFDG] {name}: dataset not found locally, preparing automatic download.")
    print(f"[TAFDG] cache root: {target_root}")
    for key, value in urls.items():
        candidates = _as_url_candidates(value)
        for idx, url in enumerate(candidates, start=1):
            print(f"[TAFDG] source[{key}][{idx}] = {url}")


def _download_from_single_url(url: str, destination: Path, chunk_size: int = 1024 * 1024, timeout: int = 60) -> Path:
    print(f"[TAFDG] downloading: {url}")
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_path, "wb") as f:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = 100.0 * downloaded / max(total_size, 1)
                print(
                    f"[TAFDG]   downloaded {downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({pct:.1f}%)",
                    end="\r",
                )
    print()
    tmp_path.replace(destination)
    print(f"[TAFDG] saved to: {destination}")
    return destination


def _download_file(urls: UrlSpec, destination: Path, chunk_size: int = 1024 * 1024, retries_per_url: int = 2) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[TAFDG] using cached file: {destination}")
        return destination

    candidates = _as_url_candidates(urls)
    if not candidates:
        raise ValueError("No valid download URLs were provided.")

    errors: List[str] = []
    for url_index, url in enumerate(candidates, start=1):
        for attempt in range(1, retries_per_url + 1):
            try:
                if len(candidates) > 1:
                    print(f"[TAFDG] trying source {url_index}/{len(candidates)} (attempt {attempt}/{retries_per_url})")
                return _download_from_single_url(url, destination=destination, chunk_size=chunk_size)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                error_line = f"{type(exc).__name__}: {exc}"
                errors.append(f"{url} | attempt {attempt}: {error_line}")
                print(f"[TAFDG] download failed: {error_line}")
                tmp_path = destination.with_suffix(destination.suffix + '.part')
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < retries_per_url:
                    time.sleep(1.0)
                    continue
                break
    joined = "\n".join(errors)
    raise RuntimeError(
        "Failed to download the dataset from all configured sources.\n"
        f"Destination: {destination}\n"
        f"Tried URLs:\n{joined}"
    )


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            member_path = destination / member.filename
            if not str(member_path.resolve()).startswith(str(destination.resolve())):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(destination)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = destination / member.name
            if not str(member_path.resolve()).startswith(str(destination.resolve())):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(destination)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    marker = destination / f".extracted_{archive_path.name.replace('.', '_')}"
    if marker.exists():
        return
    print(f"[TAFDG] extracting: {archive_path.name}")
    if archive_path.suffix.lower() == ".zip":
        _safe_extract_zip(archive_path, destination)
    elif archive_path.suffix.lower() in {".tar", ".tgz", ".gz"} or archive_path.name.endswith(".tar.gz"):
        _safe_extract_tar(archive_path, destination)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    marker.write_text("ok", encoding="utf-8")


def _default_preset_root(cfg: TAFDGConfig, preset_name: str) -> Path:
    if cfg.data_root:
        return Path(cfg.data_root)
    return Path(cfg.dataset_cache_dir) / preset_name


def _find_dir_with_file(root: Path, filename: str) -> Optional[Path]:
    if (root / filename).exists():
        return root
    for path in root.rglob(filename):
        return path.parent
    return None


def _find_gtsrb_training_root(root: Path) -> Optional[Path]:
    for path in root.rglob("Final_Training"):
        candidate = path / "Images"
        if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
            return candidate

    candidates = [
        root / "GTSRB/Final_Training/Images",
        root / "Final_Training/Images",
        root / "GTSRB/Training",
        root / "Training",
        root / "Images",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
            return candidate
    return None


def _find_tt100k_raw_root(root: Path) -> Optional[Path]:
    direct = _find_dir_with_file(root, "annotations.json")
    if direct is None:
        return None
    for candidate in [direct, direct / "data"]:
        if (candidate / "annotations.json").exists():
            image_dirs = {"train", "test", "other"}
            if any((candidate / name).exists() for name in image_dirs):
                return candidate
    return direct


def _find_miotcd_root(root: Path) -> Optional[Path]:
    try:
        class_root = _resolve_classification_root(root)
        if class_root.exists() and any(p.is_dir() for p in class_root.iterdir()):
            return root
    except Exception:
        pass

    csv_candidates = ["gt-train.csv", "gt_train.csv", "train.csv", "labels.csv"]
    for name in csv_candidates:
        folder = _find_dir_with_file(root, name)
        if folder is not None:
            return root
    return None


def _ensure_gtsrb_available(cfg: TAFDGConfig) -> Path:
    target_root = _default_preset_root(cfg, "gtsrb")
    train_existing = _find_gtsrb_training_root(target_root)
    test_existing = _find_gtsrb_test_root(target_root)
    test_csv_existing = _find_gtsrb_test_csv(target_root)
    if train_existing is not None and (test_existing is not None or not cfg.preset_use_official_test):
        return target_root
    if not cfg.auto_download:
        raise FileNotFoundError(f"GTSRB data not found under {target_root} and auto_download=False")

    raw_dir = target_root / "raw"
    urls = PRESET_DATASETS["gtsrb"]["urls"]
    assert isinstance(urls, dict)
    _print_download_banner("GTSRB", urls, target_root)
    archive = _download_file(urls["train"], raw_dir / "GTSRB_Final_Training_Images.zip")
    _extract_archive(archive, raw_dir)
    if cfg.preset_use_official_test:
        test_archive = _download_file(urls["test"], raw_dir / "GTSRB_Final_Test_Images.zip")
        _extract_archive(test_archive, raw_dir)
        gt_archive = _download_file(urls["test_gt"], raw_dir / "GTSRB_Final_Test_GT.zip")
        _extract_archive(gt_archive, raw_dir)

    train_existing = _find_gtsrb_training_root(target_root)
    test_existing = _find_gtsrb_test_root(target_root)
    test_csv_existing = _find_gtsrb_test_csv(target_root)
    if train_existing is None:
        raise RuntimeError("GTSRB download finished, but the extracted training directory was not found.")
    if cfg.preset_use_official_test and (test_existing is None or test_csv_existing is None):
        raise RuntimeError("GTSRB download finished, but the official test split or GT-final_test.csv was not found.")
    return target_root


def _ensure_tt100k_available(cfg: TAFDGConfig) -> Path:
    target_root = _default_preset_root(cfg, "tt100k")
    existing = _find_tt100k_raw_root(target_root)
    if existing is not None:
        return existing
    if not cfg.auto_download:
        raise FileNotFoundError(f"TT100K data not found under {target_root} and auto_download=False")

    raw_dir = target_root / "raw"
    urls = PRESET_DATASETS["tt100k"]["urls"]
    assert isinstance(urls, dict)
    _print_download_banner("TT100K", urls, target_root)
    archive = _download_file(urls["data"], raw_dir / "data.zip")
    _extract_archive(archive, raw_dir)

    existing = _find_tt100k_raw_root(target_root)
    if existing is None:
        raise RuntimeError("TT100K download finished, but annotations.json was not found after extraction.")
    return existing


def _ensure_miotcd_available(cfg: TAFDGConfig) -> Path:
    target_root = _default_preset_root(cfg, "miotcd")
    existing = _find_miotcd_root(target_root)
    if existing is not None:
        return existing
    if not cfg.auto_download:
        raise FileNotFoundError(f"MIO-TCD data not found under {target_root} and auto_download=False")

    raw_dir = target_root / "raw"
    urls = PRESET_DATASETS["miotcd"]["urls"]
    assert isinstance(urls, dict)
    _print_download_banner("MIO-TCD", urls, target_root)
    archive = _download_file(urls["classification"], raw_dir / "MIO-TCD-Classification.tar")
    _extract_archive(archive, raw_dir)
    code_archive = _download_file(urls["classification_code"], raw_dir / "MIO-TCD-Classification-Code.tar")
    _extract_archive(code_archive, raw_dir)

    existing = _find_miotcd_root(target_root)
    if existing is None:
        raise RuntimeError("MIO-TCD download finished, but no recognizable classification structure or CSV labels were found.")
    return existing


def _normalize_relative_path(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _collect_classification_samples(data_root: str | Path) -> Tuple[List[str], Dict[str, List[BasePresetSample]]]:
    _class_root, class_names, class_to_paths = _scan_classification_root(data_root)
    class_to_samples = {
        class_name: [BasePresetSample(path=path, label_name=class_name, bbox=None) for path in paths]
        for class_name, paths in class_to_paths.items()
    }
    return class_names, class_to_samples


def _resolve_existing_image(raw_root: Path, rel_path: str, basename_map: Dict[str, str]) -> Optional[str]:
    rel_path = _normalize_relative_path(rel_path)
    candidates = [
        raw_root / rel_path,
        raw_root / Path(rel_path).name,
        raw_root / "data" / rel_path,
        raw_root / "data" / Path(rel_path).name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    basename = Path(rel_path).name
    return basename_map.get(basename)


def _load_tt100k_samples(raw_root: Path) -> Tuple[List[str], Dict[str, List[BasePresetSample]]]:
    annotations_path = raw_root / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"TT100K annotations.json not found under {raw_root}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        annos = json.load(f)

    image_files = _list_image_files(raw_root)
    basename_map = {p.name: str(p) for p in image_files}
    class_names = [str(x) for x in annos.get("types", [])]
    class_to_samples: Dict[str, List[BasePresetSample]] = {name: [] for name in class_names}

    imgs = annos.get("imgs", {})
    if not isinstance(imgs, dict):
        raise ValueError("TT100K annotations.json does not contain the expected 'imgs' dictionary.")

    for img_entry in imgs.values():
        if not isinstance(img_entry, dict):
            continue
        rel_path = str(img_entry.get("path", "")).strip()
        if not rel_path:
            continue
        image_path = _resolve_existing_image(raw_root, rel_path, basename_map)
        if image_path is None:
            continue
        objects = img_entry.get("objects", []) or []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            category = str(obj.get("category", "")).strip()
            bbox = obj.get("bbox", {}) or {}
            if not category or not isinstance(bbox, dict):
                continue
            try:
                xmin = float(bbox["xmin"])
                ymin = float(bbox["ymin"])
                xmax = float(bbox["xmax"])
                ymax = float(bbox["ymax"])
            except Exception:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            if category not in class_to_samples:
                class_to_samples[category] = []
                class_names.append(category)
            class_to_samples[category].append(
                BasePresetSample(
                    path=image_path,
                    label_name=category,
                    bbox=(xmin, ymin, xmax, ymax),
                )
            )

    class_names = [name for name in class_names if class_to_samples.get(name)]
    if not class_names:
        raise ValueError("No usable labeled crops were parsed from TT100K annotations.")
    class_to_samples = {name: class_to_samples[name] for name in class_names}
    return class_names, class_to_samples


def _guess_csv_columns(fieldnames: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    file_candidates = ["filename", "file", "path", "image", "img", "image_path", "filepath", "sample"]
    label_candidates = ["label", "class", "category", "type", "target"]
    lower_map = {name.lower().strip(): name for name in fieldnames}

    file_col = next((lower_map[name] for name in file_candidates if name in lower_map), None)
    label_col = next((lower_map[name] for name in label_candidates if name in lower_map), None)
    return file_col, label_col


def _find_best_csv(root: Path) -> Optional[Path]:
    preferred = ["gt-train.csv", "gt_train.csv", "train.csv", "labels.csv"]
    for name in preferred:
        found = _find_dir_with_file(root, name)
        if found is not None:
            return found / name
    csvs = sorted(root.rglob("*.csv"))
    return csvs[0] if csvs else None


def _load_csv_label_samples(raw_root: Path) -> Tuple[List[str], Dict[str, List[BasePresetSample]]]:
    csv_path = _find_best_csv(raw_root)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV label file was found under {raw_root}")

    image_files = _list_image_files(raw_root)
    basename_map = {p.name: str(p) for p in image_files}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample else True
        except Exception:
            has_header = True
        if has_header:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            file_col, label_col = _guess_csv_columns(fieldnames)
            if file_col is None or label_col is None:
                if len(fieldnames) >= 2:
                    file_col = fieldnames[0]
                    label_col = fieldnames[1]
                else:
                    raise ValueError(f"Could not infer filename/label columns from CSV: {csv_path}")

            class_to_samples: Dict[str, List[BasePresetSample]] = {}
            for row in reader:
                if not row:
                    continue
                rel_path = str(row.get(file_col, "")).strip()
                label = str(row.get(label_col, "")).strip()
                if not rel_path or not label:
                    continue
                image_path = _resolve_existing_image(raw_root, rel_path, basename_map)
                if image_path is None:
                    continue
                class_to_samples.setdefault(label, []).append(BasePresetSample(path=image_path, label_name=label, bbox=None))
        else:
            f.seek(0)
            reader2 = csv.reader(f)
            class_to_samples = {}
            for row in reader2:
                if len(row) < 2:
                    continue
                rel_path = str(row[0]).strip()
                label = str(row[1]).strip()
                if not rel_path or not label:
                    continue
                image_path = _resolve_existing_image(raw_root, rel_path, basename_map)
                if image_path is None:
                    continue
                class_to_samples.setdefault(label, []).append(BasePresetSample(path=image_path, label_name=label, bbox=None))

    class_names = sorted([name for name, samples in class_to_samples.items() if samples])
    if not class_names:
        raise ValueError(f"CSV label parsing succeeded but no usable image rows were found in {csv_path}")
    class_to_samples = {name: class_to_samples[name] for name in class_names}
    return class_names, class_to_samples



def _resolve_preset_samples(cfg: TAFDGConfig, preset_name: str) -> Tuple[List[str], Dict[str, List[BasePresetSample]]]:
    """
    Key fix:
    1. GTSRB must be parsed from the actual training-class root instead of a higher-level bundle root.
    2. Proxy-domain experiments should follow the same source-domain / holdout-domain protocol.
    """
    if preset_name == 'gtsrb':
        root = _ensure_gtsrb_available(cfg)
        train_root = _find_gtsrb_training_root(root)
        if train_root is None:
            raise FileNotFoundError('GTSRB training root was not found after download/extraction.')
        class_names, class_to_samples = _collect_classification_samples(train_root)
        if len(class_names) <= 1:
            raise ValueError(
                f'GTSRB parsing only found {len(class_names)} class(es) under {train_root}. '
                'This usually means the dataset root was resolved incorrectly.'
            )
        return class_names, class_to_samples

    if preset_name == 'tt100k':
        root = _ensure_tt100k_available(cfg)
        try:
            return _load_tt100k_samples(root)
        except Exception:
            return _collect_classification_samples(root)

    if preset_name == 'miotcd':
        root = _ensure_miotcd_available(cfg)
        try:
            return _load_csv_label_samples(root)
        except Exception:
            return _collect_classification_samples(root)

    raise ValueError(f'Unsupported preset dataset: {preset_name}')




def _allocate_proxy_class_samples(
    records: List[BasePresetSample],
    all_domains: Sequence[str],
    holdout_domain: str,
) -> Tuple[List[Tuple[str, BasePresetSample]], List[BasePresetSample]]:
    """
    More realistic proxy-domain allocation.

    Previous versions reserved only ONE sample per class for the holdout domain,
    which made test sets absurdly small (for GTSRB this becomes exactly 43 samples)
    and led to unstable or misleading evaluation. Here we keep a class-wise slice
    for the holdout domain while ensuring the source side still has enough data.
    """
    if len(records) < 2:
        return [], []

    source_domains = [d for d in all_domains if d != holdout_domain]
    if not source_domains:
        return [], []

    n = len(records)
    # About one quarter goes to the holdout proxy domain, but never all of it.
    holdout_count = max(1, int(round(n * 0.25)))
    max_holdout = max(1, n - 1)
    holdout_count = min(holdout_count, max_holdout)

    holdout_records: List[BasePresetSample] = list(records[:holdout_count])
    source_records = list(records[holdout_count:])
    if not source_records:
        holdout_records = [records[0]]
        source_records = list(records[1:])

    assigned_sources: List[Tuple[str, BasePresetSample]] = []
    for idx, record in enumerate(source_records):
        domain_name = source_domains[idx % len(source_domains)]
        assigned_sources.append((domain_name, record))
    return assigned_sources, holdout_records

def _build_generated_domain_benchmark(cfg: TAFDGConfig, preset_name: str) -> FederatedBenchmark:
    """
    Proxy-domain benchmark for GTSRB / TT100K / MIOTCD.

    Key properties:
    - holdout domain is test-only
    - source domains provide train/val only
    - a class is kept only if both source and holdout can receive samples
    - style transformation is deterministic per sample, avoiding label leakage via parsing bugs
    """
    class_names_raw, class_to_samples = _resolve_preset_samples(cfg, preset_name)
    preset = PRESET_DATASETS[preset_name]
    all_domains = list(cfg.domains or preset['default_domains'])
    if len(all_domains) < 2:
        raise ValueError('At least two domains are required for federated domain generalization.')

    holdout_domain = cfg.holdout_domain or all_domains[cfg.holdout_index]
    if holdout_domain not in all_domains:
        raise ValueError(f'Holdout domain {holdout_domain} not found in {all_domains}')

    source_domains = [d for d in all_domains if d != holdout_domain]
    train_transform = BasicImageTransform(cfg.image_size, train=True, allow_hflip=cfg.random_hflip)
    eval_transform = BasicImageTransform(cfg.image_size, train=False, allow_hflip=False)

    rng = np.random.default_rng(cfg.seed)
    source_samples: List[Sample5] = []
    test_samples: List[Sample5] = []
    kept_class_names: List[str] = []

    min_required = max(2, int(getattr(cfg, 'min_class_total_samples', 2)))

    for class_name in class_names_raw:
        records = list(class_to_samples[class_name])
        if len(records) < min_required:
            continue
        rng.shuffle(records)
        assigned_sources, holdout_records = _allocate_proxy_class_samples(records, all_domains, holdout_domain)
        if not assigned_sources or not holdout_records:
            continue

        label_idx = len(kept_class_names)
        kept_class_names.append(class_name)

        for domain_name, record in assigned_sources:
            source_samples.append((record.path, label_idx, domain_name, domain_name, record.bbox))
        for record in holdout_records:
            test_samples.append((record.path, label_idx, holdout_domain, holdout_domain, record.bbox))

    if len(kept_class_names) <= 1:
        raise ValueError(
            f'Only {len(kept_class_names)} class(es) survived proxy-domain allocation for {preset_name}. '
            'Increase data coverage or lower min_class_total_samples.'
        )

    train_samples, val_samples = _split_per_domain_per_class(source_samples, cfg.val_ratio, cfg.seed)
    _audit_and_raise(train_samples, val_samples, test_samples, kept_class_names, source_domains, holdout_domain)

    train_dataset = StyledVisionDataset(train_samples, transform=train_transform, base_seed=cfg.seed)
    val_dataset = (
        StyledVisionDataset(val_samples, transform=eval_transform, base_seed=cfg.seed + 123)
        if val_samples else None
    )
    test_dataset = StyledVisionDataset(test_samples, transform=eval_transform, base_seed=cfg.seed + 997)

    client_datasets = _build_client_subsets_from_train_samples(
        train_dataset=train_dataset,
        train_samples=train_samples,
        source_domains=source_domains,
        cfg=cfg,
    )

    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=kept_class_names,
    )


def _gaussian_blob(size: int, center_x: float, center_y: float, sigma: float) -> np.ndarray:
    xs = np.linspace(0, size - 1, size, dtype=np.float32)
    ys = np.linspace(0, size - 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def _make_synthetic_sample(
    cls: int,
    domain_id: int,
    sample_id: int,
    num_classes: int,
    image_size: int,
    noise: float,
    style_strength: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    _ = sample_id
    grid_cols = int(math.ceil(math.sqrt(num_classes)))
    grid_rows = int(math.ceil(num_classes / grid_cols))
    cell_w = image_size / grid_cols
    cell_h = image_size / grid_rows
    col = cls % grid_cols
    row = cls // grid_cols
    cx = (col + 0.5) * cell_w + rng.normal(0, 0.08 * cell_w)
    cy = (row + 0.5) * cell_h + rng.normal(0, 0.08 * cell_h)
    sigma = max(image_size / 12.0, 1.5)
    base = _gaussian_blob(image_size, cx, cy, sigma)

    stripes = np.sin(np.linspace(0, np.pi * (domain_id + 1), image_size, dtype=np.float32))[None, :]
    stripes = np.repeat(stripes, image_size, axis=0)
    checker = (((np.indices((image_size, image_size)).sum(axis=0) + domain_id) % 2) * 2 - 1).astype(np.float32)
    texture = 0.5 * stripes + 0.5 * checker

    color_bank = np.array(
        [
            [1.00, 0.45, 0.45],
            [0.45, 1.00, 0.55],
            [0.50, 0.60, 1.00],
            [1.00, 0.85, 0.40],
            [0.75, 0.50, 1.00],
            [0.40, 1.00, 0.95],
        ],
        dtype=np.float32,
    )
    color = color_bank[domain_id % len(color_bank)]

    channels = []
    for c in range(3):
        channel = base * color[c]
        channel = channel + style_strength * (0.15 + 0.1 * c) * texture
        channel = channel + rng.normal(0, noise, size=(image_size, image_size)).astype(np.float32)
        channels.append(channel)
    img = np.stack(channels, axis=0)
    img = np.clip(img, 0.0, 1.0)

    if domain_id % 2 == 1:
        img = img[:, :, ::-1].copy()
    if domain_id % 3 == 2:
        pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
        img = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.tensor(img, dtype=torch.float32)


def _build_synthetic_benchmark(cfg: TAFDGConfig) -> FederatedBenchmark:
    domains = cfg.domains or [f"domain_{i}" for i in range(cfg.num_domains)]
    holdout_domain = cfg.holdout_domain or domains[cfg.holdout_index]
    source_domains = [d for d in domains if d != holdout_domain]
    tensors: List[torch.Tensor] = []
    labels: List[int] = []
    domain_names: List[str] = []
    test_tensors: List[torch.Tensor] = []
    test_labels: List[int] = []
    test_domains: List[str] = []

    rng = np.random.default_rng(cfg.seed)
    for domain_id, domain_name in enumerate(domains):
        for cls in range(cfg.num_classes):
            for sample_id in range(cfg.synthetic_samples_per_class):
                tensor = _make_synthetic_sample(
                    cls=cls,
                    domain_id=domain_id,
                    sample_id=sample_id,
                    num_classes=cfg.num_classes,
                    image_size=cfg.image_size,
                    noise=cfg.synthetic_noise,
                    style_strength=cfg.synthetic_style_strength,
                    rng=rng,
                )
                if domain_name == holdout_domain:
                    test_tensors.append(tensor)
                    test_labels.append(cls)
                    test_domains.append(domain_name)
                else:
                    tensors.append(tensor)
                    labels.append(cls)
                    domain_names.append(domain_name)

    train_dataset = TensorDomainDataset(tensors, labels, domain_names)
    test_dataset = TensorDomainDataset(test_tensors, test_labels, test_domains)
    domain_to_indices: Dict[str, List[int]] = {d: [] for d in source_domains}
    for idx, domain in enumerate(domain_names):
        domain_to_indices[domain].append(idx)

    client_indices = _make_client_partitions(
        labels=labels,
        domain_to_indices=domain_to_indices,
        source_domains=source_domains,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        mode=getattr(cfg, 'preset_split_mode', 'balanced'),
    )
    client_datasets = [ClientSubsetDataset(train_dataset, idxs) for idxs in client_indices]
    class_names = [f"class_{i}" for i in range(cfg.num_classes)]
    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=class_names,
    )


def _repair_empty_clients(client_buckets: List[List[int]]) -> List[List[int]]:
    sizes = [len(bucket) for bucket in client_buckets]
    if min(sizes) > 0:
        return client_buckets
    for idx, bucket in enumerate(client_buckets):
        if bucket:
            continue
        donor = int(np.argmax([len(x) for x in client_buckets]))
        if len(client_buckets[donor]) <= 1:
            continue
        moved_index = client_buckets[donor].pop()
        client_buckets[idx].append(moved_index)
    return client_buckets


def _dirichlet_partition(indices: Sequence[int], labels: Sequence[int], num_clients: int, alpha: float, rng: np.random.Generator) -> List[List[int]]:
    buckets = [[] for _ in range(num_clients)]
    label_to_indices: Dict[int, List[int]] = {}
    for idx in indices:
        label_to_indices.setdefault(int(labels[idx]), []).append(int(idx))

    if alpha <= 0:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        return [list(map(int, part)) for part in np.array_split(shuffled, num_clients)]

    for cls_indices in label_to_indices.values():
        rng.shuffle(cls_indices)
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float32))
        counts = np.floor(probs * len(cls_indices)).astype(int)
        remainder = len(cls_indices) - int(counts.sum())
        if remainder > 0:
            for extra_idx in np.argsort(probs)[-remainder:]:
                counts[int(extra_idx)] += 1
        start = 0
        for client_id, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            buckets[client_id].extend(cls_indices[start:start + count])
            start += count
    return _repair_empty_clients(buckets)


def _balanced_partition(indices: Sequence[int], labels: Sequence[int], num_clients: int, rng: np.random.Generator) -> List[List[int]]:
    buckets = [[] for _ in range(num_clients)]
    label_to_indices: Dict[int, List[int]] = {}
    for idx in indices:
        label_to_indices.setdefault(int(labels[idx]), []).append(int(idx))

    for label in sorted(label_to_indices):
        cls_indices = list(label_to_indices[label])
        rng.shuffle(cls_indices)
        for offset, sample_idx in enumerate(cls_indices):
            buckets[offset % num_clients].append(int(sample_idx))
    return _repair_empty_clients(buckets)


def _sequential_partition(indices: Sequence[int], num_clients: int, rng: np.random.Generator) -> List[List[int]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    return _repair_empty_clients([list(map(int, part)) for part in np.array_split(shuffled, num_clients)])


def _make_client_partitions(
    labels: Sequence[int],
    domain_to_indices: Dict[str, List[int]],
    source_domains: List[str],
    num_clients: int,
    alpha: float,
    seed: int,
    mode: str = 'balanced',
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    clients_per_domain = [num_clients // len(source_domains) for _ in source_domains]
    for i in range(num_clients % len(source_domains)):
        clients_per_domain[i] += 1

    client_indices: List[List[int]] = []
    for domain_name, n_clients in zip(source_domains, clients_per_domain):
        indices = list(domain_to_indices[domain_name])
        rng.shuffle(indices)
        if n_clients <= 1:
            client_indices.append(indices)
            continue
        if mode == 'balanced':
            partitions = _balanced_partition(indices, labels, num_clients=n_clients, rng=rng)
        elif mode == 'sequential':
            partitions = _sequential_partition(indices, num_clients=n_clients, rng=rng)
        elif mode == 'dirichlet':
            partitions = _dirichlet_partition(indices, labels, num_clients=n_clients, alpha=alpha, rng=rng)
        else:
            raise ValueError(f'Unsupported preset_split_mode: {mode}')
        client_indices.extend(partitions)
    client_indices = [idxs for idxs in client_indices if len(idxs) > 0]
    if not client_indices:
        raise ValueError("Failed to create any non-empty client partitions.")
    return client_indices


def available_dataset_names() -> List[str]:
    return ["synthetic", "imagefolder"] + sorted(PRESET_DATASETS.keys())


def default_domains_for_dataset(dataset_name: str, num_domains: int = 4) -> List[str]:
    if dataset_name == "synthetic":
        return [f"domain_{i}" for i in range(num_domains)]
    if dataset_name == "imagefolder":
        return []
    if dataset_name in PRESET_DATASETS:
        return list(PRESET_DATASETS[dataset_name]["default_domains"])
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


def _find_gtsrb_test_root(root: Path) -> Optional[Path]:
    candidates = [
        root / "GTSRB/Final_Test/Images",
        root / "Final_Test/Images",
        root / "Final_Test",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_file() for p in candidate.rglob("*")):
            return candidate
    for path in root.rglob("Final_Test"):
        candidate = path / "Images"
        if candidate.is_dir() and any(p.is_file() for p in candidate.rglob("*")):
            return candidate
        if path.is_dir() and any(p.is_file() for p in path.rglob("*")):
            return path
    return None


def _find_gtsrb_test_csv(root: Path) -> Optional[Path]:
    for name in ["GT-final_test.csv", "GT-final_test.test.csv", "final_test.csv"]:
        found = _find_dir_with_file(root, name)
        if found is not None:
            return found / name
    for path in root.rglob("*.csv"):
        if "final_test" in path.name.lower():
            return path
    return None


def _split_train_val_samples(
    samples: List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]], List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]]]:
    if val_ratio <= 0.0:
        return list(samples), []
    rng = np.random.default_rng(seed)
    group_to_indices: Dict[Tuple[int, str], List[int]] = {}
    for idx, (_path, label, domain, _style, _bbox) in enumerate(samples):
        group_to_indices.setdefault((int(label), str(domain)), []).append(idx)
    val_indices = set()
    for group_indices in group_to_indices.values():
        group_indices = list(group_indices)
        rng.shuffle(group_indices)
        if len(group_indices) <= 1:
            continue
        n_val = max(1, int(round(len(group_indices) * val_ratio)))
        n_val = min(n_val, len(group_indices) - 1)
        val_indices.update(group_indices[:n_val])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples


def _build_client_subsets_from_train_samples(
    train_dataset: Dataset,
    train_samples: List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]],
    source_domains: List[str],
    cfg: TAFDGConfig,
) -> List[ClientSubsetDataset]:
    domain_to_indices: Dict[str, List[int]] = {d: [] for d in source_domains}
    labels: List[int] = []
    for idx, (_path, label, domain, _style, _bbox) in enumerate(train_samples):
        domain_to_indices[domain].append(idx)
        labels.append(label)
    client_indices = _make_client_partitions(
        labels=labels,
        domain_to_indices=domain_to_indices,
        source_domains=source_domains,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        mode=getattr(cfg, 'preset_split_mode', 'balanced'),
    )
    return [ClientSubsetDataset(train_dataset, idxs) for idxs in client_indices]



def _build_gtsrb_real_benchmark(cfg: TAFDGConfig) -> FederatedBenchmark:
    bundle_root = _default_preset_root(cfg, "gtsrb")
    _ensure_gtsrb_available(cfg)
    train_root = _find_gtsrb_training_root(bundle_root)
    test_root = _find_gtsrb_test_root(bundle_root)
    test_csv = _find_gtsrb_test_csv(bundle_root)
    if train_root is None or test_root is None or test_csv is None:
        return _build_generated_domain_benchmark(cfg, "gtsrb")

    class_names, class_to_samples = _collect_classification_samples(train_root)
    all_domains = list(cfg.domains or PRESET_DATASETS["gtsrb"]["default_domains"])
    if len(all_domains) < 2:
        raise ValueError("At least two domains are required for federated domain generalization.")
    holdout_domain = cfg.holdout_domain or all_domains[cfg.holdout_index]
    if holdout_domain not in all_domains:
        raise ValueError(f"Holdout domain {holdout_domain} not found in {all_domains}")
    source_domains = [d for d in all_domains if d != holdout_domain]
    label_map = {name: idx for idx, name in enumerate(class_names)}

    train_transform = BasicImageTransform(cfg.image_size, train=True, allow_hflip=cfg.random_hflip)
    eval_transform = BasicImageTransform(cfg.image_size, train=False, allow_hflip=False)

    rng = np.random.default_rng(cfg.seed)
    train_samples: List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]] = []
    for class_name in class_names:
        records = list(class_to_samples[class_name])
        rng.shuffle(records)
        source_buckets = [[] for _ in source_domains]
        for idx, record in enumerate(records):
            source_buckets[idx % len(source_domains)].append(record)
        for domain_name, chunk in zip(source_domains, source_buckets):
            for record in chunk:
                train_samples.append((record.path, label_map[class_name], domain_name, domain_name, record.bbox))

    train_samples, val_samples = _split_train_val_samples(train_samples, cfg.val_ratio, cfg.seed)

    basename_map = {p.name: str(p) for p in _list_image_files(test_root)}
    test_samples: List[Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]] = []
    with open(test_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=';,') if sample else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            filename = str(row.get("Filename", "") or row.get("filename", "")).strip()
            class_id = row.get("ClassId", row.get("classid", row.get("class_id", "")))
            if not filename or class_id in (None, ""):
                continue
            image_path = basename_map.get(Path(filename).name)
            if image_path is None:
                candidate = test_root / filename
                if candidate.exists():
                    image_path = str(candidate)
            if image_path is None:
                continue
            label_idx = int(class_id)
            if label_idx < 0 or label_idx >= len(class_names):
                continue
            test_samples.append((image_path, label_idx, holdout_domain, holdout_domain, None))

    if not train_samples or not test_samples:
        raise ValueError("GTSRB official split parsing produced an empty train or test set.")

    train_dataset = StyledVisionDataset(train_samples, transform=train_transform, base_seed=cfg.seed)
    val_dataset = StyledVisionDataset(val_samples, transform=eval_transform, base_seed=cfg.seed + 123) if val_samples else None
    test_dataset = StyledVisionDataset(test_samples, transform=eval_transform, base_seed=cfg.seed + 997)
    client_datasets = _build_client_subsets_from_train_samples(train_dataset, train_samples, source_domains, cfg)
    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=class_names,
    )




def _dataset_domain_counts(dataset: Dataset) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    samples = getattr(dataset, 'samples', None)
    if samples is not None:
        for sample in samples:
            domain = str(sample[2])
            counts[domain] = counts.get(domain, 0) + 1
        return counts
    domains = getattr(dataset, 'domains', None)
    if domains is not None:
        for domain in domains:
            domain = str(domain)
            counts[domain] = counts.get(domain, 0) + 1
        return counts
    return counts


def _dataset_label_counts(dataset: Dataset) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    samples = getattr(dataset, 'samples', None)
    if samples is not None:
        for sample in samples:
            label = int(sample[1])
            counts[label] = counts.get(label, 0) + 1
        return counts
    labels = getattr(dataset, 'labels', None)
    if labels is not None:
        for label in labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
        return counts
    return counts


def summarize_benchmark(benchmark: FederatedBenchmark) -> Dict[str, object]:
    train_size = int(sum(len(ds) for ds in benchmark.client_datasets))
    val_size = int(len(benchmark.val_dataset)) if benchmark.val_dataset is not None else 0
    test_size = int(len(benchmark.test_dataset))

    client_sizes = [int(len(ds)) for ds in benchmark.client_datasets]
    train_domain_counts: Dict[str, int] = {}
    for domain_name in benchmark.source_domains:
        train_domain_counts[domain_name] = 0
    for ds in benchmark.client_datasets:
        base = getattr(ds, 'base_dataset', None)
        indices = getattr(ds, 'indices', [])
        if base is None:
            continue
        samples = getattr(base, 'samples', None)
        if samples is not None:
            for base_idx in indices:
                domain = str(samples[base_idx][2])
                train_domain_counts[domain] = train_domain_counts.get(domain, 0) + 1
            continue
        domains = getattr(base, 'domains', None)
        if domains is not None:
            for base_idx in indices:
                domain = str(domains[base_idx])
                train_domain_counts[domain] = train_domain_counts.get(domain, 0) + 1

    val_domain_counts = _dataset_domain_counts(benchmark.val_dataset) if benchmark.val_dataset is not None else {}
    test_domain_counts = _dataset_domain_counts(benchmark.test_dataset)
    test_label_counts = _dataset_label_counts(benchmark.test_dataset)

    return {
        'holdout_domain': benchmark.holdout_domain,
        'source_domains': list(benchmark.source_domains),
        'num_classes': int(len(benchmark.class_names)),
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'num_clients': int(len(benchmark.client_datasets)),
        'client_size_min': int(min(client_sizes)) if client_sizes else 0,
        'client_size_mean': float(np.mean(client_sizes)) if client_sizes else 0.0,
        'client_size_max': int(max(client_sizes)) if client_sizes else 0,
        'train_domain_counts': train_domain_counts,
        'val_domain_counts': val_domain_counts,
        'test_domain_counts': test_domain_counts,
        'test_label_coverage': int(len(test_label_counts)),
    }

def build_benchmark(cfg: TAFDGConfig) -> FederatedBenchmark:
    if cfg.dataset == 'synthetic':
        return _build_synthetic_benchmark(cfg)

    if cfg.dataset == 'imagefolder':
        return _build_imagefolder_benchmark(cfg)

    if cfg.dataset in PRESET_DATASETS:
        # For FedDG / TAFDG proxy-domain experiments, always follow:
        # source domains -> train/val, holdout domain -> test.
        return _build_generated_domain_benchmark(cfg, cfg.dataset)

    raise ValueError(f'Unsupported dataset type: {cfg.dataset}')
