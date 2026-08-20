"""
GeoVision AI - Manual + Automatic Dataset Prediction

Usage:

Land-cover:
    python predict.py --landcover inputs/satellite.jpg

Change detection:
    python predict.py --changedet inputs/before.jpg inputs/after.jpg

Automatic dataset mode:
    python predict.py --auto
    python predict.py --auto --limit 5
    python predict.py --auto --landcover-only
    python predict.py --auto --changedet-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import random
try:
    import tifffile
except ImportError:
    tifffile = None

from PIL.ImageChops import overlay
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from geovision.config import get_config
from geovision.models.vit_landcover import build_landcover_vit
from geovision.models.siamese_vit_changedet import (
    build_change_detection_vit,
)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 70)
        print("GPU DETECTED")
        print("=" * 70)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print("=" * 70)
    else:
        device = torch.device("cpu")
        print("CUDA not available - using CPU.")

    return device


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def setup_output_directory(cfg):
    output_dir = Path(cfg.paths.output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPrediction output directory:")
    print(output_dir)

    return output_dir


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint_path}\n"
        )

    print(f"\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Your training code appears to save model_state.
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        # Also support checkpoints containing the raw state_dict.
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# LAND COVER PREDICTION
# ============================================================

def predict_landcover(cfg, image_path, checkpoint_path, output_dir, device):

    print("\n" + "=" * 70)
    print("LAND COVER PREDICTION")
    print("=" * 70)

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"\nInput image not found:\n{image_path}")

    print(f"\nInput image: {image_path}")

    transform = transforms.Compose([
        transforms.Resize(cfg.vit.input_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    tensor = transform(image).unsqueeze(0).to(device)

    model = build_landcover_vit(cfg)
    model = load_checkpoint(model, checkpoint_path, device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()
    predicted_class = cfg.landcover.classes[predicted_index]

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Image       : {image_path.name}")
    print(f"Prediction  : {predicted_class}")
    print(f"Confidence  : {confidence * 100:.2f}%")

    print("\nAll class probabilities:")
    for index, class_name in enumerate(cfg.landcover.classes):
        probability = probabilities[0, index].item()
        print(f"  {class_name:<25} {probability * 100:6.2f}%")
    print("=" * 70)

    # Text result.
    result_file = output_dir / f"{image_path.stem}_landcover_result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("GeoVision AI - Land Cover Prediction\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Prediction: {predicted_class}\n")
        f.write(f"Confidence: {confidence * 100:.2f}%\n\n")
        f.write("Class probabilities:\n")
        for index, class_name in enumerate(cfg.landcover.classes):
            probability = probabilities[0, index].item()
            f.write(f"{class_name}: {probability * 100:.2f}%\n")

    # Visual result: original image with prediction written on it.
    # This is a classification model, so the model predicts the whole image,
    # not a pixel-by-pixel segmentation mask.
    visual = image.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(visual)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, visual.width // 25))
    except Exception:
        font = ImageFont.load_default()

    label = f"GeoVision: {predicted_class} ({confidence * 100:.1f}%)"
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 10
    box = (10, 10, bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((10 + pad, 10 + pad), label, fill=(255, 255, 255), font=font)

    visual_path = output_dir / f"{image_path.stem}_landcover_prediction.jpg"
    visual.save(visual_path, quality=95)

    print(f"\nText result saved to:\n{result_file}")
    print(f"Visual result saved to:\n{visual_path}")

    return predicted_class, confidence


# ============================================================
# CHANGE DETECTION
# ============================================================

def load_change_image(image_path, cfg):
    """
    Load a change-detection input.

    Automatic OSCD mode:
        image_path is an imgs_1_rect/imgs_2_rect DIRECTORY.
        Load B04/B03/B02, stack RGB, divide by scene max, resize to 224.

    Manual mode:
        image_path is a normal image file such as PNG/JPG.
        Load as RGB, scale to [0,1], resize to 224.
    """
    image_path = Path(image_path)
    target = cfg.vit.input_size[:2]

    # ------------------------------------------------------------
    # OSCD scene directory
    # ------------------------------------------------------------
    if image_path.is_dir():
        def read_band(name):
            path = image_path / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Required OSCD band not found:\n{path}"
                )

            if tifffile is not None:
                return tifffile.imread(str(path)).astype(np.float64)

            return np.asarray(Image.open(path)).astype(np.float64)

        r = read_band("B04.tif")
        g = read_band("B03.tif")
        b = read_band("B02.tif")

        if not (r.shape == g.shape == b.shape):
            raise ValueError(
                f"OSCD RGB bands have different shapes: "
                f"B04={r.shape}, B03={g.shape}, B02={b.shape}"
            )

        print(
            f"  OSCD scene: {image_path.name} | "
            f"size={r.shape}"
        )
        print(
            f"  B04 range: {r.min():.0f} - {r.max():.0f}"
        )
        print(
            f"  B03 range: {g.min():.0f} - {g.max():.0f}"
        )
        print(
            f"  B02 range: {b.min():.0f} - {b.max():.0f}"
        )

        rgb = np.stack([r, g, b], axis=-1)

        # EXACT training preprocessing from prepare_oscd.py.
        rgb = rgb / (rgb.max() + 1e-8)
        rgb = rgb.astype(np.float32)

        from skimage.transform import resize as sk_resize
        rgb = sk_resize(
            rgb,
            target,
            anti_aliasing=True
        ).astype(np.float32)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0)

    # ------------------------------------------------------------
    # Manual image file
    # ------------------------------------------------------------
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    print(f"  Manual image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        target,
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

    return tensor.unsqueeze(0)


def save_change_visuals(before_dir, after_dir, change_mask, output_dir,
                        round_number, city_name, changed_area):
    """Save T1, T2, highlighted change overlay, and a 3-panel result."""
    def rgb_scene(scene_dir):
        def band(name):
            if tifffile is not None:
                return tifffile.imread(str(scene_dir / name)).astype(np.float32)
            return np.asarray(Image.open(scene_dir / name)).astype(np.float32)

        rgb = np.stack([band("B04.tif"), band("B03.tif"), band("B02.tif")], axis=-1)
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            lo, hi = np.percentile(rgb[..., c], 2), np.percentile(rgb[..., c], 98)
            if hi > lo:
                out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
        return out

    before_rgb = rgb_scene(Path(before_dir))
    after_rgb = rgb_scene(Path(after_dir))

    h, w = after_rgb.shape[:2]
    mask_img = Image.fromarray(change_mask.astype(np.uint8)).resize(
        (w, h), Image.Resampling.NEAREST
    )
    mask = np.asarray(mask_img) > 0

    overlay = after_rgb.copy()
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * np.array([1., 0., 0.])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"round{round_number:02d}_{city_name}"

    before_file = out_dir / f"{prefix}_before.png"
    after_file = out_dir / f"{prefix}_after.png"
    overlay_file = out_dir / f"{prefix}_change_overlay.png"
    panel_file = out_dir / f"{prefix}_visual_result.png"

    plt.imsave(before_file, before_rgb)
    plt.imsave(after_file, after_rgb)
    plt.imsave(overlay_file, overlay)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(before_rgb); ax[0].set_title("BEFORE (T1)")
    ax[1].imshow(after_rgb); ax[1].set_title("AFTER (T2)")
    ax[2].imshow(overlay); ax[2].set_title(f"CHANGES — {changed_area:.2f}%")
    for a in ax: a.axis("off")
    fig.suptitle(f"GeoVision AI — {city_name} — Round {round_number}", fontsize=16)
    fig.tight_layout()
    fig.savefig(panel_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nVisual outputs saved:")
    print(f"  BEFORE : {before_file}")
    print(f"  AFTER  : {after_file}")
    print(f"  OVERLAY: {overlay_file}")
    print(f"  PANEL  : {panel_file}")


def predict_changedet(
    cfg,
    before_path,
    after_path,
    checkpoint_path,
    output_dir,
    device
):

    print("\n" + "=" * 70)
    print("CHANGE DETECTION MANUAL PREDICTION")
    print("=" * 70)

    before_path = Path(before_path)
    after_path = Path(after_path)

    print(f"\nBefore image: {before_path}")
    print(f"After image : {after_path}")

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    before = load_change_image(
        before_path,
        cfg
    )

    after = load_change_image(
        after_path,
        cfg
    )

    before = before.to(device)
    after = after.to(device)

    print(
        f"\nModel input: "
        f"{cfg.vit.input_size[:2]}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_change_detection_vit(cfg)

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        probability_map = model(
            before,
            after
        )

    probability_map = probability_map[0, 0].cpu().numpy()

    # ============================================================
    # CREATE CHANGE MASK
    # ============================================================

    threshold = 0.5

    change_mask = probability_map > threshold

    changed_area = int(change_mask.sum())
    total_pixels = int(change_mask.size)

    change_percentage = (
        changed_area / total_pixels * 100
        if total_pixels > 0 else 0.0
    )

    # ============================================================
    # CREATE VISUAL OVERLAY
    # ============================================================

    # Convert PyTorch tensor -> NumPy image
    before_np = before[0].detach().cpu().numpy()

    # CHW -> HWC
    before_np = np.transpose(before_np, (1, 2, 0))

    # Normalize for visualization
    before_np = before_np - before_np.min()
    if before_np.max() > 0:
        before_np = before_np / before_np.max()

    # Convert to 8-bit RGB
    before_np = (before_np * 255).astype(np.uint8)

    # Create overlay
    overlay = before_np.copy()

    # Mark detected changes in RED
    overlay[change_mask] = [255, 0, 0]
    
    

    # --------------------------------------------------------
    # Create binary change map
    # --------------------------------------------------------

    # Print probability statistics BEFORE thresholding.
    print("\nProbability map statistics:")
    print(f"  Min  : {probability_map.min():.6f}")
    print(f"  Max  : {probability_map.max():.6f}")
    print(f"  Mean : {probability_map.mean():.6f}")
    print(f"  >0.1 : {np.mean(probability_map >= 0.1) * 100:.2f}%")
    print(f"  >0.2 : {np.mean(probability_map >= 0.2) * 100:.2f}%")
    print(f"  >0.3 : {np.mean(probability_map >= 0.3) * 100:.2f}%")
    print(f"  >0.5 : {np.mean(probability_map >= 0.5) * 100:.2f}%")

    # Use a configurable threshold. Default remains 0.5 until we inspect
    # the actual model probability distribution.
    threshold = getattr(cfg, "changedet_threshold", 0.1)

    change_mask = (
        probability_map >= threshold
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Save change map
    # --------------------------------------------------------

    # Use city names for OSCD directory inputs and image stems for
    # ordinary manual image inputs.
    if before_path.is_dir():
        city_name = before_path.parent.name
        before_name = "T1"
        after_name = "T2"
    else:
        city_name = "manual"
        before_name = before_path.stem
        after_name = after_path.stem

    change_map_path = (
        output_dir /
        f"{city_name}_{before_name}_to_"
        f"{after_name}_change_map.png"
    )

    Image.fromarray(
        change_mask
    ).save(change_map_path)

    # --------------------------------------------------------
    # Calculate percentage
    # --------------------------------------------------------

    changed_pixels = np.sum(
        change_mask > 0
    )

    total_pixels = change_mask.size

    change_percentage = (
        changed_pixels /
        total_pixels
    ) * 100.0

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------


    if before_path.is_dir() and after_path.is_dir():
        try:
            save_change_visuals(
                before_path,
                after_path,
                change_mask,
                output_dir,
                getattr(cfg, "_current_round", 1),
                before_path.parent.name,
                changed_area,
            )
        except Exception as e:
            print(f"WARNING: Could not create visual overlay: {e}")

    print("\n" + "=" * 70)
    print("CHANGE DETECTION RESULT")
    print("=" * 70)

    print(f"Changed pixels : {changed_pixels}")
    print(f"Total pixels   : {total_pixels}")
    print(
        f"Changed area   : "
        f"{change_percentage:.2f}%"
    )

    print(f"\nChange map saved to:")
    print(change_map_path)

    print("=" * 70)

    # --------------------------------------------------------
    # Save text result
    # --------------------------------------------------------

    result_file = (
        output_dir /
        f"{city_name}_{before_name}_to_"
        f"{after_name}_change_result.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "GeoVision AI - Change Detection\n"
        )
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Before image: {before_path}\n"
        )

        f.write(
            f"After image: {after_path}\n"
        )

        f.write(
            f"Threshold: {threshold}\n"
        )

        f.write(
            f"Changed pixels: "
            f"{changed_pixels}\n"
        )

        f.write(
            f"Total pixels: "
            f"{total_pixels}\n"
        )

        f.write(
            f"Changed area: "
            f"{change_percentage:.2f}%\n"
        )

        f.write(
            f"Change map: "
            f"{change_map_path}\n"
        )

    print(f"Result details saved to:")
    print(result_file)

    return probability_map, change_mask



# ============================================================
# AUTOMATIC DATASET DISCOVERY
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _unique_paths(paths):
    seen = set()
    result = []
    for p in paths:
        p = Path(p)
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen and p.exists():
            seen.add(key)
            result.append(p)
    return result


def find_dataset_roots(project_root):
    """Find likely dataset folders without requiring manual image paths."""
    candidates = [
        project_root / "datasets",
        project_root / "dataset",
        project_root / "data",
        project_root / "inputs",
        project_root / "inputs" / "datasets",
    ]

    # Also use config paths if available.
    try:
        cfg = get_config()
        for value in vars(cfg.paths).values():
            if isinstance(value, (str, Path)):
                candidates.append(Path(value))
    except Exception:
        pass

    return _unique_paths(candidates)


def find_landcover_images(project_root, rounds):
    """
    Select ONE random EuroSAT folder and ONE random image for EACH round.

    Example with --limit 4:
        Round 1 -> random folder + random image
        Round 2 -> another random folder + random image
        Round 3 -> another random folder + random image
        Round 4 -> another random folder + random image

    Folders are selected independently and images are selected
    independently inside the selected folder.
    """
    eurosat_classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]

    eurosat_root = Path(project_root) / "data" / "EuroSAT"

    if not eurosat_root.is_dir():
        possible = [
            Path(project_root) / "datasets" / "EuroSAT",
            Path(project_root) / "dataset" / "EuroSAT",
            Path(project_root) / "inputs" / "EuroSAT",
        ]
        eurosat_root = next(
            (p for p in possible if p.is_dir()),
            eurosat_root
        )

    if not eurosat_root.is_dir():
        print(f"EuroSAT dataset not found: {eurosat_root}")
        return []

    class_folders = [
        eurosat_root / class_name
        for class_name in eurosat_classes
        if (eurosat_root / class_name).is_dir()
    ]

    if not class_folders:
        print(f"No EuroSAT class folders found in: {eurosat_root}")
        return []

    rng = random.SystemRandom()

    print("\n" + "=" * 70)
    print("RANDOM EURO-SAT ROUND SELECTION")
    print("=" * 70)
    print(f"EuroSAT root       : {eurosat_root}")
    print(f"Class folders found: {len(class_folders)}")
    print(f"Number of rounds   : {rounds}")
    print("=" * 70)

    selected = []

    # Avoid repeating a folder until all folders have had a chance to be used.
    # If rounds > 10, a new shuffled cycle is started.
    available_folders = []

    for round_number in range(1, max(1, rounds) + 1):
        if not available_folders:
            available_folders = class_folders.copy()
            rng.shuffle(available_folders)

        selected_folder = available_folders.pop()

        try:
            images = [
                p for p in selected_folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        except (PermissionError, OSError):
            continue

        if not images:
            print(f"Round {round_number}: skipped empty folder {selected_folder.name}")
            continue

        # Random image from THIS round's folder.
        selected_image = rng.choice(images)

        print(f"\nROUND {round_number}")
        print(f"  Random folder : {selected_folder.name}")
        print(f"  Random image  : {selected_image.name}")

        selected.append((round_number, selected_folder, selected_image))

    print("\n" + "=" * 70)
    print("SELECTED ROUNDS")
    print("=" * 70)
    for round_number, folder, image in selected:
        print(
            f"Round {round_number}: "
            f"{folder.name} -> {image.name}"
        )
    print("=" * 70)

    return selected


def _normalized_stem(path):
    stem = path.stem.lower()
    stem = re.sub(r"(_before|_after|before|after|_t1|_t2)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def find_changedet_pairs(project_root, rounds):
    """
    Random OSCD rounds matching the EXACT training input format.

    Each round:
      1. randomly selects an OSCD city
      2. uses imgs_1_rect as T1 scene
      3. uses imgs_2_rect as T2 scene
      4. predict_changedet loads B04/B03/B02 from each scene

    This matches prepare_oscd.py, which trains on RGB composites made
    from B04/B03/B02.
    """
    oscd_root = Path(project_root) / "data" / "OSCD"

    images_root = (
        oscd_root /
        "Onera Satellite Change Detection dataset - Images"
    )

    if not images_root.is_dir():
        print(f"OSCD Images folder not found: {images_root}")
        return []

    cities = [
        p for p in images_root.iterdir()
        if p.is_dir()
        and (p / "imgs_1_rect").is_dir()
        and (p / "imgs_2_rect").is_dir()
        and (p / "imgs_1_rect" / "B04.tif").exists()
        and (p / "imgs_1_rect" / "B03.tif").exists()
        and (p / "imgs_1_rect" / "B02.tif").exists()
        and (p / "imgs_2_rect" / "B04.tif").exists()
        and (p / "imgs_2_rect" / "B03.tif").exists()
        and (p / "imgs_2_rect" / "B02.tif").exists()
    ]

    if not cities:
        print("No OSCD cities with B02/B03/B04 found.")
        return []

    rng = random.SystemRandom()
    available = []
    selected = []

    print("\n" + "=" * 70)
    print("RANDOM OSCD ROUND SELECTION")
    print("=" * 70)
    print(f"OSCD root          : {oscd_root}")
    print(f"Cities available   : {len(cities)}")
    print(f"Number of rounds   : {rounds}")
    print("Input bands        : B04 + B03 + B02 (RGB)")
    print("Normalization      : divide by scene maximum")
    print("=" * 70)

    for round_number in range(1, max(1, rounds) + 1):
        if not available:
            available = cities.copy()
            rng.shuffle(available)

        city = available.pop()

        t1_dir = city / "imgs_1_rect"
        t2_dir = city / "imgs_2_rect"

        print(f"\nROUND {round_number}")
        print(f"  Random city      : {city.name}")
        print("  T1               : imgs_1_rect (B04/B03/B02)")
        print("  T2               : imgs_2_rect (B04/B03/B02)")

        selected.append((round_number, city, t1_dir, t2_dir))

    print("\n" + "=" * 70)
    print("SELECTED OSCD ROUNDS")
    print("=" * 70)
    for n, city, t1, t2 in selected:
        print(f"Round {n}: {city.name}")
    print("=" * 70)

    return selected


def run_auto(cfg, args, checkpoint_dir, output_dir, device):
    """Run GeoVision on dataset images automatically."""
    project_root = Path.cwd()

    print("\n" + "=" * 70)
    print("GEOVISION AI - AUTOMATIC DATASET MODE")
    print("=" * 70)
    print(f"Project root : {project_root}")
    print(f"Sample limit : {args.limit}")
    print("=" * 70)

    # ---------------- LAND COVER ----------------
    if not args.changedet_only:
        print("\nStarting random folder + random image rounds...")
        land_rounds = find_landcover_images(project_root, args.limit)

        if not land_rounds:
            print("No land-cover rounds could be created.")
            print("Put the EuroSAT dataset under data/EuroSAT.")
        else:
            for round_number, selected_folder, image_path in land_rounds:
                print("\n" + "#" * 70)
                print(f"ROUND {round_number}")
                print(f"FOLDER: {selected_folder.name}")
                print(f"IMAGE : {image_path.name}")
                print("#" * 70)

                try:
                    predict_landcover(
                        cfg=cfg,
                        image_path=image_path,
                        checkpoint_path=checkpoint_dir / "landcover_vit_final.pt",
                        output_dir=output_dir,
                        device=device,
                    )
                except Exception as e:
                    print(f"FAILED IN ROUND {round_number}: {e}")

    # ---------------- CHANGE DETECTION ----------------
    if not args.landcover_only:
        print("\nStarting random OSCD city + image-pair rounds...")
        pairs = find_changedet_pairs(project_root, args.limit)

        if not pairs:
            print("No OSCD before/after pairs were found automatically.")
            print("Expected: data/OSCD/Onera Satellite Change Detection dataset - Images/<city>/imgs_1_rect and imgs_2_rect")
        else:
            for round_number, city, before, after in pairs:
                print("\n" + "#" * 70)
                print(f"CHANGE DETECTION ROUND {round_number}")
                print(f"CITY  : {city.name}")
                print("BEFORE: imgs_1_rect (B04/B03/B02)")
                print("AFTER : imgs_2_rect (B04/B03/B02)")
                print("#" * 70)

                try:
                    cfg._current_round = round_number

                    predict_changedet(
                        cfg=cfg,
                        before_path=before,
                        after_path=after,
                        checkpoint_path=checkpoint_dir / "changedet_epoch05.pt",
                        output_dir=output_dir,
                        device=device,
                    )
                except Exception as e:
                    print(f"FAILED IN CHANGE-DETECTION ROUND {round_number}: {e}")



# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GeoVision AI - Manual or Automatic Dataset Prediction"
    )

    # --------------------------------------------------------
    # Land cover
    # --------------------------------------------------------

    parser.add_argument(
        "--landcover",
        type=str,
        help="Path to one image for land-cover classification"
    )

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------

    parser.add_argument(
        "--changedet",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Two images for change detection"
    )

    # --------------------------------------------------------
    # Optional checkpoint paths
    # --------------------------------------------------------

    parser.add_argument(
        "--landcover-checkpoint",
        type=str,
        default=None,
        help="Land-cover checkpoint path"
    )

    parser.add_argument(
        "--changedet-checkpoint",
        type=str,
        default=None,
        help="Change-detection checkpoint path"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Randomly select one dataset folder and run predictions"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of random rounds in automatic mode (land-cover and OSCD)"
    )

    parser.add_argument(
        "--landcover-only",
        action="store_true",
        help="In --auto mode, run only land-cover prediction"
    )

    parser.add_argument(
        "--changedet-only",
        action="store_true",
        help="In --auto mode, run only change detection"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Require one mode unless automatic dataset mode is selected
    # --------------------------------------------------------

    if args.auto and (args.landcover is not None or args.changedet is not None):
        parser.error("--auto cannot be combined with --landcover or --changedet")

    if args.auto and args.landcover_only and args.changedet_only:
        parser.error("Choose only one of --landcover-only or --changedet-only")

    if not args.auto and args.landcover is None and args.changedet is None:
        parser.error("Choose --auto, --landcover, or --changedet")

    if not args.auto and args.landcover is not None and args.changedet is not None:
        parser.error("Use only one prediction mode at a time")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    cfg = get_config()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    cfg.device = device

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = setup_output_directory(cfg)

    # --------------------------------------------------------
    # Default checkpoints
    # --------------------------------------------------------

    checkpoint_dir = Path(
        cfg.paths.output_dir
    ) / "checkpoints"

    if args.landcover_checkpoint:

        landcover_checkpoint = Path(
            args.landcover_checkpoint
        )

    else:

        landcover_checkpoint = (
            checkpoint_dir /
            "landcover_vit_final.pt"
        )

    if args.changedet_checkpoint:

        changedet_checkpoint = Path(
            args.changedet_checkpoint
        )

    else:

        changedet_checkpoint = (
            checkpoint_dir /
            "changedet_epoch05.pt"
        )

    # --------------------------------------------------------
    # Automatic dataset mode
    # --------------------------------------------------------

    if args.auto:
        run_auto(
            cfg=cfg,
            args=args,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            device=device,
        )
        return

    # --------------------------------------------------------
    # Land cover
    # --------------------------------------------------------

    if args.landcover:

        predict_landcover(
            cfg=cfg,
            image_path=args.landcover,
            checkpoint_path=landcover_checkpoint,
            output_dir=output_dir,
            device=device
        )

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------

    elif args.changedet:

        before_path, after_path = args.changedet

        predict_changedet(
            cfg=cfg,
            before_path=before_path,
            after_path=after_path,
            checkpoint_path=changedet_checkpoint,
            output_dir=output_dir,
            device=device
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()