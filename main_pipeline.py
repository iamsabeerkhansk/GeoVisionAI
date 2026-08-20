"""
GeoVision AI - Main Pipeline
Multi-Temporal Satellite Image Change Detection and
Land Cover Classification Using Vision Transformers.

Datasets:
    - EuroSAT  -> Land cover classification
    - OSCD     -> Bi-temporal change detection

Backbone:
    - Vision Transformer (ViT-B/16)

Features:
    - Automatic NVIDIA CUDA / CPU detection
    - GPU information display
    - CPU fallback
    - Quick-test mode
    - Stage-wise execution
    - Checkpoint directory creation
    - Results saving

Usage:
    python main_pipeline.py
    python main_pipeline.py --stage landcover
    python main_pipeline.py --stage changedet
    python main_pipeline.py --quick-test
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from geovision.config import get_config
from geovision.data.prepare_eurosat import prepare_eurosat
from geovision.data.prepare_oscd import prepare_oscd

from geovision.models.vit_landcover import build_landcover_vit
from geovision.models.siamese_vit_changedet import (
    build_change_detection_vit
)

from geovision.train_landcover import train_landcover_model
from geovision.train_changedet import (
    train_change_detection_model
)

from geovision.utils.metrics import (
    evaluate_landcover,
    evaluate_change_detection
)

from geovision.utils.visualize import (
    visualize_change_detection,
    plot_confusion_matrix
)


# ============================================================
# DEVICE DETECTION
# ============================================================

def setup_device(cfg):
    """
    Automatically select NVIDIA CUDA GPU if available.
    Otherwise use CPU.
    """

    print("\n" + "=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA available : YES")
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA version   : "
            f"{torch.version.cuda}"
        )

        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)

            print(
                f"GPU memory     : "
                f"{gpu_memory_gb:.2f} GB"
            )

        except Exception:
            pass

        print("Selected device : CUDA GPU")

    else:

        device = torch.device("cpu")

        print("CUDA available : NO")
        print("Selected device : CPU")

        print(
            "\nWARNING:"
            "\nNVIDIA CUDA GPU was not detected."
            "\nTraining will run on CPU."
        )

    print("=" * 70)

    # Add device to configuration object
    cfg.device = device

    return device


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

def setup_output_directories(cfg):
    """
    Create output and checkpoint directories.
    """

    output_dir = Path(cfg.paths.output_dir)

    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Store checkpoint path in configuration
    cfg.paths.checkpoint_dir = checkpoint_dir

    print("\nOutput directories:")
    print(f"Output      : {output_dir}")
    print(f"Checkpoints : {checkpoint_dir}")
    print(f"Figures     : {figure_dir}")

    return output_dir, checkpoint_dir


# ============================================================
# STAGE 1
# LAND COVER CLASSIFICATION
# ============================================================

def run_landcover(cfg):

    print("\n" + "=" * 70)
    print(
        "STAGE 1: Land Cover Classification "
        "(EuroSAT + ViT transfer learning)"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nPreparing EuroSAT dataset...")

    train_ds, val_ds, test_ds = prepare_eurosat(cfg)

    print("\nEuroSAT dataset ready.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding Land Cover ViT model...")

    net = build_landcover_vit(cfg)

    # Move model to selected device
    device = cfg.device

    net = net.to(device)

    print(f"Model device: {device}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting land-cover training...")

    net, history = train_landcover_model(
        net,
        train_ds,
        val_ds,
        cfg
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating land-cover model...")

    results = evaluate_landcover(
        net,
        test_ds,
        cfg
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    confusion_path = (
        Path(cfg.paths.output_dir)
        / "figures"
        / "landcover_confusion.png"
    )

    plot_confusion_matrix(
        results["confusion_matrix"],
        cfg.landcover.classes,
        confusion_path
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{confusion_path}"
    )

    return net, history, results


# ============================================================
# STAGE 2
# CHANGE DETECTION
# ============================================================

def run_changedet(cfg):

    print("\n" + "=" * 70)
    print(
        "STAGE 2: Bi-Temporal Change Detection "
        "(OSCD + Siamese ViT)"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nPreparing OSCD dataset...")

    train_ds, val_ds, test_ds = prepare_oscd(cfg)

    print("\nOSCD dataset ready.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding Siamese Change Detection ViT...")

    net = build_change_detection_vit(cfg)

    # Move model to selected device
    device = cfg.device

    net = net.to(device)

    print(f"Model device: {device}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting change-detection training...")

    net, history = train_change_detection_model(
        net,
        train_ds,
        val_ds,
        cfg
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating change-detection model...")

    test_stats = evaluate_change_detection(
        net,
        test_ds,
        cfg
    )

    print(
        "\n=== Change Detection Results (test) ==="
    )

    print(
        f"IoU: {test_stats['iou']:.3f} | "
        f"F1: {test_stats['f1']:.3f} | "
        f"Precision: {test_stats['precision']:.3f} | "
        f"Recall: {test_stats['recall']:.3f} | "
        f"OA: "
        f"{test_stats['overall_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    print("\nGenerating change-detection visualizations...")

    visualize_change_detection(
        net,
        test_ds,
        cfg
    )

    return net, history, test_stats


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(all_results, cfg):

    output_dir = Path(cfg.paths.output_dir)

    out_path = (
        output_dir /
        "geovision_results.pkl"
    )

    with open(
        out_path,
        "wb"
    ) as f:

        pickle.dump(
            all_results,
            f
        )

    print(
        f"\nResults saved to:\n"
        f"{out_path}"
    )


# ============================================================
# GPU MEMORY CLEANUP
# ============================================================

def clear_gpu_memory():

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        print(
            "\nCUDA cache cleared."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GeoVision AI pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "landcover",
            "changedet"
        ],
        default="all",
        help=(
            "Select pipeline stage"
        )
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help=(
            "Run one epoch for each selected stage "
            "to sanity-check the pipeline."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    cfg = get_config()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    setup_device(cfg)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    setup_output_directories(cfg)

    # --------------------------------------------------------
    # Quick test
    # --------------------------------------------------------

    if args.quick_test:

        cfg.landcover.max_epochs = 1
        cfg.changedet.max_epochs = 1

        print(
            "\n[quick-test]"
            " Running with max_epochs=1."
        )

    # --------------------------------------------------------
    # Results container
    # --------------------------------------------------------

    all_results = {}

    # ========================================================
    # STAGE 1
    # ========================================================

    if args.stage in (
        "all",
        "landcover"
    ):

        lc_net, lc_history, lc_results = (
            run_landcover(cfg)
        )

        all_results["landcover"] = {
            "history": lc_history,
            "results": lc_results
        }

        # Free memory before Stage 2
        del lc_net

        clear_gpu_memory()

    # ========================================================
    # STAGE 2
    # ========================================================

    if args.stage in (
        "all",
        "changedet"
    ):

        cd_net, cd_history, cd_results = (
            run_changedet(cfg)
        )

        all_results["changedet"] = {
            "history": cd_history,
            "results": cd_results
        }

        del cd_net

        clear_gpu_memory()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        all_results,
        cfg
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GEOVISION AI PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nOutputs:\n"
        f"{cfg.paths.output_dir}"
    )

    print(
        "\nTraining device:"
        f" {cfg.device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU used:"
            f" {torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()