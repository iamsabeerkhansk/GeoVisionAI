"""
============================================================
GEOVISION AI - STREAMLIT INTERFACE
============================================================

Modes:
    1. Dashboard
    2. Manual Land Cover
    3. Manual Change Detection
    4. Automatic Dataset Mode

Run:
    python -m streamlit run app.py
============================================================
"""

from pathlib import Path
import sys
import random

import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# GEOVISION IMPORTS
# ============================================================

try:

    from geovision.config import get_config

    from geovision.models.vit_landcover import (
        build_landcover_vit
    )

    from geovision.models.siamese_vit_changedet import (
        build_change_detection_vit
    )

    from predict import (
        load_checkpoint,
        load_change_image,
        find_landcover_images,
        find_changedet_pairs,
    )

    IMPORT_OK = True

except Exception as e:

    IMPORT_OK = False
    IMPORT_ERROR = e


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GeoVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 750;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        min-height: 125px;
    }

    .info-title {
        font-size: 15px;
        font-weight: 600;
    }

    .info-value {
        font-size: 25px;
        font-weight: 800;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ GeoVision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Satellite Image Analysis Platform'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CHECK IMPORTS
# ============================================================

if not IMPORT_OK:

    st.error(
        "GeoVision AI modules could not be imported."
    )

    st.exception(IMPORT_ERROR)

    st.stop()


# ============================================================
# INITIALIZE
# ============================================================

try:

    cfg = get_config()

    if torch.cuda.is_available():

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(0)

    else:

        device = torch.device("cpu")

        gpu_name = "CPU"

    cfg.device = device

    # Output directory
    output_dir = (
        Path(cfg.paths.output_dir)
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Checkpoints
    checkpoint_dir = (
        Path(cfg.paths.output_dir)
        / "checkpoints"
    )

    landcover_checkpoint = (
        checkpoint_dir
        / "landcover_vit_final.pt"
    )

    changedet_checkpoint = (
        checkpoint_dir
        / "changedet_epoch05.pt"
    )

except Exception as e:

    st.error(
        "GeoVision AI initialization failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ GeoVision AI")

    st.markdown("---")

    st.subheader("System")

    if device.type == "cuda":

        st.success("🟢 CUDA GPU")

        st.write(
            f"GPU: `{gpu_name}`"
        )

    else:

        st.warning("🟡 CPU")

    st.write(
        f"PyTorch: `{torch.__version__}`"
    )

    st.markdown("---")

    st.subheader("Checkpoints")

    if landcover_checkpoint.exists():

        st.success(
            "✅ Land Cover"
        )

    else:

        st.error(
            "❌ Land Cover missing"
        )

    if changedet_checkpoint.exists():

        st.success(
            "✅ Change Detection"
        )

    else:

        st.error(
            "❌ Change Detection missing"
        )

    st.markdown("---")

    st.subheader("Output")

    st.code(
        str(output_dir)
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_landcover, tab_changedet, tab_auto = st.tabs(
    [
        "🏠 Dashboard",
        "🛰️ Land Cover",
        "🔄 Change Detection",
        "🤖 AUTO MODE",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.markdown(
        '<div class="section-title">'
        '📊 Project Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        **GeoVision AI** is an AI-powered satellite image
        analysis system with two major capabilities.

        ### 🛰️ Land Cover Classification

        A Vision Transformer analyzes a satellite image
        and predicts its land-cover category.

        ### 🔄 Change Detection

        A Siamese Vision Transformer compares satellite
        images from two different times and produces a
        change probability map.
        """
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    🛰️ LAND COVER
                </div>
                <div class="info-value">
                    ViT
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    🔄 CHANGE DETECTION
                </div>
                <div class="info-value">
                    Siamese ViT
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">
                    ⚡ DEVICE
                </div>
                <div class="info-value">
                    {device.type.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        result_count = len(
            list(output_dir.glob("*"))
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">
                    📁 RESULTS
                </div>
                <div class="info-value">
                    {result_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader(
        "📂 Project Structure"
    )

    st.code(
        f"""
GeoVisionAI/
│
├── app.py
├── predict.py
│
├── geovision/
│   ├── config.py
│   └── models/
│       ├── vit_landcover.py
│       └── siamese_vit_changedet.py
│
├── data/
│   ├── EuroSAT/
│   └── OSCD/
│
└── outputs/
    ├── checkpoints/
    │   ├── landcover_vit_final.pt
    │   └── changedet_epoch05.pt
    │
    └── predictions/
        """
    )

    st.info(
        "Use AUTO MODE to test your trained models "
        "automatically without manually selecting images."
    )


# ============================================================
# MANUAL LAND COVER
# ============================================================

with tab_landcover:

    st.header(
        "🛰️ Land Cover Classification"
    )

    st.write(
        "Upload one satellite image and classify it."
    )

    uploaded = st.file_uploader(
        "Choose satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="manual_landcover",
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                use_container_width=True,
            )

        with c2:

            st.subheader(
                "Image Information"
            )

            st.write(
                f"Filename: `{uploaded.name}`"
            )

            st.write(
                f"Size: `{image.width} × {image.height}`"
            )

            st.write(
                "Format: RGB"
            )

        st.markdown("---")

        if st.button(
            "🚀 PREDICT LAND COVER",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Running Land Cover model..."
            ):

                try:

                    transform = transforms.Compose(
                        [
                            transforms.Resize(
                                cfg.vit.input_size[:2]
                            ),

                            transforms.ToTensor(),

                            transforms.Normalize(
                                mean=[
                                    0.485,
                                    0.456,
                                    0.406,
                                ],

                                std=[
                                    0.229,
                                    0.224,
                                    0.225,
                                ],
                            ),
                        ]
                    )

                    tensor = (
                        transform(image)
                        .unsqueeze(0)
                        .to(device)
                    )

                    model = (
                        build_landcover_vit(
                            cfg
                        )
                    )

                    model = load_checkpoint(
                        model,
                        landcover_checkpoint,
                        device,
                    )

                    with torch.no_grad():

                        logits = model(
                            tensor
                        )

                        probabilities = (
                            torch.softmax(
                                logits,
                                dim=1,
                            )
                        )

                        confidence, prediction = (
                            torch.max(
                                probabilities,
                                dim=1,
                            )
                        )

                    index = prediction.item()

                    confidence = (
                        confidence.item()
                    )

                    predicted_class = (
                        cfg.landcover.classes[
                            index
                        ]
                    )

                    st.success(
                        "Prediction completed!"
                    )

                    r1, r2 = st.columns(2)

                    with r1:

                        st.metric(
                            "Predicted Class",
                            predicted_class,
                        )

                    with r2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%",
                        )

                    # Probabilities
                    st.subheader(
                        "📊 Class Probabilities"
                    )

                    probs = (
                        probabilities[
                            0
                        ]
                        .cpu()
                        .numpy()
                    )

                    probability_dict = {}

                    for i, name in enumerate(
                        cfg.landcover.classes
                    ):

                        probability_dict[
                            name
                        ] = float(
                            probs[i] * 100
                        )

                    st.bar_chart(
                        probability_dict
                    )

                    # Visual result
                    visual = image.copy()

                    draw = ImageDraw.Draw(
                        visual
                    )

                    label = (
                        f"GeoVision: "
                        f"{predicted_class} "
                        f"({confidence * 100:.1f}%)"
                    )

                    try:

                        font = (
                            ImageFont.truetype(
                                "arial.ttf",
                                max(
                                    18,
                                    image.width // 25,
                                ),
                            )
                        )

                    except Exception:

                        font = (
                            ImageFont.load_default()
                        )

                    bbox = draw.textbbox(
                        (0, 0),
                        label,
                        font=font,
                    )

                    draw.rectangle(
                        (
                            10,
                            10,
                            bbox[2] + 30,
                            bbox[3] + 30,
                        ),
                        fill="black",
                    )

                    draw.text(
                        (20, 20),
                        label,
                        fill="white",
                        font=font,
                    )

                    result_path = (
                        output_dir
                        / (
                            Path(
                                uploaded.name
                            ).stem
                            + "_landcover_GUI.png"
                        )
                    )

                    visual.save(
                        result_path
                    )

                    st.subheader(
                        "🖼️ Prediction"
                    )

                    st.image(
                        visual,
                        use_container_width=True,
                    )

                    st.success(
                        f"Saved: {result_path}"
                    )

                except Exception as e:

                    st.error(
                        "Land Cover prediction failed."
                    )

                    st.exception(e)


# ============================================================
# MANUAL CHANGE DETECTION
# ============================================================

with tab_changedet:

    st.header(
        "🔄 Change Detection"
    )

    st.write(
        "Compare a BEFORE image with an AFTER image."
    )

    before_file = st.file_uploader(
        "📷 BEFORE image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="manual_before",
    )

    after_file = st.file_uploader(
        "📷 AFTER image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="manual_after",
    )

    threshold = st.slider(
        "🎚️ Change Detection Threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.10,
        step=0.05,
    )

    st.write(
        f"Threshold: **{threshold:.2f}**"
    )

    if before_file and after_file:

        before_image = Image.open(
            before_file
        ).convert("RGB")

        after_image = Image.open(
            after_file
        ).convert("RGB")

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("BEFORE")

            st.image(
                before_image,
                use_container_width=True,
            )

        with c2:

            st.subheader("AFTER")

            st.image(
                after_image,
                use_container_width=True,
            )

        if st.button(
            "🔍 DETECT CHANGES",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Running Change Detection..."
            ):

                try:

                    temp_dir = (
                        PROJECT_ROOT
                        / ".streamlit_temp"
                    )

                    temp_dir.mkdir(
                        exist_ok=True
                    )

                    before_path = (
                        temp_dir
                        / "manual_before.png"
                    )

                    after_path = (
                        temp_dir
                        / "manual_after.png"
                    )

                    before_image.save(
                        before_path
                    )

                    after_image.save(
                        after_path
                    )

                    before_tensor = (
                        load_change_image(
                            before_path,
                            cfg,
                        )
                    )

                    after_tensor = (
                        load_change_image(
                            after_path,
                            cfg,
                        )
                    )

                    before_tensor = (
                        before_tensor.to(
                            device
                        )
                    )

                    after_tensor = (
                        after_tensor.to(
                            device
                        )
                    )

                    model = (
                        build_change_detection_vit(
                            cfg
                        )
                    )

                    model = load_checkpoint(
                        model,
                        changedet_checkpoint,
                        device,
                    )

                    with torch.no_grad():

                        probability_map = model(
                            before_tensor,
                            after_tensor,
                        )

                    probability_map = (
                        probability_map[
                            0,
                            0,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    change_mask = (
                        probability_map
                        >= threshold
                    )

                    changed_pixels = int(
                        change_mask.sum()
                    )

                    total_pixels = int(
                        change_mask.size
                    )

                    change_percentage = (
                        changed_pixels
                        / total_pixels
                        * 100
                    )

                    # Resize mask
                    mask = Image.fromarray(
                        (
                            change_mask.astype(
                                np.uint8
                            )
                            * 255
                        )
                    )

                    mask = mask.resize(
                        after_image.size,
                        Image.Resampling.NEAREST,
                    )

                    full_mask = (
                        np.asarray(mask)
                        > 0
                    )

                    # Red overlay
                    after_np = np.asarray(
                        after_image
                    ).astype(
                        np.float32
                    )

                    overlay = (
                        after_np.copy()
                    )

                    overlay[full_mask] = (
                        0.35
                        * overlay[full_mask]
                        + 0.65
                        * np.array(
                            [
                                255,
                                0,
                                0,
                            ]
                        )
                    )

                    overlay = np.clip(
                        overlay,
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )

                    overlay_image = Image.fromarray(
                        overlay
                    )

                    # Save
                    overlay_path = (
                        output_dir
                        / "manual_change_overlay.png"
                    )

                    mask_path = (
                        output_dir
                        / "manual_change_mask.png"
                    )

                    overlay_image.save(
                        overlay_path
                    )

                    mask.save(
                        mask_path
                    )

                    # Results
                    st.success(
                        "Change detection completed!"
                    )

                    r1, r2, r3 = st.columns(3)

                    with r1:

                        st.metric(
                            "Changed Area",
                            f"{change_percentage:.2f}%",
                        )

                    with r2:

                        st.metric(
                            "Threshold",
                            f"{threshold:.2f}",
                        )

                    with r3:

                        st.metric(
                            "Changed Pixels",
                            f"{changed_pixels:,}",
                        )

                    st.subheader(
                        "🔴 Change Result"
                    )

                    a, b, c = st.columns(3)

                    with a:

                        st.write("BEFORE")

                        st.image(
                            before_image,
                            use_container_width=True,
                        )

                    with b:

                        st.write("AFTER")

                        st.image(
                            after_image,
                            use_container_width=True,
                        )

                    with c:

                        st.write(
                            "DETECTED CHANGES"
                        )

                        st.image(
                            overlay_image,
                            use_container_width=True,
                        )

                    st.subheader(
                        "🗺️ Binary Change Mask"
                    )

                    st.image(
                        mask,
                        use_container_width=True,
                    )

                except Exception as e:

                    st.error(
                        "Change Detection failed."
                    )

                    st.exception(e)

    else:

        st.info(
            "Upload both BEFORE and AFTER images."
        )


# ============================================================
# AUTO MODE
# ============================================================

with tab_auto:

    st.header(
        "🤖 Automatic Dataset Mode"
    )

    st.write(
        """
        Automatically select satellite images from your
        datasets and run GeoVision AI without manually
        uploading images.
        """
    )

    st.markdown("---")

    # ==========================================
    # AUTO SETTINGS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        rounds = st.number_input(
            "🔢 Number of rounds",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )

    with col2:

        auto_mode = st.selectbox(
            "🎯 Prediction type",
            [
                "Both",
                "Land Cover Only",
                "Change Detection Only",
            ],
        )

    auto_threshold = st.slider(
        "🎚️ Change Detection Threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.10,
        step=0.05,
        key="auto_threshold",
    )

    st.write(
        f"Automatic threshold: "
        f"**{auto_threshold:.2f}**"
    )

    st.markdown("---")

    st.subheader(
        "📂 Expected Dataset Structure"
    )

    st.code(
        """
data/
│
├── EuroSAT/
│   ├── AnnualCrop/
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   ├── Highway/
│   ├── Industrial/
│   ├── Pasture/
│   ├── PermanentCrop/
│   ├── Residential/
│   ├── River/
│   └── SeaLake/
│
└── OSCD/
    └── Onera Satellite Change Detection dataset - Images/
        ├── city1/
        │   ├── imgs_1_rect/
        │   │   ├── B02.tif
        │   │   ├── B03.tif
        │   │   └── B04.tif
        │   │
        │   └── imgs_2_rect/
        │       ├── B02.tif
        │       ├── B03.tif
        │       └── B04.tif
        """
    )

    st.markdown("---")

    # ==========================================
    # RUN AUTO
    # ==========================================

    if st.button(
        "🚀 RUN AUTOMATIC MODE",
        type="primary",
        use_container_width=True,
    ):

        # ======================================
        # SELECT LAND COVER ROUNDS
        # ======================================

        land_rounds = []

        if auto_mode in [
            "Both",
            "Land Cover Only",
        ]:

            with st.spinner(
                "Selecting random EuroSAT images..."
            ):

                try:

                    land_rounds = (
                        find_landcover_images(
                            PROJECT_ROOT,
                            int(rounds),
                        )
                    )

                except Exception as e:

                    st.error(
                        "Could not find EuroSAT dataset."
                    )

                    st.exception(e)

        # ======================================
        # SELECT CHANGE DETECTION ROUNDS
        # ======================================

        change_rounds = []

        if auto_mode in [
            "Both",
            "Change Detection Only",
        ]:

            with st.spinner(
                "Selecting random OSCD cities..."
            ):

                try:

                    change_rounds = (
                        find_changedet_pairs(
                            PROJECT_ROOT,
                            int(rounds),
                        )
                    )

                except Exception as e:

                    st.error(
                        "Could not find OSCD dataset."
                    )

                    st.exception(e)

        # ======================================
        # SUMMARY
        # ======================================

        st.markdown("---")

        st.subheader(
            "📋 Automatic Run Summary"
        )

        s1, s2 = st.columns(2)

        with s1:

            st.metric(
                "Land Cover Rounds",
                len(land_rounds),
            )

        with s2:

            st.metric(
                "Change Detection Rounds",
                len(change_rounds),
            )

        # ======================================
        # LAND COVER AUTO
        # ======================================

        if land_rounds:

            st.markdown("---")

            st.header(
                "🛰️ Automatic Land Cover Results"
            )

            # Load model once
            try:

                land_model = (
                    build_landcover_vit(
                        cfg
                    )
                )

                land_model = load_checkpoint(
                    land_model,
                    landcover_checkpoint,
                    device,
                )

                land_model.eval()

            except Exception as e:

                st.error(
                    "Could not load Land Cover model."
                )

                st.exception(e)

                land_model = None

            for (
                round_number,
                selected_folder,
                image_path,
            ) in land_rounds:

                st.markdown("---")

                st.subheader(
                    f"ROUND {round_number}"
                )

                st.write(
                    f"📁 Folder: "
                    f"**{selected_folder.name}**"
                )

                st.write(
                    f"🖼️ Image: "
                    f"**{image_path.name}**"
                )

                if land_model is None:
                    continue

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    transform = transforms.Compose(
                        [
                            transforms.Resize(
                                cfg.vit.input_size[:2]
                            ),

                            transforms.ToTensor(),

                            transforms.Normalize(
                                mean=[
                                    0.485,
                                    0.456,
                                    0.406,
                                ],

                                std=[
                                    0.229,
                                    0.224,
                                    0.225,
                                ],
                            ),
                        ]
                    )

                    tensor = (
                        transform(image)
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():

                        logits = land_model(
                            tensor
                        )

                        probabilities = (
                            torch.softmax(
                                logits,
                                dim=1,
                            )
                        )

                        confidence, prediction = (
                            torch.max(
                                probabilities,
                                dim=1,
                            )
                        )

                    index = prediction.item()

                    confidence = (
                        confidence.item()
                    )

                    predicted_class = (
                        cfg.landcover.classes[
                            index
                        ]
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.image(
                            image,
                            caption=(
                                "Original Satellite Image"
                            ),
                            use_container_width=True,
                        )

                    with c2:

                        st.success(
                            f"Prediction: "
                            f"**{predicted_class}**"
                        )

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%",
                        )

                        probs = (
                            probabilities[
                                0
                            ]
                            .cpu()
                            .numpy()
                        )

                        probability_dict = {}

                        for i, name in enumerate(
                            cfg.landcover.classes
                        ):

                            probability_dict[
                                name
                            ] = float(
                                probs[i] * 100
                            )

                        st.bar_chart(
                            probability_dict
                        )

                    # Save result
                    result_path = (
                        output_dir
                        / (
                            f"auto_round"
                            f"{round_number:02d}_"
                            f"{image_path.stem}_"
                            f"landcover.png"
                        )
                    )

                    visual = image.copy()

                    draw = ImageDraw.Draw(
                        visual
                    )

                    label = (
                        f"{predicted_class} "
                        f"({confidence * 100:.1f}%)"
                    )

                    draw.rectangle(
                        (
                            10,
                            10,
                            420,
                            65,
                        ),
                        fill="black",
                    )

                    draw.text(
                        (
                            20,
                            20,
                        ),
                        label,
                        fill="white",
                    )

                    visual.save(
                        result_path
                    )

                except Exception as e:

                    st.error(
                        f"Round {round_number} failed."
                    )

                    st.exception(e)

        # ======================================
        # CHANGE DETECTION AUTO
        # ======================================

        if change_rounds:

            st.markdown("---")

            st.header(
                "🔄 Automatic Change Detection Results"
            )

            # Load model once
            try:

                change_model = (
                    build_change_detection_vit(
                        cfg
                    )
                )

                change_model = load_checkpoint(
                    change_model,
                    changedet_checkpoint,
                    device,
                )

                change_model.eval()

            except Exception as e:

                st.error(
                    "Could not load Change Detection model."
                )

                st.exception(e)

                change_model = None

            for (
                round_number,
                city,
                before_dir,
                after_dir,
            ) in change_rounds:

                st.markdown("---")

                st.subheader(
                    f"CHANGE DETECTION ROUND "
                    f"{round_number}"
                )

                st.write(
                    f"🌍 City: **{city.name}**"
                )

                st.write(
                    "BEFORE: `imgs_1_rect`"
                )

                st.write(
                    "AFTER: `imgs_2_rect`"
                )

                if change_model is None:
                    continue

                try:

                    # ----------------------------------
                    # LOAD T1/T2
                    # ----------------------------------

                    before_tensor = (
                        load_change_image(
                            before_dir,
                            cfg,
                        )
                    )

                    after_tensor = (
                        load_change_image(
                            after_dir,
                            cfg,
                        )
                    )

                    before_tensor = (
                        before_tensor.to(
                            device
                        )
                    )

                    after_tensor = (
                        after_tensor.to(
                            device
                        )
                    )

                    # ----------------------------------
                    # PREDICT
                    # ----------------------------------

                    with torch.no_grad():

                        probability_map = (
                            change_model(
                                before_tensor,
                                after_tensor,
                            )
                        )

                    probability_map = (
                        probability_map[
                            0,
                            0,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # ----------------------------------
                    # THRESHOLD
                    # ----------------------------------

                    change_mask = (
                        probability_map
                        >= auto_threshold
                    )

                    changed_pixels = int(
                        change_mask.sum()
                    )

                    total_pixels = int(
                        change_mask.size
                    )

                    change_percentage = (
                        changed_pixels
                        / total_pixels
                        * 100
                    )

                    # ----------------------------------
                    # LOAD ORIGINAL RGB
                    # ----------------------------------

                    def read_band(
                        scene_dir,
                        name,
                    ):

                        tif_path = (
                            Path(scene_dir)
                            / name
                        )

                        try:

                            import tifffile

                            return tifffile.imread(
                                str(tif_path)
                            ).astype(
                                np.float32
                            )

                        except Exception:

                            return np.asarray(
                                Image.open(
                                    tif_path
                                )
                            ).astype(
                                np.float32
                            )

                    def scene_rgb(
                        scene_dir
                    ):

                        r = read_band(
                            scene_dir,
                            "B04.tif",
                        )

                        g = read_band(
                            scene_dir,
                            "B03.tif",
                        )

                        b = read_band(
                            scene_dir,
                            "B02.tif",
                        )

                        rgb = np.stack(
                            [
                                r,
                                g,
                                b,
                            ],
                            axis=-1,
                        )

                        result = np.zeros_like(
                            rgb,
                            dtype=np.float32,
                        )

                        for channel in range(3):

                            lo = np.percentile(
                                rgb[..., channel],
                                2,
                            )

                            hi = np.percentile(
                                rgb[..., channel],
                                98,
                            )

                            if hi > lo:

                                result[
                                    ...,
                                    channel,
                                ] = np.clip(
                                    (
                                        rgb[
                                            ...,
                                            channel,
                                        ]
                                        - lo
                                    )
                                    / (
                                        hi - lo
                                    ),
                                    0,
                                    1,
                                )

                        return (
                            result * 255
                        ).astype(
                            np.uint8
                        )

                    before_rgb = scene_rgb(
                        before_dir
                    )

                    after_rgb = scene_rgb(
                        after_dir
                    )

                    # ----------------------------------
                    # RESIZE MASK
                    # ----------------------------------

                    mask_small = Image.fromarray(
                        (
                            change_mask.astype(
                                np.uint8
                            )
                            * 255
                        )
                    )

                    mask_full = (
                        mask_small.resize(
                            (
                                after_rgb.shape[1],
                                after_rgb.shape[0],
                            ),
                            Image.Resampling.NEAREST,
                        )
                    )

                    full_mask = (
                        np.asarray(
                            mask_full
                        )
                        > 0
                    )

                    # ----------------------------------
                    # RED OVERLAY
                    # ----------------------------------

                    overlay = (
                        after_rgb.astype(
                            np.float32
                        )
                        .copy()
                    )

                    overlay[full_mask] = (
                        0.35
                        * overlay[full_mask]
                        + 0.65
                        * np.array(
                            [
                                255,
                                0,
                                0,
                            ],
                            dtype=np.float32,
                        )
                    )

                    overlay = np.clip(
                        overlay,
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )

                    overlay_image = (
                        Image.fromarray(
                            overlay
                        )
                    )

                    # ----------------------------------
                    # DISPLAY
                    # ----------------------------------

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.write(
                            "BEFORE"
                        )

                        st.image(
                            before_rgb,
                            use_container_width=True,
                        )

                    with c2:

                        st.write(
                            "AFTER"
                        )

                        st.image(
                            after_rgb,
                            use_container_width=True,
                        )

                    with c3:

                        st.write(
                            "🔴 DETECTED CHANGES"
                        )

                        st.image(
                            overlay_image,
                            use_container_width=True,
                        )

                    m1, m2, m3 = st.columns(3)

                    with m1:

                        st.metric(
                            "Changed Area",
                            f"{change_percentage:.2f}%",
                        )

                    with m2:

                        st.metric(
                            "Threshold",
                            f"{auto_threshold:.2f}",
                        )

                    with m3:

                        st.metric(
                            "Changed Pixels",
                            f"{changed_pixels:,}",
                        )

                    # ----------------------------------
                    # PROBABILITY STATS
                    # ----------------------------------

                    with st.expander(
                        "📊 Probability Statistics"
                    ):

                        st.write(
                            f"Minimum: "
                            f"`{probability_map.min():.6f}`"
                        )

                        st.write(
                            f"Maximum: "
                            f"`{probability_map.max():.6f}`"
                        )

                        st.write(
                            f"Mean: "
                            f"`{probability_map.mean():.6f}`"
                        )

                        st.write(
                            f">= 0.10: "
                            f"{np.mean(probability_map >= 0.10) * 100:.2f}%"
                        )

                        st.write(
                            f">= 0.20: "
                            f"{np.mean(probability_map >= 0.20) * 100:.2f}%"
                        )

                        st.write(
                            f">= 0.30: "
                            f"{np.mean(probability_map >= 0.30) * 100:.2f}%"
                        )

                        st.write(
                            f">= 0.50: "
                            f"{np.mean(probability_map >= 0.50) * 100:.2f}%"
                        )

                    # ----------------------------------
                    # SAVE
                    # ----------------------------------

                    overlay_path = (
                        output_dir
                        / (
                            f"auto_round"
                            f"{round_number:02d}_"
                            f"{city.name}_"
                            f"change_overlay.png"
                        )
                    )

                    mask_path = (
                        output_dir
                        / (
                            f"auto_round"
                            f"{round_number:02d}_"
                            f"{city.name}_"
                            f"change_mask.png"
                        )
                    )

                    overlay_image.save(
                        overlay_path
                    )

                    mask_full.save(
                        mask_path
                    )

                    st.success(
                        f"Saved results for "
                        f"{city.name}"
                    )

                except Exception as e:

                    st.error(
                        f"Change Detection Round "
                        f"{round_number} failed."
                    )

                    st.exception(e)

        # ======================================
        # FINISHED
        # ======================================

        st.markdown("---")

        st.success(
            "🤖 AUTOMATIC MODE COMPLETED"
        )

        st.write(
            "All generated results are saved in:"
        )

        st.code(
            str(output_dir)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "GeoVision AI | Vision Transformer | "
    "Siamese Vision Transformer | Satellite Analysis"
)
