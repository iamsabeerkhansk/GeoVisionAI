"""
============================================================
GEOVISION AI - VERSION 2
============================================================

Professional Streamlit interface for:

    1. Dashboard
    2. Automatic Analysis
    3. Land Cover Classification
    4. Change Detection V2
    5. Analytics
    6. Results

IMPORTANT:
    app.py = Version 1 (DO NOT MODIFY)

RUN:
    python -m streamlit run app_v2.py
============================================================
"""

from pathlib import Path
import sys
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT GEOVISION
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
    page_title="GeoVision AI V2",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .hero {
        padding: 30px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
    }

    .hero-subtitle {
        font-size: 19px;
        margin-top: 8px;
    }

    .card {
        border: 1px solid rgba(128,128,128,0.30);
        border-radius: 15px;
        padding: 20px;
        min-height: 130px;
    }

    .card-title {
        font-size: 15px;
        font-weight: 600;
    }

    .card-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 10px;
    }

    .section {
        font-size: 30px;
        font-weight: 800;
        margin-top: 10px;
    }

    .small {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛰️ GeoVision AI</div>
        <div class="hero-subtitle">
            Version 2 • Intelligent Satellite Image Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMPORT CHECK
# ============================================================

if not IMPORT_OK:

    st.error(
        "GeoVision AI could not be initialized."
    )

    st.exception(IMPORT_ERROR)

    st.stop()


# ============================================================
# CONFIGURATION
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

    output_dir = (
        Path(cfg.paths.output_dir)
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "Configuration error."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "auto_results" not in st.session_state:
    st.session_state.auto_results = []

if "landcover_results" not in st.session_state:
    st.session_state.landcover_results = []

if "changedet_results" not in st.session_state:
    st.session_state.changedet_results = []

if "last_run" not in st.session_state:
    st.session_state.last_run = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ GeoVision AI V2")

    st.caption(
        "Satellite Intelligence Dashboard"
    )

    st.markdown("---")

    st.subheader("⚡ System")

    if device.type == "cuda":

        st.success("GPU ACTIVE")

        st.write(
            gpu_name
        )

    else:

        st.warning("CPU MODE")

    st.write(
        f"PyTorch `{torch.__version__}`"
    )

    st.markdown("---")

    st.subheader("🧠 Models")

    st.write(
        "Land Cover: **ViT**"
    )

    st.write(
        "Change Detection: **Siamese ViT**"
    )

    st.markdown("---")

    st.subheader("📦 Checkpoints")

    if landcover_checkpoint.exists():

        st.success(
            "Land Cover ✓"
        )

    else:

        st.error(
            "Land Cover ✗"
        )

    if changedet_checkpoint.exists():

        st.success(
            "Change Detection ✓"
        )

    else:

        st.error(
            "Change Detection ✗"
        )

    st.markdown("---")

    st.subheader("📁 Output")

    st.code(
        str(output_dir)
    )


# ============================================================
# HELPER: LAND COVER MODEL
# ============================================================

@st.cache_resource
def load_landcover_model():

    model = build_landcover_vit(
        cfg
    )

    model = load_checkpoint(
        model,
        landcover_checkpoint,
        device,
    )

    model.eval()

    return model


# ============================================================
# HELPER: CHANGE DETECTION MODEL
# ============================================================

@st.cache_resource
def load_changedet_model():

    model = build_change_detection_vit(
        cfg
    )

    model = load_checkpoint(
        model,
        changedet_checkpoint,
        device,
    )

    model.eval()

    return model


# ============================================================
# HELPER: LAND COVER PREDICTION
# ============================================================

def run_landcover(image):

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

    model = load_landcover_model()

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

    confidence = confidence.item()

    predicted_class = (
        cfg.landcover.classes[
            index
        ]
    )

    probabilities = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    return (
        predicted_class,
        confidence,
        probabilities,
    )


# ============================================================
# HELPER: CHANGE DETECTION
# ============================================================

def run_changedet(
    before,
    after,
    threshold,
):

    before_tensor = (
        load_change_image(
            before,
            cfg,
        )
    )

    after_tensor = (
        load_change_image(
            after,
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

    model = load_changedet_model()

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
        if total_pixels > 0
        else 0
    )

    return (
        probability_map,
        change_mask,
        change_percentage,
    )


# ============================================================
# HELPER: OSCD RGB
# ============================================================

def read_oscd_rgb(scene_dir):

    try:

        import tifffile

        def read_band(name):

            return tifffile.imread(
                str(
                    Path(scene_dir)
                    / name
                )
            ).astype(
                np.float32
            )

    except ImportError:

        def read_band(name):

            return np.asarray(
                Image.open(
                    Path(scene_dir)
                    / name
                )
            ).astype(
                np.float32
            )

    r = read_band(
        "B04.tif"
    )

    g = read_band(
        "B03.tif"
    )

    b = read_band(
        "B02.tif"
    )

    rgb = np.stack(
        [
            r,
            g,
            b,
        ],
        axis=-1,
    )

    output = np.zeros_like(
        rgb,
        dtype=np.float32,
    )

    for c in range(3):

        lo = np.percentile(
            rgb[..., c],
            2,
        )

        hi = np.percentile(
            rgb[..., c],
            98,
        )

        if hi > lo:

            output[
                ...,
                c,
            ] = np.clip(
                (
                    rgb[
                        ...,
                        c,
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
        output * 255
    ).astype(
        np.uint8
    )


# ============================================================
# HELPER: CREATE RED OVERLAY
# ============================================================

def create_overlay(
    image,
    mask,
):

    image_np = np.asarray(
        image
    ).astype(
        np.float32
    )

    overlay = (
        image_np.copy()
    )

    red = np.array(
        [
            255,
            0,
            0,
        ],
        dtype=np.float32,
    )

    overlay[mask] = (
        0.35
        * overlay[mask]
        + 0.65
        * red
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_auto, tab_land, tab_change, tab_analytics, tab_results = st.tabs(
    [
        "🏠 Dashboard",
        "🤖 Auto Analysis",
        "🛰️ Land Cover",
        "🔄 Change Detection V2",
        "📊 Analytics",
        "📁 Results",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.markdown(
        '<div class="section">📊 System Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Welcome to **GeoVision AI Version 2**.

        This interface combines your trained Vision Transformer
        models into one satellite-image analysis platform.
        """
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="card">
            <div class="card-title">
            🛰️ LAND COVER
            </div>
            <div class="card-value">
            ViT
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="card">
            <div class="card-title">
            🔄 CHANGE DETECTION
            </div>
            <div class="card-value">
            Siamese ViT
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="card">
            <div class="card-title">
            ⚡ COMPUTE
            </div>
            <div class="card-value">
            {device.type.upper()}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        result_count = len(
            list(
                output_dir.glob("*")
            )
        )

        st.markdown(
            f"""
            <div class="card">
            <div class="card-title">
            📁 SAVED RESULTS
            </div>
            <div class="card-value">
            {result_count}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader(
        "🧠 Model Information"
    )

    model_table = pd.DataFrame(
        {
            "Model": [
                "Land Cover",
                "Change Detection",
            ],

            "Architecture": [
                "Vision Transformer",
                "Siamese Vision Transformer",
            ],

            "Checkpoint": [
                landcover_checkpoint.name,
                changedet_checkpoint.name,
            ],

            "Status": [
                "READY"
                if landcover_checkpoint.exists()
                else "MISSING",

                "READY"
                if changedet_checkpoint.exists()
                else "MISSING",
            ],
        }
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "📂 Dataset Status"
    )

    eurosat_root = (
        PROJECT_ROOT
        / "data"
        / "EuroSAT"
    )

    oscd_root = (
        PROJECT_ROOT
        / "data"
        / "OSCD"
    )

    d1, d2 = st.columns(2)

    with d1:

        if eurosat_root.exists():

            st.success(
                "✅ EuroSAT dataset detected"
            )

            class_count = len(
                [
                    p
                    for p in eurosat_root.iterdir()
                    if p.is_dir()
                ]
            )

            st.write(
                f"Class folders: **{class_count}**"
            )

        else:

            st.error(
                "❌ EuroSAT dataset not found"
            )

    with d2:

        if oscd_root.exists():

            st.success(
                "✅ OSCD dataset detected"
            )

        else:

            st.error(
                "❌ OSCD dataset not found"
            )


# ============================================================
# AUTO ANALYSIS
# ============================================================

with tab_auto:

    st.markdown(
        '<div class="section">🤖 Automatic Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Run GeoVision AI automatically using your datasets.
        No manual image upload is required.
        """
    )

    st.markdown("---")

    a1, a2, a3 = st.columns(3)

    with a1:

        rounds = st.number_input(
            "Number of rounds",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )

    with a2:

        auto_type = st.selectbox(
            "Analysis type",
            [
                "Both",
                "Land Cover Only",
                "Change Detection Only",
            ],
        )

    with a3:

        auto_threshold = st.slider(
            "Change threshold",
            0.05,
            0.90,
            0.10,
            0.05,
        )

    st.markdown("---")

    st.info(
        f"""
        **Auto configuration**

        Rounds: {rounds}

        Analysis: {auto_type}

        Change threshold: {auto_threshold:.2f}

        Device: {device}
        """
    )

    if st.button(
        "🚀 START AUTOMATIC ANALYSIS",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.auto_results = []

        progress = st.progress(
            0
        )

        status = st.empty()

        # ------------------------------------------
        # LAND COVER
        # ------------------------------------------

        land_rounds = []

        if auto_type in [
            "Both",
            "Land Cover Only",
        ]:

            status.write(
                "🔎 Finding EuroSAT images..."
            )

            try:

                land_rounds = (
                    find_landcover_images(
                        PROJECT_ROOT,
                        int(rounds),
                    )
                )

            except Exception as e:

                st.error(
                    "EuroSAT search failed."
                )

                st.exception(e)

        # ------------------------------------------
        # CHANGE DETECTION
        # ------------------------------------------

        change_rounds = []

        if auto_type in [
            "Both",
            "Change Detection Only",
        ]:

            status.write(
                "🔎 Finding OSCD city pairs..."
            )

            try:

                change_rounds = (
                    find_changedet_pairs(
                        PROJECT_ROOT,
                        int(rounds),
                    )
                )

            except Exception as e:

                st.error(
                    "OSCD search failed."
                )

                st.exception(e)

        total_jobs = (
            len(land_rounds)
            + len(change_rounds)
        )

        completed = 0

        # ==========================================
        # LAND COVER JOBS
        # ==========================================

        if land_rounds:

            st.subheader(
                "🛰️ Land Cover Analysis"
            )

            try:

                land_model = (
                    load_landcover_model()
                )

            except Exception as e:

                st.error(
                    "Land Cover model failed to load."
                )

                st.exception(e)

                land_model = None

            for (
                round_number,
                folder,
                image_path,
            ) in land_rounds:

                if land_model is None:
                    break

                status.write(
                    f"🛰️ Land Cover "
                    f"Round {round_number}: "
                    f"{image_path.name}"
                )

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    predicted_class, confidence, probabilities = run_landcover(
                        image
                    )

                    result = {
                        "Type": "Land Cover",
                        "Round": round_number,
                        "Input": image_path.name,
                        "Class": predicted_class,
                        "Confidence": confidence * 100,
                    }

                    st.session_state.auto_results.append(
                        result
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.image(
                            image,
                            caption=(
                                f"Round {round_number} • "
                                f"{folder.name}"
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

                    result_path = (
                        output_dir
                        / (
                            f"v2_auto_land_"
                            f"round{round_number:02d}_"
                            f"{image_path.stem}.png"
                        )
                    )

                    image.save(
                        result_path
                    )

                except Exception as e:

                    st.error(
                        f"Land Cover Round "
                        f"{round_number} failed."
                    )

                    st.exception(e)

                completed += 1

                if total_jobs:

                    progress.progress(
                        completed
                        / total_jobs
                    )

        # ==========================================
        # CHANGE DETECTION JOBS
        # ==========================================

        if change_rounds:

            st.subheader(
                "🔄 Change Detection Analysis"
            )

            try:

                change_model = (
                    load_changedet_model()
                )

            except Exception as e:

                st.error(
                    "Change Detection model failed to load."
                )

                st.exception(e)

                change_model = None

            for (
                round_number,
                city,
                before_dir,
                after_dir,
            ) in change_rounds:

                if change_model is None:
                    break

                status.write(
                    f"🔄 Change Detection "
                    f"Round {round_number}: "
                    f"{city.name}"
                )

                try:

                    probability_map, change_mask, change_percentage = run_changedet(
                        before_dir,
                        after_dir,
                        auto_threshold,
                    )

                    before_rgb = (
                        read_oscd_rgb(
                            before_dir
                        )
                    )

                    after_rgb = (
                        read_oscd_rgb(
                            after_dir
                        )
                    )

                    mask_img = Image.fromarray(
                        (
                            change_mask.astype(
                                np.uint8
                            )
                            * 255
                        )
                    )

                    mask_img = (
                        mask_img.resize(
                            (
                                after_rgb.shape[1],
                                after_rgb.shape[0],
                            ),
                            Image.Resampling.NEAREST,
                        )
                    )

                    full_mask = (
                        np.asarray(
                            mask_img
                        )
                        > 0
                    )

                    overlay = create_overlay(
                        after_rgb,
                        full_mask,
                    )

                    result = {
                        "Type": "Change Detection",
                        "Round": round_number,
                        "Input": city.name,
                        "Class": "-",
                        "Confidence": "-",
                        "Change %": change_percentage,
                        "Threshold": auto_threshold,
                    }

                    st.session_state.auto_results.append(
                        result
                    )

                    st.markdown(
                        f"### Round {round_number} • {city.name}"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.image(
                            before_rgb,
                            caption="BEFORE",
                            use_container_width=True,
                        )

                    with c2:

                        st.image(
                            after_rgb,
                            caption="AFTER",
                            use_container_width=True,
                        )

                    with c3:

                        st.image(
                            overlay,
                            caption="🔴 DETECTED CHANGES",
                            use_container_width=True,
                        )

                    m1, m2 = st.columns(2)

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

                    overlay_path = (
                        output_dir
                        / (
                            f"v2_auto_change_"
                            f"round{round_number:02d}_"
                            f"{city.name}_overlay.png"
                        )
                    )

                    mask_path = (
                        output_dir
                        / (
                            f"v2_auto_change_"
                            f"round{round_number:02d}_"
                            f"{city.name}_mask.png"
                        )
                    )

                    overlay.save(
                        overlay_path
                    )

                    mask_img.save(
                        mask_path
                    )

                except Exception as e:

                    st.error(
                        f"Change Detection Round "
                        f"{round_number} failed."
                    )

                    st.exception(e)

                completed += 1

                if total_jobs:

                    progress.progress(
                        completed
                        / total_jobs
                    )

        progress.progress(
            1.0
        )

        status.success(
            "✅ Automatic analysis completed."
        )

        st.session_state.last_run = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


# ============================================================
# MANUAL LAND COVER
# ============================================================

with tab_land:

    st.markdown(
        '<div class="section">🛰️ Land Cover Classification</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="v2_landcover",
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        st.image(
            image,
            caption="Input Satellite Image",
            use_container_width=True,
        )

        if st.button(
            "🚀 RUN LAND COVER",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    predicted_class, confidence, probabilities = run_landcover(
                        image
                    )

                    st.success(
                        "Analysis complete."
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.metric(
                            "Predicted Class",
                            predicted_class,
                        )

                    with c2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%",
                        )

                    st.subheader(
                        "📊 Class Probabilities"
                    )

                    data = {}

                    for i, name in enumerate(
                        cfg.landcover.classes
                    ):

                        data[name] = (
                            probabilities[i]
                            * 100
                        )

                    st.bar_chart(
                        data
                    )

                    st.session_state.landcover_results.append(
                        {
                            "Image": uploaded.name,
                            "Class": predicted_class,
                            "Confidence": confidence * 100,
                        }
                    )

                except Exception as e:

                    st.error(
                        "Land Cover analysis failed."
                    )

                    st.exception(e)


# ============================================================
# CHANGE DETECTION V2
# ============================================================

with tab_change:

    st.markdown(
        '<div class="section">🔄 Change Detection V2</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Version 2 provides probability statistics and
        threshold analysis in addition to the final
        change overlay.
        """
    )

    before_file = st.file_uploader(
        "BEFORE image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="v2_before",
    )

    after_file = st.file_uploader(
        "AFTER image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        key="v2_after",
    )

    threshold = st.slider(
        "Detection threshold",
        0.05,
        0.90,
        0.10,
        0.05,
        key="v2_threshold",
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

            st.image(
                before_image,
                caption="BEFORE",
                use_container_width=True,
            )

        with c2:

            st.image(
                after_image,
                caption="AFTER",
                use_container_width=True,
            )

        if st.button(
            "🔍 RUN CHANGE DETECTION V2",
            type="primary",
            use_container_width=True,
        ):

            temp_dir = (
                PROJECT_ROOT
                / ".streamlit_temp_v2"
            )

            temp_dir.mkdir(
                exist_ok=True
            )

            before_path = (
                temp_dir
                / "v2_before.png"
            )

            after_path = (
                temp_dir
                / "v2_after.png"
            )

            before_image.save(
                before_path
            )

            after_image.save(
                after_path
            )

            with st.spinner(
                "Running Change Detection V2..."
            ):

                try:

                    probability_map, change_mask, change_percentage = run_changedet(
                        before_path,
                        after_path,
                        threshold,
                    )

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
                            after_image.size,
                            Image.Resampling.NEAREST,
                        )
                    )

                    full_mask = (
                        np.asarray(
                            mask_full
                        )
                        > 0
                    )

                    overlay = create_overlay(
                        after_image,
                        full_mask,
                    )

                    st.success(
                        "Change Detection V2 completed."
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:

                        st.metric(
                            "Changed Area",
                            f"{change_percentage:.2f}%",
                        )

                    with m2:

                        st.metric(
                            "Maximum Probability",
                            f"{probability_map.max():.4f}",
                        )

                    with m3:

                        st.metric(
                            "Mean Probability",
                            f"{probability_map.mean():.4f}",
                        )

                    st.subheader(
                        "🔴 Detection Result"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.image(
                            before_image,
                            caption="BEFORE",
                            use_container_width=True,
                        )

                    with c2:

                        st.image(
                            after_image,
                            caption="AFTER",
                            use_container_width=True,
                        )

                    with c3:

                        st.image(
                            overlay,
                            caption="🔴 CHANGES",
                            use_container_width=True,
                        )

                    st.subheader(
                        "🗺️ Binary Change Mask"
                    )

                    st.image(
                        mask_full,
                        use_container_width=True,
                    )

                    st.subheader(
                        "📈 Threshold Analysis"
                    )

                    threshold_values = [
                        0.10,
                        0.20,
                        0.30,
                        0.40,
                        0.50,
                    ]

                    threshold_results = {}

                    for t in threshold_values:

                        threshold_results[
                            f"{t:.2f}"
                        ] = (
                            np.mean(
                                probability_map
                                >= t
                            )
                            * 100
                        )

                    st.bar_chart(
                        threshold_results
                    )

                    stats = pd.DataFrame(
                        {
                            "Threshold": threshold_values,
                            "Detected Area %": [
                                threshold_results[
                                    f"{t:.2f}"
                                ]
                                for t in threshold_values
                            ],
                        }
                    )

                    st.dataframe(
                        stats,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Save
                    overlay_path = (
                        output_dir
                        / "v2_manual_change_overlay.png"
                    )

                    mask_path = (
                        output_dir
                        / "v2_manual_change_mask.png"
                    )

                    overlay.save(
                        overlay_path
                    )

                    mask_full.save(
                        mask_path
                    )

                    st.success(
                        f"Saved to {output_dir}"
                    )

                    st.session_state.changedet_results.append(
                        {
                            "Before": before_file.name,
                            "After": after_file.name,
                            "Threshold": threshold,
                            "Change %": change_percentage,
                        }
                    )

                except Exception as e:

                    st.error(
                        "Change Detection V2 failed."
                    )

                    st.exception(e)


# ============================================================
# ANALYTICS
# ============================================================

with tab_analytics:

    st.markdown(
        '<div class="section">📊 Analytics</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Automatic Analysis History"
    )

    if st.session_state.auto_results:

        df = pd.DataFrame(
            st.session_state.auto_results
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        change_df = df[
            df["Type"]
            == "Change Detection"
        ]

        if not change_df.empty:

            st.subheader(
                "🔄 Change Detection Statistics"
            )

            if "Change %" in change_df.columns:

                numeric_change = pd.to_numeric(
                    change_df["Change %"],
                    errors="coerce",
                ).dropna()

                if not numeric_change.empty:

                    st.metric(
                        "Average Changed Area",
                        f"{numeric_change.mean():.2f}%",
                    )

                    st.bar_chart(
                        numeric_change
                    )

    else:

        st.info(
            "Run Auto Analysis to generate analytics."
        )

    st.markdown("---")

    st.subheader(
        "Land Cover History"
    )

    if st.session_state.landcover_results:

        st.dataframe(
            pd.DataFrame(
                st.session_state.landcover_results
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No manual land-cover results yet."
        )

    st.markdown("---")

    st.subheader(
        "Manual Change Detection History"
    )

    if st.session_state.changedet_results:

        st.dataframe(
            pd.DataFrame(
                st.session_state.changedet_results
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No manual change-detection results yet."
        )


# ============================================================
# RESULTS
# ============================================================

with tab_results:

    st.markdown(
        '<div class="section">📁 Results Center</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Results directory: `{output_dir}`"
    )

    files = sorted(
        output_dir.glob("*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not files:

        st.info(
            "No prediction results found yet."
        )

    else:

        st.write(
            f"**{len(files)} result files found.**"
        )

        for file in files[:50]:

            c1, c2, c3 = st.columns(
                [4, 2, 1]
            )

            with c1:

                st.write(
                    file.name
                )

            with c2:

                size_kb = (
                    file.stat().st_size
                    / 1024
                )

                st.write(
                    f"{size_kb:.1f} KB"
                )

            with c3:

                if file.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:

                    try:

                        with open(
                            file,
                            "rb",
                        ) as f:

                            st.download_button(
                                "⬇️",
                                f,
                                file_name=file.name,
                                key=f"download_{file.name}",
                            )

                    except Exception:
                        pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "GeoVision AI V2 • Satellite Image Intelligence • "
    "Land Cover + Change Detection"
)