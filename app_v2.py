"""
============================================================
GEOVISION AI — RESEARCH PROJECT WEBSITE V2
============================================================

V1:
    app.py              <- DO NOT MODIFY

V2:
    app_v2.py           <- Research website

Run:
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

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    page_title="GeoVision AI | Research",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RESEARCH WEBSITE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Header */

    .brand {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .hero {
        padding: 70px 45px;
        border-radius: 24px;
        margin-top: 20px;
        margin-bottom: 35px;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .hero-small {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 52px;
        line-height: 1.05;
        font-weight: 850;
        margin-top: 15px;
    }

    .hero-description {
        font-size: 20px;
        line-height: 1.6;
        max-width: 850px;
        margin-top: 22px;
    }

    /* Section */

    .section-label {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 35px;
    }

    .section-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .section-text {
        font-size: 17px;
        line-height: 1.7;
    }

    /* Cards */

    .research-card {
        border: 1px solid rgba(128,128,128,0.30);
        border-radius: 18px;
        padding: 25px;
        min-height: 205px;
    }

    .research-card h3 {
        margin-top: 0;
        font-size: 21px;
    }

    .research-card p {
        line-height: 1.6;
    }

    /* Metric */

    .metric-card {
        border: 1px solid rgba(128,128,128,0.30);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Pipeline */

    .pipeline {
        text-align: center;
        border: 1px solid rgba(128,128,128,0.30);
        border-radius: 15px;
        padding: 20px 10px;
        min-height: 110px;
    }

    .pipeline-icon {
        font-size: 30px;
    }

    .pipeline-title {
        font-weight: 750;
        margin-top: 8px;
    }

    /* Footer */

    .footer {
        text-align: center;
        padding-top: 40px;
        opacity: 0.7;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZATION
# ============================================================

if not IMPORT_OK:

    st.error(
        "GeoVision AI modules could not be imported."
    )

    st.exception(IMPORT_ERROR)

    st.stop()


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
        exist_ok=True
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
        "GeoVision AI initialization failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []

if "landcover_history" not in st.session_state:
    st.session_state.landcover_history = []

if "change_history" not in st.session_state:
    st.session_state.change_history = []


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def get_landcover_model():

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


@st.cache_resource
def get_changedet_model():

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
# LAND COVER INFERENCE
# ============================================================

def predict_landcover(image):

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

    model = get_landcover_model()

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_index = prediction.item()

    predicted_class = (
        cfg.landcover.classes[
            class_index
        ]
    )

    return (
        predicted_class,
        confidence.item(),
        probabilities[0]
        .cpu()
        .numpy()
    )


# ============================================================
# CHANGE DETECTION INFERENCE
# ============================================================

def predict_change(
    before_path,
    after_path,
    threshold,
):

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

    model = get_changedet_model()

    with torch.no_grad():

        probability_map = model(
            before,
            after
        )

    probability_map = (
        probability_map[
            0,
            0
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

    return (
        probability_map,
        change_mask,
        change_percentage
    )


# ============================================================
# RGB FOR OSCD
# ============================================================

def load_oscd_rgb(scene):

    try:

        import tifffile

        def read_band(name):

            return tifffile.imread(
                str(
                    Path(scene)
                    / name
                )
            ).astype(
                np.float32
            )

    except ImportError:

        def read_band(name):

            return np.asarray(
                Image.open(
                    Path(scene)
                    / name
                )
            ).astype(
                np.float32
            )

    r = read_band("B04.tif")
    g = read_band("B03.tif")
    b = read_band("B02.tif")

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    result = np.zeros_like(
        rgb,
        dtype=np.float32
    )

    for c in range(3):

        lo = np.percentile(
            rgb[..., c],
            2
        )

        hi = np.percentile(
            rgb[..., c],
            98
        )

        if hi > lo:

            result[..., c] = np.clip(
                (
                    rgb[..., c]
                    - lo
                )
                / (
                    hi - lo
                ),
                0,
                1
            )

    return (
        result * 255
    ).astype(
        np.uint8
    )


# ============================================================
# RED CHANGE OVERLAY
# ============================================================

def make_overlay(
    image,
    mask
):

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )

    overlay = (
        image_array.copy()
    )

    overlay[mask] = (
        0.35
        * overlay[mask]
        +
        0.65
        * np.array(
            [
                255,
                0,
                0
            ],
            dtype=np.float32
        )
    )

    return Image.fromarray(
        np.clip(
            overlay,
            0,
            255
        ).astype(
            np.uint8
        )
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">GEOVISION AI</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Research Project • Version 2"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Research",
            "Methodology",
            "Experiments",
            "Results",
            "Analytics",
            "About",
        ]
    )

    st.markdown("---")

    st.subheader(
        "System Status"
    )

    if device.type == "cuda":

        st.success(
            "GPU Available"
        )

        st.caption(
            gpu_name
        )

    else:

        st.warning(
            "CPU Mode"
        )

    st.markdown("---")

    st.caption(
        f"Output: {output_dir}"
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-small">
        RESEARCH PROJECT
        </div>

        <div class="hero-title">
        GeoVision AI
        </div>

        <div class="hero-description">
        AI-Based Satellite Image Analysis using
        Vision Transformers for Land Cover Classification
        and Multi-Temporal Change Detection.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-label">RESEARCH OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Understanding Earth from Satellite Imagery'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-text">

        GeoVision AI is a research-oriented computer vision
        framework for analysing satellite imagery.

        The system investigates two complementary tasks:
        **land cover classification** and **multi-temporal
        change detection**.

        Vision Transformer architectures are used to learn
        spatial and semantic representations from satellite
        imagery, while a Siamese architecture is used to
        compare observations acquired at different times.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">KEY COMPONENTS</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="research-card">

            <h3>🛰️ Land Cover</h3>

            <p>
            Vision Transformer based classification of
            satellite scenes into land-cover categories.
            </p>

            <b>Dataset:</b> EuroSAT<br>
            <b>Architecture:</b> ViT

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="research-card">

            <h3>🔄 Change Detection</h3>

            <p>
            Multi-temporal comparison of satellite scenes
            to identify spatial regions affected by change.
            </p>

            <b>Dataset:</b> OSCD<br>
            <b>Architecture:</b> Siamese ViT

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="research-card">

            <h3>🤖 Automated Analysis</h3>

            <p>
            Automatic dataset selection and model inference
            without requiring manual image selection.
            </p>

            <b>Mode:</b> Automated Experimentation<br>
            <b>Output:</b> Visual + Numerical

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">SYSTEM PIPELINE</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    pipeline = [
        ("🛰️", "Satellite", "Input"),
        ("🧹", "Preprocess", "Normalize"),
        ("🧠", "Transformer", "Inference"),
        ("📊", "Prediction", "Analysis"),
        ("🗺️", "Visualization", "Output"),
    ]

    for column, item in zip(
        [p1, p2, p3, p4, p5],
        pipeline
    ):

        with column:

            st.markdown(
                f"""
                <div class="pipeline">

                <div class="pipeline-icon">
                {item[0]}
                </div>

                <div class="pipeline-title">
                {item[1]}
                </div>

                <div class="small">
                {item[2]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">CURRENT SYSTEM</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Land Cover Model",
            "ViT"
        )

    with c2:

        st.metric(
            "Change Model",
            "Siamese ViT"
        )

    with c3:

        st.metric(
            "Compute",
            device.type.upper()
        )

    with c4:

        result_count = len(
            list(
                output_dir.glob("*")
            )
        )

        st.metric(
            "Saved Results",
            result_count
        )


# ============================================================
# RESEARCH
# ============================================================

elif page == "Research":

    st.markdown(
        '<div class="section-label">01 • RESEARCH</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Research Problem'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Problem Statement

        Satellite imagery provides a continuous source of
        information about the Earth's surface. However,
        extracting meaningful information from large
        collections of remotely sensed images requires
        automated computer vision techniques.

        GeoVision AI investigates the use of Transformer-based
        deep learning architectures for two related problems:

        1. **Land cover classification**
        2. **Multi-temporal change detection**

        ### Research Objective

        The primary objective is to develop an integrated
        satellite image analysis pipeline capable of identifying
        semantic land-cover categories and detecting changes
        between satellite observations acquired at different
        times.

        ### Research Questions

        **RQ1:** Can Vision Transformer representations be used
        effectively for satellite land-cover classification?

        **RQ2:** Can Siamese Transformer representations identify
        spatial changes between multi-temporal satellite images?

        **RQ3:** How does the detection threshold influence the
        estimated changed area?

        **RQ4:** Can the complete workflow be automated for
        repeatable satellite-image experiments?
        """
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">CONTRIBUTIONS</div>',
        unsafe_allow_html=True
    )

    contributions = [
        (
            "01",
            "Transformer-based classification",
            "A Vision Transformer based land-cover analysis pipeline."
        ),
        (
            "02",
            "Multi-temporal analysis",
            "A Siamese architecture for comparing satellite observations."
        ),
        (
            "03",
            "Automated experimentation",
            "Automatic dataset selection and repeatable inference."
        ),
        (
            "04",
            "Visual interpretation",
            "Change masks and red-overlay visualizations for interpretation."
        ),
    ]

    for number, title, description in contributions:

        c1, c2 = st.columns(
            [1, 5]
        )

        with c1:

            st.markdown(
                f"### {number}"
            )

        with c2:

            st.markdown(
                f"### {title}"
            )

            st.write(
                description
            )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="section-label">02 • METHODOLOGY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Proposed Methodology'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        GeoVision AI consists of two complementary inference
        pipelines.
        """
    )

    st.markdown("---")

    st.subheader(
        "🛰️ Pipeline A — Land Cover Classification"
    )

    a1, a2, a3, a4 = st.columns(4)

    steps = [
        ("01", "EuroSAT", "Satellite scene"),
        ("02", "Preprocessing", "Resize + Normalize"),
        ("03", "Vision Transformer", "Feature learning"),
        ("04", "Classification", "Land-cover class"),
    ]

    for col, step in zip(
        [a1, a2, a3, a4],
        steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="pipeline">

                <b>{step[0]}</b>

                <h4>{step[1]}</h4>

                <small>
                {step[2]}
                </small>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader(
        "🔄 Pipeline B — Change Detection"
    )

    b1, b2, b3, b4, b5 = st.columns(5)

    steps = [
        ("01", "Before", "T1 image"),
        ("02", "After", "T2 image"),
        ("03", "Siamese ViT", "Feature comparison"),
        ("04", "Probability", "Change map"),
        ("05", "Threshold", "Binary mask"),
    ]

    for col, step in zip(
        [b1, b2, b3, b4, b5],
        steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="pipeline">

                <b>{step[0]}</b>

                <h4>{step[1]}</h4>

                <small>
                {step[2]}
                </small>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader(
        "📚 Datasets"
    )

    dataset_table = pd.DataFrame(
        {
            "Dataset": [
                "EuroSAT",
                "OSCD",
            ],

            "Purpose": [
                "Land cover classification",
                "Multi-temporal change detection",
            ],

            "Input": [
                "Satellite scenes",
                "Before / After scenes",
            ],
        }
    )

    st.dataframe(
        dataset_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPERIMENTS
# ============================================================

elif page == "Experiments":

    st.markdown(
        '<div class="section-label">03 • EXPERIMENTS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Live Research Experiments'
        '</div>',
        unsafe_allow_html=True
    )

    experiment_type = st.selectbox(
        "Experiment",
        [
            "Automatic Experiment",
            "Land Cover",
            "Change Detection",
        ]
    )

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if experiment_type == "Automatic Experiment":

        st.subheader(
            "🤖 Automated Experiment"
        )

        rounds = st.number_input(
            "Number of rounds",
            min_value=1,
            max_value=20,
            value=3
        )

        mode = st.selectbox(
            "Analysis",
            [
                "Both",
                "Land Cover Only",
                "Change Detection Only"
            ]
        )

        threshold = st.slider(
            "Change threshold",
            0.05,
            0.90,
            0.10,
            0.05
        )

        if st.button(
            "🚀 RUN RESEARCH EXPERIMENT",
            type="primary",
            use_container_width=True
        ):

            progress = st.progress(
                0
            )

            status = st.empty()

            jobs = []

            if mode in [
                "Both",
                "Land Cover Only"
            ]:

                try:

                    jobs.extend(
                        [
                            (
                                "land",
                                x
                            )
                            for x in find_landcover_images(
                                ROOT,
                                int(rounds)
                            )
                        ]
                    )

                except Exception as e:

                    st.error(
                        "EuroSAT dataset could not be loaded."
                    )

                    st.exception(e)

            if mode in [
                "Both",
                "Change Detection Only"
            ]:

                try:

                    jobs.extend(
                        [
                            (
                                "change",
                                x
                            )
                            for x in find_changedet_pairs(
                                ROOT,
                                int(rounds)
                            )
                        ]
                    )

                except Exception as e:

                    st.error(
                        "OSCD dataset could not be loaded."
                    )

                    st.exception(e)

            total = len(jobs)

            for index, job in enumerate(
                jobs
            ):

                status.write(
                    f"Running experiment "
                    f"{index + 1}/{total}"
                )

                try:

                    if job[0] == "land":

                        (
                            round_number,
                            folder,
                            image_path
                        ) = job[1]

                        image = Image.open(
                            image_path
                        ).convert(
                            "RGB"
                        )

                        (
                            predicted,
                            confidence,
                            probabilities
                        ) = predict_landcover(
                            image
                        )

                        st.session_state.experiment_history.append(
                            {
                                "Type": "Land Cover",
                                "Round": round_number,
                                "Input": image_path.name,
                                "Prediction": predicted,
                                "Confidence": confidence * 100,
                            }
                        )

                        with st.expander(
                            f"Round {round_number} • "
                            f"Land Cover • "
                            f"{image_path.name}",
                            expanded=True
                        ):

                            c1, c2 = st.columns(
                                2
                            )

                            with c1:

                                st.image(
                                    image,
                                    use_container_width=True
                                )

                            with c2:

                                st.success(
                                    f"Prediction: {predicted}"
                                )

                                st.metric(
                                    "Confidence",
                                    f"{confidence * 100:.2f}%"
                                )

                    else:

                        (
                            round_number,
                            city,
                            before_dir,
                            after_dir
                        ) = job[1]

                        (
                            probability_map,
                            change_mask,
                            change_percentage
                        ) = predict_change(
                            before_dir,
                            after_dir,
                            threshold
                        )

                        st.session_state.experiment_history.append(
                            {
                                "Type": "Change Detection",
                                "Round": round_number,
                                "Input": city.name,
                                "Prediction": "Change Map",
                                "Change %": change_percentage,
                                "Threshold": threshold,
                            }
                        )

                        st.success(
                            f"Round {round_number} • "
                            f"{city.name} • "
                            f"Changed Area "
                            f"{change_percentage:.2f}%"
                        )

                except Exception as e:

                    st.error(
                        f"Experiment {index + 1} failed."
                    )

                    st.exception(e)

                progress.progress(
                    (index + 1) / max(
                        total,
                        1
                    )
                )

            status.success(
                "Research experiment completed."
            )

    # --------------------------------------------------------
    # MANUAL LAND COVER
    # --------------------------------------------------------

    elif experiment_type == "Land Cover":

        st.subheader(
            "🛰️ Land Cover Experiment"
        )

        uploaded = st.file_uploader(
            "Satellite image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

        if uploaded:

            image = Image.open(
                uploaded
            ).convert(
                "RGB"
            )

            st.image(
                image,
                use_container_width=True
            )

            if st.button(
                "RUN CLASSIFICATION",
                type="primary"
            ):

                with st.spinner(
                    "Running Vision Transformer..."
                ):

                    try:

                        (
                            predicted,
                            confidence,
                            probabilities
                        ) = predict_landcover(
                            image
                        )

                        st.success(
                            f"Predicted class: "
                            f"{predicted}"
                        )

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                        chart_data = {}

                        for i, name in enumerate(
                            cfg.landcover.classes
                        ):

                            chart_data[name] = (
                                probabilities[i]
                                * 100
                            )

                        st.bar_chart(
                            chart_data
                        )

                    except Exception as e:

                        st.error(
                            "Classification failed."
                        )

                        st.exception(e)

    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    else:

        st.subheader(
            "🔄 Change Detection Experiment"
        )

        before = st.file_uploader(
            "BEFORE image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="research_before"
        )

        after = st.file_uploader(
            "AFTER image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="research_after"
        )

        threshold = st.slider(
            "Detection threshold",
            0.05,
            0.90,
            0.10,
            0.05,
            key="research_threshold"
        )

        if before and after:

            before_image = Image.open(
                before
            ).convert(
                "RGB"
            )

            after_image = Image.open(
                after
            ).convert(
                "RGB"
            )

            if st.button(
                "RUN CHANGE DETECTION",
                type="primary"
            ):

                temp = (
                    ROOT
                    / ".research_temp"
                )

                temp.mkdir(
                    exist_ok=True
                )

                before_path = (
                    temp
                    / "before.png"
                )

                after_path = (
                    temp
                    / "after.png"
                )

                before_image.save(
                    before_path
                )

                after_image.save(
                    after_path
                )

                with st.spinner(
                    "Running Siamese Vision Transformer..."
                ):

                    try:

                        (
                            probability_map,
                            change_mask,
                            change_percentage
                        ) = predict_change(
                            before_path,
                            after_path,
                            threshold
                        )

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
                            Image.Resampling.NEAREST
                        )

                        full_mask = (
                            np.asarray(
                                mask
                            )
                            > 0
                        )

                        overlay = make_overlay(
                            after_image,
                            full_mask
                        )

                        st.success(
                            "Change detection completed."
                        )

                        m1, m2, m3 = st.columns(3)

                        with m1:

                            st.metric(
                                "Changed Area",
                                f"{change_percentage:.2f}%"
                            )

                        with m2:

                            st.metric(
                                "Maximum Probability",
                                f"{probability_map.max():.4f}"
                            )

                        with m3:

                            st.metric(
                                "Mean Probability",
                                f"{probability_map.mean():.4f}"
                            )

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.image(
                                before_image,
                                caption="BEFORE",
                                use_container_width=True
                            )

                        with c2:

                            st.image(
                                after_image,
                                caption="AFTER",
                                use_container_width=True
                            )

                        with c3:

                            st.image(
                                overlay,
                                caption="DETECTED CHANGE",
                                use_container_width=True
                            )

                    except Exception as e:

                        st.error(
                            "Change detection failed."
                        )

                        st.exception(e)


# ============================================================
# RESULTS
# ============================================================

elif page == "Results":

    st.markdown(
        '<div class="section-label">04 • RESULTS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Experimental Results'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section presents outputs generated by the
        GeoVision AI experiments. Metrics shown here are
        generated from actual model inference.
        """
    )

    files = sorted(
        output_dir.glob("*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    st.metric(
        "Generated Files",
        len(files)
    )

    st.markdown("---")

    if files:

        for file in files[:50]:

            st.write(
                f"📄 `{file.name}`"
            )

    else:

        st.info(
            "No experiment results have been saved yet."
        )

    st.markdown("---")

    st.subheader(
        "Current Checkpoints"
    )

    checkpoint_data = pd.DataFrame(
        {
            "Model": [
                "Land Cover",
                "Change Detection"
            ],

            "Checkpoint": [
                landcover_checkpoint.name,
                changedet_checkpoint.name
            ],

            "Available": [
                landcover_checkpoint.exists(),
                changedet_checkpoint.exists()
            ]
        }
    )

    st.dataframe(
        checkpoint_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="section-label">05 • ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Experimental Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.experiment_history:

        st.info(
            "Run an experiment to generate analytics."
        )

    else:

        df = pd.DataFrame(
            st.session_state.experiment_history
        )

        st.subheader(
            "Experiment History"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # Land cover confidence
        if "Confidence" in df.columns:

            confidence = pd.to_numeric(
                df["Confidence"],
                errors="coerce"
            ).dropna()

            if not confidence.empty:

                st.subheader(
                    "Land Cover Confidence"
                )

                st.bar_chart(
                    confidence
                )

        # Change area
        if "Change %" in df.columns:

            change = pd.to_numeric(
                df["Change %"],
                errors="coerce"
            ).dropna()

            if not change.empty:

                st.subheader(
                    "Detected Change Area"
                )

                st.metric(
                    "Mean Change Area",
                    f"{change.mean():.2f}%"
                )

                st.bar_chart(
                    change
                )


# ============================================================
# ABOUT
# ============================================================

else:

    st.markdown(
        '<div class="section-label">06 • ABOUT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'About GeoVision AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Project

        **GeoVision AI** is an AI-based satellite image
        analysis research project focused on automated
        interpretation of Earth observation imagery.

        ### Technologies

        - Python
        - PyTorch
        - Vision Transformers
        - Siamese Vision Transformers
        - Streamlit
        - EuroSAT
        - OSCD
        - CUDA acceleration

        ### Research Areas

        - Remote sensing
        - Computer vision
        - Deep learning
        - Earth observation
        - Land-cover classification
        - Change detection

        ### Future Work

        Possible future research directions include:

        - Improved change-detection architectures
        - Better spatial resolution
        - Multi-spectral feature fusion
        - Attention visualization
        - Robust threshold selection
        - Quantitative benchmark evaluation
        - Larger satellite datasets
        - Explainable AI
        """
    )

    st.markdown("---")

    st.subheader(
        "Reproducibility"
    )

    st.write(
        """
        Experiments should be evaluated using fixed datasets,
        documented checkpoints, reproducible preprocessing,
        and quantitative evaluation metrics.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    GeoVision AI • Research Prototype • Version 2

    <br><br>

    Satellite Image Intelligence using Vision Transformers

    </div>
    """,
    unsafe_allow_html=True
)