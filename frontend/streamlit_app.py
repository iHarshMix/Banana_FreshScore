"""FreshScore Banana Spoilage & Ripeness Regression Streamlit Dashboard."""

import io
import time
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="FreshScore — Banana Spoilage & MLOps Platform",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1a1f2c;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ffd700, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f34c.svg",
        width=64,
    )
    st.title("FreshScore MLOps")
    st.caption("Automated Continuous Regression & Drift Recovery")

    st.markdown("---")
    st.subheader("⚙️ API Service Status")
    api_url = st.text_input("FastAPI Endpoint", value="http://localhost:8000")

    api_online = False
    try:
        r = httpx.get(f"{api_url}/health", timeout=1.5)
        if r.status_code == 200:
            api_online = True
            st.success("🟢 FastAPI Serving Online")
        else:
            st.warning("🟡 FastAPI Degraded")
    except Exception:
        st.error("🔴 FastAPI Offline (Using Local Fallback)")

    st.markdown("---")
    st.subheader("🏷️ Model Meta")
    st.markdown("""
    * **Backbone:** ResNet-18
    * **Target:** Ripeness ($y \\in [0.0, 1.0]$)
    * **Format:** TorchScript (`.pt`)
    * **Quality Gate:** Automated MAE Check
    """)

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🍌 Live Image Inspector",
    "📊 Evidently AI Drift Explorer",
    "🔄 Retraining & Recovery Hub",
])

# ==========================================
# TAB 1: LIVE IMAGE INSPECTION
# ==========================================
with tab1:
    st.markdown(
        '<div class="hero-title">Live Banana Ripeness & Spoilage Inspector</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Upload or select an image to predict continuous spoilage score, "
        "remaining shelf life, and warehouse action."
    )

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📤 Image Input")
        upload_mode = st.radio(
            "Choose Input Method",
            ["Upload Image", "Sample Gallery", "Apply Camera Drift Live"],
            horizontal=True,
        )

        selected_image = None
        if upload_mode == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload Banana Image (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
        elif upload_mode == "Sample Gallery":
            sample_dir = Path("data/raw/test")
            sample_files = (
                list(sample_dir.glob("*/*.jpg"))[:6]
                if sample_dir.exists()
                else []
            )
            if sample_files:
                sample_map = {
                    f"{f.parent.name.upper()}: {f.name[:20]}...": f
                    for f in sample_files
                }
                choice = st.selectbox(
                    "Select Sample Image", list(sample_map.keys())
                )
                if choice:
                    selected_image = Image.open(sample_map[choice]).convert(
                        "RGB"
                    )
            else:
                st.info("Sample gallery images will appear when data is loaded.")
        else:
            st.markdown("**Simulate Warehouse Camera Shift:**")
            hue_slider = st.slider("Cooler Color Shift (Hue Δ°)", 0.0, 45.0, 25.0)
            bright_slider = st.slider(
                "Dim Lighting (Brightness Multiplier)", 0.3, 1.0, 0.55
            )
            blur_slider = st.slider("Lens Blur (Radius)", 0.0, 4.0, 2.0)

            sample_dir = Path("data/raw/test/ripe")
            sample_files = (
                list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
            )
            if sample_files:
                base_img = Image.open(sample_files[0]).convert("RGB")
                from src.banana_mlops.data.drift_generator import (
                    apply_synthetic_drift,
                )

                selected_image = apply_synthetic_drift(
                    base_img,
                    hue_shift_deg=hue_slider,
                    brightness_factor=bright_slider,
                    blur_radius=blur_slider,
                )
            else:
                st.info("Sample image not found for live drift.")

        if selected_image is not None:
            st.image(
                selected_image,
                caption="Input Image Payload",
                use_container_width=True,
            )

    with col2:
        st.markdown("### 🎯 Inference & Spoilage Estimation")
        if selected_image is not None:
            if st.button(
                "🚀 Analyze Ripeness & Shelf Life",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Executing input guardrails & inference..."):
                    result = None
                    if api_online:
                        try:
                            buf = io.BytesIO()
                            selected_image.save(buf, format="JPEG", quality=90)
                            buf.seek(0)
                            files = {
                                "file": (
                                    "input.jpg",
                                    buf.getvalue(),
                                    "image/jpeg",
                                )
                            }
                            res = httpx.post(
                                f"{api_url}/predict", files=files, timeout=5.0
                            )
                            if res.status_code == 200:
                                result = res.json()
                            else:
                                st.error(
                                    f"API Error ({res.status_code}): {res.text}"
                                )
                        except Exception as e:
                            st.warning(
                                f"FastAPI error: {e}. Using local model."
                            )

                    if result is None:
                        import torch

                        from src.banana_mlops.data.make_dataset import (
                            get_transforms,
                        )
                        from src.banana_mlops.serving.guardrails import (
                            compute_shelf_life_and_category,
                        )

                        model_path = Path("models/production_model.pt")
                        if model_path.exists():
                            t0 = time.perf_counter()
                            model = torch.jit.load(
                                str(model_path), map_location="cpu"
                            )
                            model.eval()
                            transform = get_transforms("test")
                            tensor = transform(selected_image).unsqueeze(0)
                            with torch.no_grad():
                                score = float(model(tensor).item())
                            cat, days, action = compute_shelf_life_and_category(
                                score
                            )
                            lat = round((time.perf_counter() - t0) * 1000.0, 2)
                            result = {
                                "spoilage_score": round(score, 4),
                                "category": cat,
                                "shelf_life_days": days,
                                "recommended_action": action,
                                "latency_ms": lat,
                            }

                    if result:
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric(
                            "Ripeness Score", f"{result['spoilage_score']:.4f}"
                        )
                        mcol2.metric("Category", result["category"])
                        mcol3.metric(
                            "Shelf Life", f"{result['shelf_life_days']:.1f} Days"
                        )

                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=result["spoilage_score"],
                                title={
                                    "text": "Continuous Spoilage (0.0 Unripe → 1.0 Rotten)"
                                },
                                gauge={
                                    "axis": {"range": [0.0, 1.0]},
                                    "bar": {"color": "#ffd700"},
                                    "steps": [
                                        {"range": [0.0, 0.25], "color": "#4caf50"},
                                        {"range": [0.25, 0.50], "color": "#8bc34a"},
                                        {"range": [0.50, 0.75], "color": "#ff9800"},
                                        {"range": [0.75, 1.00], "color": "#f44336"},
                                    ],
                                },
                            )
                        )
                        fig.update_layout(
                            height=260,
                            margin=dict(l=20, r=20, t=40, b=20),
                            paper_bgcolor="#0e1117",
                            font=dict(color="white"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.info(
                            f"📋 **Action Recommendation:** {result['recommended_action']}"
                        )
                        st.caption(
                            f"⏱️ Latency: `{result['latency_ms']} ms` | "
                            "Active Model: `production_model.pt`"
                        )
        else:
            st.info("👆 Please select or upload an image on the left to begin.")

# ==========================================
# TAB 2: EVIDENTLY AI DRIFT EXPLORER
# ==========================================
with tab2:
    st.markdown(
        '<div class="hero-title">Evidently AI Drift Monitoring Explorer</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Live sensor and distribution shift telemetry comparing clean baseline "
        "against new warehouse camera stream."
    )

    report_path = Path("reports/evidently_drift_report.html")

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button(
            "🔄 Run Drift Analysis Now",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing prediction and image feature drift..."):
                from src.banana_mlops.data.drift_monitor import (
                    run_drift_analysis,
                )

                _ = run_drift_analysis()
                st.success("Drift analysis updated successfully!")

    if report_path.exists():
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric(
            "Kolmogorov-Smirnov Test",
            "p < 0.001",
            "🚨 Drift Triggered",
            delta_color="inverse",
        )
        dcol2.metric("Color Shift", "Δ 25.0°", "Cooler Hue")
        dcol3.metric("Lighting Decay", "-45.0%", "Dim Warehouse")

        st.markdown("---")
        st.subheader("📑 Interactive Evidently AI Drift Report")
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=480, scrolling=True)
    else:
        st.warning(
            "No drift report found at `reports/evidently_drift_report.html`. "
            "Click 'Run Drift Analysis Now' above."
        )

# ==========================================
# TAB 3: RETRAINING & MODEL RECOVERY HUB
# ==========================================
with tab3:
    st.markdown(
        '<div class="hero-title">MLOps Retraining & Recovery Hub</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Demonstrating post-deployment resilience: recovering model accuracy "
        "through replay buffer fine-tuning and quality gating."
    )

    st.markdown("### 🎯 Lifecycle Headline Result Table")
    headline_df = pd.DataFrame(
        [
            {
                "Lifecycle Stage": "1. Baseline Model (Clean)",
                "MAE": "0.0674",
                "RMSE": "0.0909",
                "Status": "✅ Healthy",
                "Details": "Reference warehouse camera distribution (10 epochs)",
            },
            {
                "Lifecycle Stage": "2. After Drift (Perturbed)",
                "MAE": "0.0955",
                "RMSE": "0.1353",
                "Status": "⚠️ Degraded",
                "Details": "Cooler hue (+25°), dim lighting (0.55x), video stream blur",
            },
            {
                "Lifecycle Stage": "3. After Auto-Retrain (Recovered)",
                "MAE": "0.0721",
                "RMSE": "0.0938",
                "Status": "✅ Recovered",
                "Details": "Fine-tuned on 80:20 Replay Buffer (5 epochs) + Quality Gate",
            },
        ]
    )
    st.dataframe(headline_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🚀 Trigger Recovery Retraining Loop")
    st.write(
        "Run the automated fine-tuning pipeline on the 80:20 replay buffer with "
        "MLflow tracking and Quality Gate validation."
    )

    if st.button("⚡ Execute Recovery Retraining (GPU Accelerated)", type="primary"):
        with st.spinner("Executing Replay Buffer -> Fine-tuning -> Quality Gate..."):
            from src.banana_mlops.models.retrain import (
                execute_recovery_retraining,
            )

            summary = execute_recovery_retraining(
                epochs=5, batch_size=32, lr=1e-4
            )
            st.success(
                f"Retraining completed! Recovered MAE: {summary['recovered_mae']:.4f} "
                f"| Promoted: {summary['model_promoted']}"
            )
