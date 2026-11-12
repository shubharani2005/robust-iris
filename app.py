"""Streamlit web interface for iris-based disease detection."""

import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import MODEL_PROFILES, weights_path  # noqa: E402
from src.inference import format_display_name, load_trained_model, predict_disease  # noqa: E402
from src.preprocessing import load_image_from_bytes  # noqa: E402
from src.webcam import prepare_webcam_from_capture  # noqa: E402
from src.auth import authenticate_user, register_user  # noqa: E402
from src.reporting import build_screening_report  # noqa: E402

st.set_page_config(
    page_title="Iris Disease Detection",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0f172a 0%, #16233d 45%, #1b2a4a 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("A Robust Iris-Based Disease Detection Framework")
st.caption("Deep learning + iridology for eye and systemic health screening")


def render_authentication() -> None:
    """Show account screens until an authenticated session exists."""
    st.markdown("### Secure access")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            account = authenticate_user(email, password)
            if account:
                st.session_state["user"] = account
                st.rerun()
            st.error("Invalid email or password.")
    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Full name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", help="Use at least 8 characters.")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(name, email, password)
                (st.success if success else st.error)(message)


if "user" not in st.session_state:
    render_authentication()
    st.stop()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.caption(f"Signed in as **{st.session_state['user']['name']}**")
with top_right:
    if st.button("Sign out"):
        del st.session_state["user"]
        st.rerun()

# --- Photo-type selection ---
# These are two genuinely different pathways, not a preference toggle: DR and
# chronic glaucoma have no visible signal in an external eye photo (they're
# retina / optic-nerve findings), so a fundus-camera photo is required to
# screen for them at all. Picking the wrong mode for your photo type would
# reproduce the original bug this project shipped with — confident,
# meaningless predictions on a photo type the model never learned from.
profile_labels = {key: cfg["label"] for key, cfg in MODEL_PROFILES.items()}
profile_label = st.radio(
    "Photo type",
    list(profile_labels.values()),
    horizontal=True,
    help="Pick the option matching the photo you actually have — the two modes use different models trained on different kinds of images.",
)
profile = next(key for key, label in profile_labels.items() if label == profile_label)
st.caption(MODEL_PROFILES[profile]["description"])

model_exists = weights_path(profile).exists()
if not model_exists:
    st.warning(
        f"No trained model found for '{MODEL_PROFILES[profile]['label']}'. "
        f"Train with: `python scripts/train.py --profile {profile}`"
    )


@st.cache_resource
def get_model(profile: str):
    return load_trained_model(profile)


def render_prediction_charts(result: dict, profile: str) -> None:
    """Render disease probability and systemic-risk charts for this scan."""
    st.subheader("Prediction charts")
    disease_col, systemic_col = st.columns(2)

    with disease_col:
        st.markdown("**Eye disease probability**")
        probabilities = result.get("all_probs") or {}
        if probabilities:
            chart_data = pd.DataFrame(
                {
                    "Condition": [format_display_name(label) for label in probabilities],
                    "Probability (%)": [round(value * 100, 1) for value in probabilities.values()],
                }
            ).set_index("Condition")
            st.bar_chart(chart_data, color="#38bdf8", height=300)
        else:
            st.info("The disease probability chart appears when a trained model is available.")

    with systemic_col:
        st.markdown("**Systemic health risk indicators**")
        conditions = (result.get("systemic") or {}).get("conditions", [])
        if profile == "external_eye" and conditions:
            chart_data = pd.DataFrame(
                {
                    "Condition": [item["label"] for item in conditions],
                    "Risk indicator (%)": [round(item["risk_score"] * 100, 1) for item in conditions],
                }
            ).set_index("Condition")
            st.line_chart(chart_data, color="#4ade80", height=300)
            st.caption("The connected values compare this scan's risk indicators; they are not a time trend.")
        else:
            st.info("Systemic risk charts apply to external-eye scans with a detected iris.")


def render_analysis(result: dict, source: str, profile: str, webcam_info: dict | None = None):
    """Display preprocessing, predictions, and systemic results."""
    if not result.get("valid_image", True):
        st.error(result.get("validation_error", "Invalid image. Please upload a supported eye image."))
        return

    st.subheader(f"Analysis — {source}")

    if webcam_info:
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            st.markdown("**Original capture**")
            st.image(cv2.cvtColor(webcam_info.get("raw_preview", webcam_info["preview"]), cv2.COLOR_BGR2RGB))
        with wcol2:
            st.markdown("**After zoom**")
            st.image(cv2.cvtColor(webcam_info.get("zoomed_preview", webcam_info["analysis_image"]), cv2.COLOR_BGR2RGB))
        with wcol3:
            st.markdown("**Eye crop for prediction**")
            st.image(cv2.cvtColor(webcam_info["analysis_image"], cv2.COLOR_BGR2RGB))

        q = webcam_info["quality"]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Zoom", f"{webcam_info.get('zoom', 1.0):.1f}x")
        mcol2.metric("Blur score", f"{q['blur']:.0f}", help="Higher is sharper (need > 80)")
        mcol3.metric("Brightness", f"{q['brightness']:.0f}", help="Ideal range 45–210")
        mcol4.metric("Quality", "OK" if q["ok"] else "Low")

        if q["ok"]:
            st.success(webcam_info["eye_meta"].get("message", "Ready for analysis."))
        else:
            for msg in q["messages"]:
                st.warning(msg)
            st.info("Prediction runs anyway, but results are more reliable when quality is OK.")

    st.markdown("### Image Processing")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(result["preprocessed"]["original"], cv2.COLOR_BGR2RGB))

    with col2:
        st.markdown("**Grayscale + Blur**")
        st.image(result["preprocessed"]["enhanced"], clamp=True)

    with col3:
        st.markdown("**Iris Detection**")
        detected = result["segmentation"]["detected_iris"]
        if detected is not None:
            st.image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        else:
            st.info("Iris circle not detected.")

    with col4:
        st.markdown("**K-Means Segmentation**")
        segmented = result["segmentation"]["segmented"]
        if segmented is not None:
            st.image(segmented)
        else:
            st.info("Segmentation unavailable.")

    st.divider()

    # Systemic/iridology screening reads iris texture — meaningless on a
    # fundus photo (no iris is visible in one), so those tabs only apply
    # to the external-eye pathway.
    if profile == "external_eye":
        tab_names = ["Eye Diseases (Deep Learning)", "Systemic Conditions (Iridology)", "Organ Zone Map"]
    else:
        tab_names = ["Eye Diseases (Deep Learning)"]
    tabs = st.tabs(tab_names)
    tab_eye = tabs[0]

    with tab_eye:
        if profile == "external_eye":
            st.subheader("Cataract Detection — MobileNetV2")
        else:
            st.subheader("Eye Disease Classification — MobileNetV2 (Fundus)")

        if not result["model_loaded"]:
            st.info("Train the CNN model to enable detection.")
        elif result["class_name"]:
            label = format_display_name(result["class_name"])
            confidence = result["confidence"] * 100
            if result.get("low_confidence"):
                st.warning("Low-confidence prediction. Treat this result as uncertain and seek a clinical assessment.")
            if result["class_name"].lower() == "normal":
                st.success(f"**{label}** — Confidence: {confidence:.1f}%")
            else:
                st.error(f"**{label}** detected — Confidence: {confidence:.1f}%")
        if profile == "external_eye":
            st.caption(
                "Detects: Cataract, Normal — trained on external eye photos. "
                "Glaucoma and diabetic retinopathy are not visible in external eye photos "
                '(they require a retinal exam); switch to "Fundus / Retinal Photo" mode '
                "above for those, or see the Systemic Conditions tab for iris-based screening."
            )
        else:
            st.caption(
                "Detects: Cataract, Diabetic Retinopathy, Glaucoma, Normal — requires a genuine "
                "fundus-camera photo (taken through the pupil, showing the retina). A regular "
                "phone/webcam photo of the eye will not work in this mode."
            )

    if profile == "external_eye":
        tab_systemic, tab_zones = tabs[1], tabs[2]

        with tab_systemic:
            st.subheader("Systemic Health Screening — Iridology (Punia et al.)")
            systemic = result.get("systemic") or {}

            if not systemic.get("iris_detected"):
                st.warning(systemic.get("error", "Iris not detected for systemic analysis."))
            else:
                st.info(
                    "Organ zones on the iris are analyzed for texture and density markers. "
                    "Results are screening indicators — not a clinical diagnosis."
                )
                for item in systemic.get("conditions", []):
                    status = item["status"]
                    risk = item["risk_score"] * 100
                    label = item["label"]

                    if status == "Normal":
                        st.success(f"**{label}** — {status} (risk {risk:.0f}%)")
                    elif status == "Borderline":
                        st.warning(f"**{label}** — {status} (risk {risk:.0f}%)")
                    else:
                        st.error(f"**{label}** — {status} (risk {risk:.0f}%)")

                    for flag in item.get("flags", []):
                        st.caption(f"  • {flag}")

                st.caption("Covers: Heart disease, Diabetes, Cholesterol, Lung / respiratory conditions")

        with tab_zones:
            st.subheader("Iridology Organ Zones")
            zone_results = (result.get("systemic") or {}).get("zone_results", {})

            if not zone_results:
                st.warning("Zone map requires a detected iris.")
            else:
                zcol1, zcol2 = st.columns(2)
                keys = list(zone_results.keys())
                for i, key in enumerate(keys):
                    zr = zone_results[key]
                    col = zcol1 if i % 2 == 0 else zcol2
                    with col:
                        st.markdown(f"**{zr['label']}** — {zr['status']}")
                        st.image(cv2.cvtColor(zr["overlay"], cv2.COLOR_BGR2RGB))

    st.divider()
    render_prediction_charts(result, profile)
    st.divider()
    report_bytes, report_filename = build_screening_report(
        result=result,
        profile_label=MODEL_PROFILES[profile]["label"],
        source=source,
        user=st.session_state["user"],
    )
    st.subheader("Your screening report")
    st.caption("A report is prepared automatically after each analysis. It includes the prediction, confidence, and any applicable systemic screening indicators.")
    st.download_button(
        "Download screening report",
        data=report_bytes,
        file_name=report_filename,
        mime="text/html",
        type="primary",
    )
    st.caption(
        "Disclaimer: AI-assisted screening tool for academic/research use. "
        "Always consult a healthcare professional for medical diagnosis."
    )


# --- Input mode selection ---
# A webcam can only ever produce an external eye photo — there's no way to
# capture a genuine fundus photo without specialized clinical equipment — so
# live capture is only offered in external-eye mode.
input_options = ["Upload Image", "Live Webcam"] if profile == "external_eye" else ["Upload Image"]
input_mode = st.radio(
    "Choose input method",
    input_options,
    horizontal=True,
)

image_bgr = None
webcam_info = None
source_label = ""

if input_mode == "Upload Image":
    upload_label = (
        "Upload an eye image (left eye preferred for systemic analysis)"
        if profile == "external_eye"
        else "Upload a fundus/retinal photo"
    )
    uploaded = st.file_uploader(
        upload_label,
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        image_bgr = load_image_from_bytes(uploaded.read())
        source_label = "Uploaded image"

else:
    st.markdown(
        "**Live scan:** Capture your eye, then use **Zoom** and **Pan** until the eye fills the "
        "right preview. The orange box shows what will be magnified."
    )

    if "webcam_zoom_slider" not in st.session_state:
        st.session_state["webcam_zoom_slider"] = 2.0
    if "webcam_pan_x_slider" not in st.session_state:
        st.session_state["webcam_pan_x_slider"] = 50
    if "webcam_pan_y_slider" not in st.session_state:
        st.session_state["webcam_pan_y_slider"] = 50

    camera_photo = st.camera_input("Capture your eye", key="iris_camera")

    # Persist capture in session — camera widget clears on slider rerun otherwise
    if camera_photo is not None:
        st.session_state["webcam_bytes"] = camera_photo.getvalue()
        st.session_state["webcam_analyzed"] = False

    if st.session_state.get("webcam_bytes"):
        if st.button("Retake photo"):
            del st.session_state["webcam_bytes"]
            st.session_state["webcam_analyzed"] = False

        frame_bgr = load_image_from_bytes(st.session_state["webcam_bytes"])

        st.markdown("### Frame your eye — adjust zoom & pan")
        zcol1, zcol2 = st.columns(2)
        with zcol1:
            zoom = st.slider(
                "Digital zoom",
                min_value=1.0,
                max_value=4.0,
                step=0.1,
                key="webcam_zoom_slider",
                help="Increase until your eye fills the zoomed preview on the right.",
            )
        with zcol2:
            pan_x = st.slider(
                "Pan left ↔ right",
                min_value=0,
                max_value=100,
                key="webcam_pan_x_slider",
            )

        pan_y = st.slider(
            "Pan up ↕ down",
            min_value=0,
            max_value=100,
            key="webcam_pan_y_slider",
        )

        preview_info = prepare_webcam_from_capture(
            frame_bgr,
            zoom=zoom,
            pan_x=pan_x / 100.0,
            pan_y=pan_y / 100.0,
        )

        st.caption(f"Current zoom: **{zoom:.1f}x** — orange box = area being magnified")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Capture + crop area**")
            st.image(cv2.cvtColor(preview_info["raw_preview"], cv2.COLOR_BGR2RGB))
        with pcol2:
            st.markdown("**Zoomed preview (live)**")
            st.image(cv2.cvtColor(preview_info["zoomed_preview"], cv2.COLOR_BGR2RGB))

        analyze = st.button("Analyze eye", type="primary")

        if analyze:
            st.session_state["webcam_analyzed"] = True

        if st.session_state.get("webcam_analyzed"):
            webcam_info = preview_info
            image_bgr = webcam_info["analysis_image"]
            source_label = f"Live webcam ({zoom:.1f}x zoom)"

# --- Run analysis ---
if image_bgr is not None:
    model = get_model(profile) if model_exists else None
    with st.spinner("Analyzing image..."):
        result = predict_disease(image_bgr, model=model, profile=profile)
    render_analysis(result, source_label, profile, webcam_info=webcam_info)

else:
    st.markdown(
        """
        ### Supported screening

        **"External Eye Photo" mode (Deep Learning — MobileNetV2)**
        - Cataract, Normal — from a regular phone/webcam close-up of the eye

        **"Fundus / Retinal Photo" mode (Deep Learning — MobileNetV2)**
        - Cataract, Diabetic Retinopathy, Glaucoma, Normal — requires an actual
          fundus-camera photo (specialized clinical equipment). A regular photo
          of the eye will not work in this mode — the disease signal for DR and
          chronic glaucoma simply isn't visible from outside the eye.

        **Systemic conditions (Iridology — Punia et al.)** — External Eye Photo mode only
        - Heart / cardiovascular disease
        - Diabetes (pancreas zone)
        - Cholesterol abnormalities
        - Lung / respiratory conditions

        ### Input options
        1. **Upload Image** — best accuracy in either mode
        2. **Live Webcam** — External Eye Photo mode only (a webcam can't capture a fundus photo)

        ### Setup
        ```bash
        pip install -r requirements.txt
        python scripts/train.py --profile external_eye --model mobilenet
        python scripts/train.py --profile fundus --model mobilenet
        python scripts/calibrate_systemic.py
        streamlit run app.py
        ```
        """
    )
