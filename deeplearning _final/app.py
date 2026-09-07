import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================
st.set_page_config(
    page_title="Road Accident Risk Prediction | AI Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING (MODERN DARK / SLATE THEME)
# ============================================================
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Header Container */
    .app-header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .app-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #38bdf8;
        margin-bottom: 8px;
    }

    .app-desc {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    /* Pulse Status Indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Cards */
    .custom-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }

    /* Severity Badges */
    .badge-fatal {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }

    .badge-serious {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid #f59e0b;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }

    .badge-slight {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid #3b82f6;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }

    /* Risk Level Box */
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.4));
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .risk-banner-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(120, 53, 15, 0.4));
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(127, 29, 29, 0.4));
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    /* Summary Metric Box */
    .metric-chip {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-chip-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-chip-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Clean Streamlit elements styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DNN_MODEL_PATH = os.path.join(BASE_DIR, "accident_risk_dnn.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "SHAP_feature_importance.csv")

LOCAL_YOLO_PATH = os.path.join(BASE_DIR, "best.pt")
FALLBACK_YOLO_PATH = r"C:\Users\shita\runs\detect\runs\dats2022_yolo_fast\weights\best.pt"
YOLO_MODEL_PATH = LOCAL_YOLO_PATH if os.path.exists(LOCAL_YOLO_PATH) else FALLBACK_YOLO_PATH

SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "traffic_image.jpg")


# ============================================================
# RESOURCE LOADING
# ============================================================
@st.cache_resource
def load_dnn_model():
    """Loads the trained Deep Neural Network model."""
    if not os.path.exists(DNN_MODEL_PATH):
        raise FileNotFoundError(f"DNN Model file not found at: {DNN_MODEL_PATH}")
    return tf.keras.models.load_model(DNN_MODEL_PATH)


@st.cache_resource
def load_scaler():
    """Loads the fitted StandardScaler for feature preprocessing."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_yolo_model():
    """Loads the trained YOLOv8 model for road-scene object detection."""
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_MODEL_PATH}")
    return YOLO(YOLO_MODEL_PATH)


@st.cache_data
def load_shap_data():
    """Loads precomputed SHAP feature importance values."""
    if os.path.exists(SHAP_PATH):
        return pd.read_csv(SHAP_PATH)
    return None


# Attempt Resource Initialization
dnn_loaded = False
scaler_loaded = False
yolo_loaded = False
init_errors = []

try:
    dnn_model = load_dnn_model()
    dnn_loaded = True
except Exception as e:
    dnn_model = None
    init_errors.append(f"DNN Model error: {e}")

try:
    scaler = load_scaler()
    scaler_loaded = True
except Exception as e:
    scaler = None
    init_errors.append(f"Scaler error: {e}")

try:
    yolo_model = load_yolo_model()
    yolo_loaded = True
except Exception as e:
    yolo_model = None
    init_errors.append(f"YOLO Model error: {e}")

shap_df = load_shap_data()
all_systems_ready = dnn_loaded and scaler_loaded and yolo_loaded

# ============================================================
# HEADER SECTION
# ============================================================
st.markdown(
    f"""
    <div class="app-header-box">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div class="app-title">🚦 Road Accident Risk Prediction</div>
                <div class="app-subtitle">AI-Based Accident Severity & Road Scene Risk Assessment</div>
                <div class="app-desc">Analyze accident-related information and road-scene images to estimate accident severity and overall risk.</div>
            </div>
            <div>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>{"● AI System Ready" if all_systems_ready else "⚠️ System Initializing"}</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if init_errors:
    with st.expander("System Diagnostic Notices", expanded=False):
        for err in init_errors:
            st.warning(err)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🚦 System Overview")
    st.caption("Road Accident Risk Prediction Intelligence")

    st.markdown(
        """
        <div class="custom-card" style="padding: 14px; margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">ACTIVE ARCHITECTURE</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">• DNN Model (186 Features)</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 2px;">• YOLO Vision Model (DATS-22)</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 2px;">• SHAP Explainability Engine</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 2px;">• Multi-Model Benchmarking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Module:",
        ["🏠 Risk Assessment", "📊 Model Analysis", "🔍 Explainability"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚙️ Engine Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.write("**DNN Model:**")
        st.write("**YOLOv8:**")
        st.write("**Scaler:**")
        st.write("**SHAP Data:**")
    with status_col2:
        st.write("🟢 Loaded" if dnn_loaded else "🔴 Error")
        st.write("🟢 Loaded" if yolo_loaded else "🔴 Error")
        st.write("🟢 Loaded" if scaler_loaded else "🔴 Error")
        st.write("🟢 Available" if shap_df is not None else "⚪ Not found")

    st.divider()
    st.caption("Accident Risk Intelligence • Deep Learning & Computer Vision")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def construct_feature_vector(inputs_dict, fitted_scaler):
    """
    Constructs an exact 186-feature vector matching fitted_scaler.feature_names_in_.
    Initializes a zero DataFrame with all expected columns, populates numerical
    features and one-hot categorical indicator flags.
    """
    expected_features = list(fitted_scaler.feature_names_in_)
    df = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)

    # Numerical inputs
    df["Number_of_vehicles_involved"] = float(inputs_dict.get("Number_of_vehicles_involved", 2))
    df["Number_of_casualties"] = float(inputs_dict.get("Number_of_casualties", 1))
    df["Hour"] = float(inputs_dict.get("Hour", 12))

    # Categorical mappings (set corresponding one-hot column to 1 if present)
    for col_prefix, val in inputs_dict.items():
        if col_prefix in ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]:
            continue
        col_name = f"{col_prefix}_{val}"
        if col_name in df.columns:
            df[col_name] = 1.0

    # Scale using exact saved scaler
    scaled_array = fitted_scaler.transform(df)
    return scaled_array


def render_risk_gauge(score, level):
    """Generates a professional semi-circular gauge using Plotly."""
    if level == "Low":
        bar_color = "#10b981"
    elif level == "Medium":
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f", "font": {"size": 34, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0.0, 1.0], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#1e293b",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0.0, 0.40], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [0.40, 0.70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [0.70, 1.00], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=25, b=10),
        height=220,
        font=dict(color="#f8fafc", family="Inter"),
    )
    return fig


# ============================================================
# MODULE 1: RISK ASSESSMENT (MAIN PAGE)
# ============================================================
if app_mode == "🏠 Risk Assessment":
    st.markdown("## 🧠 Accident Risk Assessment")
    st.markdown(
        "Estimate accident severity and overall risk by combining accident parameters "
        "with real-time road scene computer vision analysis."
    )

    # Initialize session state for analysis results
    if "has_dnn_prediction" not in st.session_state:
        st.session_state.has_dnn_prediction = False
    if "has_yolo_detection" not in st.session_state:
        st.session_state.has_yolo_detection = False

    # ------------------------------------------------------------
    # SECTION A: ACCIDENT INFORMATION (INPUT FORM)
    # ------------------------------------------------------------
    st.markdown("### 📝 Accident Information")
    st.caption("Provide road, vehicle, and incident parameters required by the trained DNN model.")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown(
            """
            <div class="card-title">🌦️ Environmental & Road Conditions</div>
            """,
            unsafe_allow_html=True,
        )
        weather_cond = st.selectbox(
            "Weather Condition",
            [
                "Normal",
                "Raining",
                "Raining and Windy",
                "Fog or mist",
                "Cloudy",
                "Windy",
                "Snow",
                "Other",
            ],
            index=0,
        )

        road_surface = st.selectbox(
            "Road Surface Condition",
            ["Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"],
            index=0,
        )

        light_cond = st.selectbox(
            "Light Condition",
            [
                "Daylight",
                "Darkness - lights lit",
                "Darkness - lights unlit",
                "Darkness - no lighting",
            ],
            index=0,
        )

        road_align = st.selectbox(
            "Road Alignment",
            [
                "Tangent road with flat terrain",
                "Gentle horizontal curve",
                "Sharp reverse curve",
                "Steep grade downward with mountainous terrain",
                "Tangent road with mild grade and flat terrain",
                "Tangent road with mountainous terrain and",
            ],
            index=0,
        )

    with col_input2:
        st.markdown(
            """
            <div class="card-title">🚗 Traffic & Incident Dynamics</div>
            """,
            unsafe_allow_html=True,
        )

        num_vehicles = st.slider(
            "Number of Vehicles Involved",
            min_value=1,
            max_value=7,
            value=2,
        )

        num_casualties = st.slider(
            "Number of Casualties",
            min_value=1,
            max_value=8,
            value=1,
        )

        accident_hr = st.slider(
            "Accident / Current Hour (24h)",
            min_value=0,
            max_value=23,
            value=14,
        )

        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
        )

        area_accident = st.selectbox(
            "Area Accident Occurred",
            [
                "Office areas",
                "Residential areas",
                "Hospital areas",
                "Industrial areas",
                "Church areas",
                "School areas",
                "Other",
            ],
            index=1,
        )

    with col_input3:
        st.markdown(
            """
            <div class="card-title">👤 Driver & Incident Factors</div>
            """,
            unsafe_allow_html=True,
        )

        driver_age = st.selectbox(
            "Driver Age Band",
            ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
            index=0,
        )

        driving_exp = st.selectbox(
            "Driving Experience",
            ["Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Licence", "Unknown"],
            index=2,
        )

        veh_type = st.selectbox(
            "Vehicle Type",
            [
                "Automobile",
                "Motorcycle",
                "Public (12 seats)",
                "Pick up upto 10Q",
                "Lorry (41?100Q)",
                "Long lorry",
                "Bajaj",
                "Other",
            ],
            index=0,
        )

        collision_type = st.selectbox(
            "Type of Collision",
            [
                "Vehicle with vehicle collision",
                "Collision with roadside objects",
                "Collision with pedestrians",
                "Rollover",
                "Collision with roadside-parked vehicles",
                "Other",
            ],
            index=0,
        )

        accident_cause = st.selectbox(
            "Primary Cause of Accident",
            [
                "No distancing",
                "Driving at high speed",
                "Driving carelessly",
                "Overspeed",
                "Changing lane to the right",
                "Changing lane to the left",
                "No priority to vehicle",
                "Other",
            ],
            index=1,
        )

    predict_btn = st.button("🔮 Predict Accident Risk", use_container_width=True, type="primary")

    # Run DNN prediction on click
    if predict_btn:
        if not dnn_loaded or not scaler_loaded:
            st.error("Error: Trained DNN model or Scaler could not be loaded from disk.")
        else:
            # Map user inputs to exact features
            inputs_payload = {
                "Number_of_vehicles_involved": num_vehicles,
                "Number_of_casualties": num_casualties,
                "Hour": accident_hr,
                "Day_of_week": day_of_week,
                "Weather_conditions": weather_cond,
                "Road_surface_conditions": road_surface,
                "Light_conditions": light_cond,
                "Road_allignment": road_align,
                "Area_accident_occured": area_accident,
                "Age_band_of_driver": driver_age,
                "Driving_experience": driving_exp,
                "Type_of_vehicle": veh_type,
                "Type_of_collision": collision_type,
                "Cause_of_accident": accident_cause,
            }

            try:
                # Preprocess & Scale
                scaled_input = construct_feature_vector(inputs_payload, scaler)

                # Predict with DNN
                raw_pred = dnn_model.predict(scaled_input, verbose=0)[0]
                pred_probabilities = [float(p) for p in raw_pred]

                # Target severity classes defined in project
                classes = ["Fatal Injury", "Serious Injury", "Slight Injury"]
                pred_class_idx = int(np.argmax(pred_probabilities))
                pred_class = classes[pred_class_idx]
                pred_confidence = pred_probabilities[pred_class_idx]

                # Project DNN risk score
                dnn_risk_score = float(np.max(pred_probabilities))

                # Store in session state
                st.session_state.has_dnn_prediction = True
                st.session_state.pred_probabilities = pred_probabilities
                st.session_state.pred_class = pred_class
                st.session_state.pred_confidence = pred_confidence
                st.session_state.dnn_risk_score = dnn_risk_score
                st.session_state.classes = classes

            except Exception as e:
                st.error(f"Prediction processing error: {e}")

    # ------------------------------------------------------------
    # SECTION B: DNN PREDICTION CARD
    # ------------------------------------------------------------
    if st.session_state.has_dnn_prediction:
        st.markdown("---")
        st.markdown("### 📊 DNN Accident Severity Prediction")

        dnn_res_col1, dnn_res_col2 = st.columns([1, 2])

        with dnn_res_col1:
            st.markdown(
                """
                <div class="custom-card">
                    <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;">Predicted Severity Class</div>
                """,
                unsafe_allow_html=True,
            )

            p_class = st.session_state.pred_class
            if p_class == "Fatal Injury":
                st.markdown('<div class="badge-fatal">🚨 Fatal Injury</div>', unsafe_allow_html=True)
            elif p_class == "Serious Injury":
                st.markdown('<div class="badge-serious">⚠️ Serious Injury</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-slight">ℹ️ Slight Injury</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                    <div style="margin-top: 16px;">
                        <div class="metric-chip-label">Model Confidence / DNN Score</div>
                        <div class="metric-chip-value">{st.session_state.dnn_risk_score:.4f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with dnn_res_col2:
            st.markdown(
                """
                <div class="custom-card">
                    <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">Severity Class Probabilities</div>
                """,
                unsafe_allow_html=True,
            )

            # Horizontal probability bar chart
            prob_df = pd.DataFrame(
                {
                    "Severity": st.session_state.classes,
                    "Probability": [p * 100 for p in st.session_state.pred_probabilities],
                }
            )

            prob_fig = px.bar(
                prob_df,
                x="Probability",
                y="Severity",
                orientation="h",
                text="Probability",
                color="Severity",
                color_discrete_map={
                    "Fatal Injury": "#ef4444",
                    "Serious Injury": "#f59e0b",
                    "Slight Injury": "#3b82f6",
                },
            )
            prob_fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                marker_line_color="#334155",
                marker_line_width=1,
            )
            prob_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=160,
                margin=dict(l=10, r=30, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#334155", range=[0, 110], color="#94a3b8"),
                yaxis=dict(color="#f8fafc"),
                showlegend=False,
                font=dict(family="Inter", color="#f8fafc"),
            )
            st.plotly_chart(prob_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECTION C: YOLO ROAD SCENE ANALYSIS
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📷 YOLO Road Scene Analysis")
    st.markdown("Analyze road-scene traffic density and detected obstacles using the trained YOLOv8 model.")

    yolo_file_col, yolo_demo_col = st.columns([3, 1])

    with yolo_file_col:
        uploaded_file = st.file_uploader(
            "Upload Road Scene Image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of the road or traffic environment",
        )

    with yolo_demo_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("🖼️ Load Sample Scene Image", use_container_width=True)

    active_image = None

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
    elif use_sample and os.path.exists(SAMPLE_IMAGE_PATH):
        active_image = Image.open(SAMPLE_IMAGE_PATH)

    if active_image is not None:
        if not yolo_loaded:
            st.error("Error: YOLO model could not be loaded from weights.")
        else:
            # Temporary file for YOLO inference
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                active_image.convert("RGB").save(tmp_img.name)
                tmp_path = tmp_img.name

            try:
                # YOLO Inference
                yolo_results = yolo_model.predict(
                    source=tmp_path,
                    imgsz=416,
                    conf=0.25,
                    device="cpu",
                    verbose=False,
                )
                res = yolo_results[0]

                # Annotated result
                annotated_bgr = res.plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]

                # Count detected classes
                obj_counts = {}
                if res.boxes is not None:
                    for cls_id in res.boxes.cls:
                        c_id = int(cls_id)
                        name = yolo_model.names[c_id]
                        obj_counts[name] = obj_counts.get(name, 0) + 1

                total_objs = sum(obj_counts.values())

                # Normalized YOLO risk score (strictly according to project logic)
                yolo_risk_score = min(total_objs / 10.0, 1.0)

                # Store in session state
                st.session_state.has_yolo_detection = True
                st.session_state.yolo_risk_score = yolo_risk_score
                st.session_state.total_objs = total_objs
                st.session_state.obj_counts = obj_counts
                st.session_state.annotated_image = annotated_rgb

                # Display Images & Detection Breakdown
                yolo_disp_col1, yolo_disp_col2 = st.columns([3, 2])

                with yolo_disp_col1:
                    st.subheader("YOLO Detected Image")
                    st.image(
                        annotated_rgb,
                        caption="YOLOv8 Detection Result with Bounding Boxes",
                        use_container_width=True,
                    )

                with yolo_disp_col2:
                    st.subheader("Detected Road Scene Information")

                    st.markdown(
                        f"""
                        <div class="custom-card">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <div>
                                    <div class="metric-chip-label">Total Detected Objects</div>
                                    <div class="metric-chip-value">{total_objs}</div>
                                </div>
                                <div>
                                    <div class="metric-chip-label">YOLO Risk Score</div>
                                    <div class="metric-chip-value">{yolo_risk_score:.4f}</div>
                                </div>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if obj_counts:
                        st.markdown("**Object Counts Breakdown:**")
                        for item_name, item_cnt in obj_counts.items():
                            st.write(f"• **{item_name}**: {item_cnt}")
                    else:
                        st.info("No specific road scene objects detected.")

                    st.markdown("</div>", unsafe_allow_html=True)

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        st.info("Upload a road-scene image or click 'Load Sample Scene Image' to perform YOLO analysis.")

    # ------------------------------------------------------------
    # SECTION D: OVERALL RISK ASSESSMENT & FUSION
    # ------------------------------------------------------------
    if st.session_state.has_dnn_prediction and st.session_state.has_yolo_detection:
        st.markdown("---")
        st.markdown("## ⚠️ Overall Accident Risk")

        # Project fusion formula
        dnn_score = st.session_state.dnn_risk_score
        yolo_score = st.session_state.yolo_risk_score

        final_risk_score = (0.6 * dnn_score) + (0.4 * yolo_score)

        if final_risk_score < 0.40:
            final_risk_level = "Low"
            banner_class = "risk-banner-low"
            badge_icon = "🟢"
        elif final_risk_score < 0.70:
            final_risk_level = "Medium"
            banner_class = "risk-banner-medium"
            badge_icon = "🟡"
        else:
            final_risk_level = "High"
            banner_class = "risk-banner-high"
            badge_icon = "🔴"

        gauge_col, summary_col = st.columns([1, 1])

        with gauge_col:
            st.markdown(
                """
                <div class="custom-card">
                    <div class="card-title">🎯 Overall Risk Score & Gauge</div>
                """,
                unsafe_allow_html=True,
            )
            gauge_fig = render_risk_gauge(final_risk_score, final_risk_level)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown(
                f"""
                    <div class="{banner_class}">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Calculated Overall Risk Level</div>
                        <div style="font-size: 1.8rem; font-weight: 800; margin-top: 4px;">{badge_icon} {final_risk_level.upper()} RISK</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with summary_col:
            st.markdown(
                """
                <div class="custom-card">
                    <div class="card-title">📋 AI Risk Assessment Summary</div>
                """,
                unsafe_allow_html=True,
            )

            sum_subcol1, sum_subcol2 = st.columns(2)

            with sum_subcol1:
                st.markdown(
                    f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">DNN Assessment</div>
                        <div style="font-weight: 700; color: #38bdf8; font-size: 1.05rem; margin-top: 4px;">{st.session_state.pred_class}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Score: {dnn_score:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with sum_subcol2:
                st.markdown(
                    f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Road Scene Assessment</div>
                        <div style="font-weight: 700; color: #38bdf8; font-size: 1.05rem; margin-top: 4px;">{st.session_state.total_objs} Objects Detected</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Scene Score: {yolo_score:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-top: 10px;">
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin-bottom: 6px;">Risk Fusion Formula:</div>
                    <code style="background: #0f172a; padding: 4px 8px; border-radius: 6px; color: #38bdf8;">Final Risk = (0.6 × DNN Risk) + (0.4 × YOLO Risk)</code>
                    <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                        Combines tabular severity classification with real-time road scene obstacle density.
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Contextual Accident Prevention Advice
        st.markdown("### 🛡️ Accident Prevention Recommendations")
        if final_risk_level == "Low":
            advice_list = [
                "Maintain standard safe driving speeds and lane discipline.",
                "Continue observing traffic signals and road signage.",
                "Keep standard following distance in nominal traffic conditions.",
            ]
            box_type = st.success
        elif final_risk_level == "Medium":
            advice_list = [
                "Reduce vehicle speed and drive cautiously in dense road sections.",
                "Maintain extended buffer distance from surrounding vehicles.",
                "Remain vigilant around pedestrians, two-wheelers, and commercial transit.",
                "Adhere to active speed limits and lane marking protocols.",
            ]
            box_type = st.warning
        else:
            advice_list = [
                "Immediately decelerate to minimum safe operational speed.",
                "Maintain large defensive following distance from ahead vehicles.",
                "Avoid sudden lane changes, harsh braking, and high-risk overtaking.",
                "Exercise extreme caution around mixed traffic and pedestrian zones.",
                "Follow all hazardous condition safety protocols and advisory signs.",
            ]
            box_type = st.error

        for adv in advice_list:
            box_type(f"⚠️ {adv}")

    elif st.session_state.has_dnn_prediction and not st.session_state.has_yolo_detection:
        st.info("💡 Upload or load a road scene image above to complete the combined Overall Accident Risk Assessment.")


# ============================================================
# MODULE 2: MODEL ANALYSIS
# ============================================================
elif app_mode == "📊 Model Analysis":
    st.markdown("## 📊 Model Analysis & Benchmark Comparison")
    st.markdown(
        "Comparative evaluation of machine learning and deep learning algorithms evaluated on the "
        "Road Accident Dataset. Deep Neural Network (DNN) serves as the primary deployed model."
    )

    # Primary Model Highlight
    st.markdown(
        """
        <div class="custom-card" style="border-left: 5px solid #38bdf8;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">Primary Deployed Model</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #ffffff; margin-top: 2px;">Deep Neural Network (DNN)</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                        Multi-layer Perceptron (186 input features → 128 → 64 → 32 → 3 softmax classes)
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <span class="status-badge" style="font-size: 0.95rem; padding: 6px 16px;">
                        ⭐ Primary Model • Test Accuracy: 84.58%
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Authentic experimental metrics from the project
    metrics_data = [
        {
            "Model": "Deep Neural Network (DNN)",
            "Role": "Primary Model",
            "Accuracy": 0.8458,
            "Accuracy_Pct": "84.58%",
            "Weighted_F1": 0.78,
            "Macro_F1": 0.31,
            "Key Specifications": "4-Layer Dense Network (128-64-32-3), Adam Optimizer, Softmax Output",
        },
        {
            "Model": "Logistic Regression",
            "Role": "Linear Baseline",
            "Accuracy": 0.8454,
            "Accuracy_Pct": "84.54%",
            "Weighted_F1": 0.78,
            "Macro_F1": 0.31,
            "Key Specifications": "L2 Regularization, lbfgs solver, max_iter=1000",
        },
        {
            "Model": "Random Forest",
            "Role": "Ensemble Baseline",
            "Accuracy": 0.8438,
            "Accuracy_Pct": "84.38%",
            "Weighted_F1": 0.79,
            "Macro_F1": 0.34,
            "Key Specifications": "200 Estimators, class_weight='balanced', random_state=42",
        },
        {
            "Model": "Support Vector Machine (SVM)",
            "Role": "Non-linear Baseline",
            "Accuracy": 0.6790,
            "Accuracy_Pct": "67.90%",
            "Weighted_F1": 0.71,
            "Macro_F1": 0.37,
            "Key Specifications": "RBF Kernel, class_weight='balanced', random_state=42",
        },
    ]

    metrics_df = pd.DataFrame(metrics_data)

    col_chart, col_summary = st.columns([3, 2])

    with col_chart:
        st.markdown("### 📈 Test Accuracy Comparison")
        bar_fig = px.bar(
            metrics_df,
            x="Model",
            y="Accuracy",
            text="Accuracy_Pct",
            color="Model",
            color_discrete_map={
                "Deep Neural Network (DNN)": "#38bdf8",
                "Logistic Regression": "#64748b",
                "Random Forest": "#10b981",
                "Support Vector Machine (SVM)": "#f59e0b",
            },
        )
        bar_fig.update_traces(
            textposition="outside",
            marker_line_color="#334155",
            marker_line_width=1,
        )
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=20, t=20, b=10),
            yaxis=dict(
                title="Test Accuracy",
                range=[0.5, 0.95],
                gridcolor="#334155",
                tickformat=".0%",
                color="#94a3b8",
            ),
            xaxis=dict(title="", color="#f8fafc"),
            showlegend=False,
            font=dict(family="Inter", color="#f8fafc"),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_summary:
        st.markdown("### 🏆 Architecture Insights")
        st.markdown(
            """
            <div class="custom-card">
                <div style="font-weight: 600; color: #38bdf8; margin-bottom: 8px;">Why DNN is Primary:</div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    • <b>High-Dimensional Representation:</b> Successfully maps 186 one-hot encoded multi-categorical and numerical features.<br>
                    • <b>Calibrated Softmax Probabilities:</b> Produces direct multi-class continuous probabilities essential for downstream YOLO risk fusion.<br>
                    • <b>Explainability Compatibility:</b> Seamlessly integrates with DeepExplainer SHAP to compute feature attribution vectors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Benchmark Metrics Table
    st.markdown("### 📋 Evaluated Benchmark Performance Table")
    st.dataframe(
        metrics_df[["Model", "Role", "Accuracy_Pct", "Weighted_F1", "Macro_F1", "Key Specifications"]],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODULE 3: EXPLAINABILITY (SHAP)
# ============================================================
elif app_mode == "🔍 Explainability":
    st.markdown("## 🔍 Model Explainability (SHAP Analysis)")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) helps understand how the input features influence "
        "the model's accident severity prediction by quantifying the average marginal contribution of each feature."
    )

    if shap_df is not None:
        st.markdown("### 📊 Global Feature Importance")
        st.caption("Mean absolute SHAP value calculated across test samples and severity classes.")

        top_n = st.slider("Select number of top influential features to display:", 5, 25, 12)
        top_shap_df = shap_df.head(top_n).copy()

        # Horizontal Bar Plot
        shap_fig = px.bar(
            top_shap_df[::-1],
            x="Mean_Absolute_SHAP",
            y="Feature",
            orientation="h",
            text="Mean_Absolute_SHAP",
            color="Mean_Absolute_SHAP",
            color_continuous_scale="Viridis",
        )
        shap_fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            marker_line_color="#334155",
            marker_line_width=1,
        )
        shap_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=440,
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(
                title="Mean Absolute SHAP Value",
                gridcolor="#334155",
                color="#94a3b8",
            ),
            yaxis=dict(title="", color="#f8fafc"),
            coloraxis_showscale=False,
            font=dict(family="Inter", color="#f8fafc"),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        # Top Features Interpretation
        st.markdown("### 📌 Top Influential Factors Breakdown")
        top10 = shap_df.head(10)

        cols = st.columns(2)
        for idx, row in top10.iterrows():
            target_col = cols[idx % 2]
            with target_col:
                target_col.markdown(
                    f"""
                    <div class="metric-chip" style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #f8fafc;">{idx + 1}. {row['Feature']}</span>
                            <span style="color: #38bdf8; font-weight: 700; font-size: 0.95rem;">{row['Mean_Absolute_SHAP']:.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("SHAP feature importance data file (`SHAP_feature_importance.csv`) not found.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 8px 0 16px 0;">
        <b>Road Accident Risk Prediction</b> • Deep Neural Network + YOLO Computer Vision + SHAP Explainability<br>
        Developed for Final-Year Engineering Demonstration & Road Safety Analytics
    </div>
    """,
    unsafe_allow_html=True,
)