import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Cognitive Risk Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    .group-chip {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.35rem;
        color: #111827;
    }
    .demo-chip { background-color: #76aade; }
    .z-chip { background-color: #61d461; }
    .raw-chip { background-color: #e8e848; }

    .feature-label {
        padding: 0.3rem 0.55rem;
        border-radius: 0.6rem;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        display: inline-block;
        color: #111827;
    }
    .feature-demo { background-color: #76aade; }
    .feature-z { background-color: #61d461; }
    .feature-raw { background-color: #e8e848; }

    .small-note {
        color: #4b5563;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


demographics = [
    "age7c", "gender1", "race1c", "educ1", "phx_income7", "curjob7"
]

raw_scores = [
    "craftursmb7c", "craftdremb7c", "dgtformb7c", "dgtbckmb7c",
    "dsymscrmb7c", "trailamb7c", "trailbmb7c_adjusted", "udsverfcmb7c",
    "vegmb7c", "animalsmb7c", "avlt_delayed_totalmb7c",
    "avlt_t1_totalmb7c", "avlt_t6_totalmb7c", "avlt_total_correctmb7c",
    "wrat5mb7c", "mocatotsmb7c", "craftdvrmb7c", "craftvrsmb7c",
    "udsbentcmb7c", "udsbentdmb7c", "avlt_lotmb7c", "avlt_listb_totalmb7c",
    "trailb_ceiling", "casisummb7c"
]

z_scores = [
    "memory_immed_domainmb7c", "memory_delay_domainmb7c",
    "lang_semantic_domainmb7c", "phonemic_domainmb7c",
    "attn_process_domainmb7c", "executive_domainmb7c",
    "visuo_domainmb7c", "lang_phonemic_domainmb7c"
]

reduced_core = [
    "executive_domainmb7c",
    "memory_delay_domainmb7c",
    "trailbmb7c_adjusted",
    "casisummb7c",
    "dsymscrmb7c",
    "mocatotsmb7c"
]

pretty_names = {
    "age7c": "Age",
    "gender1": "Gender",
    "race1c": "Race",
    "educ1": "Education",
    "phx_income7": "Income",
    "curjob7": "Current Job",
    "trailbmb7c_adjusted": "Trail B Adjusted",
    "memory_delay_domainmb7c": "Memory Delay Z-score",
    "executive_domainmb7c": "Executive Z-score",
    "mocatotsmb7c": "MoCA Total",
    "casisummb7c": "CASI Sum",
    "dsymscrmb7c": "Digit Symbol Score",
}


@st.cache_resource
def load_artifacts(mode_key: str):
    model = joblib.load(f"model_{mode_key}.pkl")
    threshold = joblib.load(f"threshold_{mode_key}.pkl")
    features = joblib.load(f"features_{mode_key}.pkl")
    means = joblib.load(f"feature_means_{mode_key}.pkl")
    return model, threshold, features, means


def get_group(feature: str) -> str:
    if feature in demographics:
        return "demo"
    if feature in z_scores:
        return "z"
    return "raw"


def render_feature_input(feature: str, mean_value: float, key: str):
    group = get_group(feature)

    if group == "demo":
        css_class = "feature-demo"
    elif group == "z":
        css_class = "feature-z"
    else:
        css_class = "feature-raw"

    label = pretty_names.get(feature, feature)

    st.markdown(
        f'<div class="feature-label {css_class}">{label}</div>',
        unsafe_allow_html=True
    )

    return st.number_input(
        label,
        value=float(mean_value),
        key=key,
        label_visibility="collapsed"
    )


def render_features(feature_list, means, ncols=3, prefix="main"):
    values = {}
    cols = st.columns(ncols)

    for i, feature in enumerate(feature_list):
        with cols[i % ncols]:
            values[feature] = render_feature_input(
                feature,
                means[feature],
                key=f"{prefix}_{feature}"
            )
    return values


st.title("Cognitive Risk Predictor")
st.markdown(
    '<div class="small-note">Fields are pre-filled with training-set averages. '
    'Blue = demographics, green = raw scores, purple = z-scores.</div>',
    unsafe_allow_html=True
)

chip_col1, chip_col2, chip_col3 = st.columns([1, 1, 1])
with chip_col1:
    st.markdown('<div class="group-chip demo-chip">Demographics</div>', unsafe_allow_html=True)
with chip_col2:
    st.markdown('<div class="group-chip raw-chip">Raw Scores</div>', unsafe_allow_html=True)
with chip_col3:
    st.markdown('<div class="group-chip z-chip">Z-scores</div>', unsafe_allow_html=True)


st.sidebar.header("Model Settings")

feature_mode = st.sidebar.radio(
    "Feature set",
    options=["Full", "Reduced"],
    index=0
)

include_demo = False
if feature_mode == "Reduced":
    include_demo = st.sidebar.checkbox("Include demographics", value=True)

if feature_mode == "Full":
    mode_key = "full"
    ncols = 4
elif include_demo:
    mode_key = "reduced_demo"
    ncols = 3
else:
    mode_key = "reduced_nodemo"
    ncols = 2

model, threshold, features, feature_means = load_artifacts(mode_key)

st.sidebar.markdown(f"**Loaded model:** `{mode_key}`")
st.sidebar.markdown(f"**Number of features:** {len(features)}")
st.sidebar.markdown(f"**Threshold:** {threshold:.3f}")


st.subheader("Patient Features")
inputs = render_features(features, feature_means, ncols=ncols, prefix=mode_key)


predict_col1, predict_col2 = st.columns([1, 3])

with predict_col1:
    predict_clicked = st.button("Predict", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame([inputs])[features]
    input_df = input_df.fillna(feature_means)

    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= threshold)

    st.subheader("Results")

    out1, out2, out3 = st.columns(3)
    out1.metric("Risk Probability", f"{prob:.3f}")
    out2.metric("Threshold", f"{threshold:.3f}")
    out3.metric("Decision", "High Risk" if pred == 1 else "Low Risk")

    if pred == 1:
        st.error("High Risk (Cognitive Impairment Likely)")
    else:
        st.success("Low Risk")