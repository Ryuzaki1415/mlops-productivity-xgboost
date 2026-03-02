import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import httpx
import plotly.graph_objects as go
import pandas as pd
import time
from api.api_config import FASTAPI_BASE_URL,OLLAMA_MODEL 

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Productivity Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2rem; font-weight: 700; color: #f0f0f0; }
    .sub-title    { font-size: 1rem; color: #aaa; margin-bottom: 2rem; }
    .insight-box  {
        background: #1e1e2e;
        border-left: 4px solid #4c6ef5;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e0e0e0;
    }
    .score-low      { color: #ff6b6b; font-weight: 700; font-size: 1.1rem; }
    .score-moderate { color: #ffa94d; font-weight: 700; font-size: 1.1rem; }
    .score-high     { color: #69db7c; font-weight: 700; font-size: 1.1rem; }

    /* Progress step rows */
    .step-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.45rem 0;
        font-size: 0.95rem;
        color: #ccc;
    }
    .step-done    { color: #69db7c; font-size: 1.1rem; }
    .step-active  { color: #ffa94d; font-size: 1.1rem; }
    .step-pending { color: #555;    font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#e0e0e0", size=12),
)


def make_gauge(score: float, category: str) -> go.Figure:
    color_map = {"Low": "#ff6b6b", "Moderate": "#ffa94d", "High": "#69db7c"}
    color     = color_map.get(category, "#4c6ef5")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#aaa",
                "tickfont": {"color": "#ccc"},
            },
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#444",
            "steps": [
                {"range": [0,  40],  "color": "#3a1a1a"},
                {"range": [40, 70],  "color": "#3a2e1a"},
                {"range": [70, 100], "color": "#1a3a1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={
            "text": f"Productivity Score<br>"
                    f"<span style='font-size:0.85em; color:{color}'>{category}</span>",
            "font": {"color": "#e0e0e0", "size": 16},
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=10, l=20, r=20),
        **CHART_LAYOUT,
    )
    return fig


def make_shap_bar(contributors: list[dict]) -> go.Figure:
    features  = [c["feature"] for c in contributors]
    shap_vals = [c["shap_value"] for c in contributors]
    colors    = ["#69db7c" if v > 0 else "#ff6b6b" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
        marker_line_color="#444",
        marker_line_width=0.5,
        text=[f"{v:+.2f}" for v in shap_vals],
        textposition="outside",
        textfont={"color": "#e0e0e0", "size": 12},
    ))
    fig.update_layout(
        title=dict(
            text="Top 5 Feature Contributions (SHAP)",
            font=dict(color="#e0e0e0", size=14)
        ),
        xaxis=dict(
            title="Impact on Productivity Score",
            title_font=dict(color="#aaa"),
            tickfont=dict(color="#ccc"),
            gridcolor="#2a2a3e",
            zerolinecolor="#555",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#e0e0e0", size=11),
            gridcolor="#2a2a3e",
        ),
        height=300,
        margin=dict(t=50, b=20, l=10, r=60),
        **CHART_LAYOUT,
    )
    return fig


def call_predict(payload: dict) -> dict | None:
    try:
        with httpx.Client(timeout=120.0) as client:

            # 1️⃣ Submit task
            r = client.post(f"{FASTAPI_BASE_URL}/predict", json=payload)
            r.raise_for_status()
            response = r.json()

            # If cached result returned immediately
            if "score" in response:
                return response

            task_id = response.get("task_id")

            if not task_id:
                st.error("No task_id returned from API.")
                return None

            # 2️⃣ Poll result endpoint
            while True:
                r = client.get(f"{FASTAPI_BASE_URL}/result/{task_id}")

                if r.status_code == 202:
                    time.sleep(2)
                    continue

                r.raise_for_status()

                return r.json()

    except httpx.ConnectError:
        st.error("❌ Cannot connect to the API.")
    except httpx.HTTPStatusError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

    return None

# ── Progress stepper ───────────────────────────────────────────────────────────
STEPS = [
    ("🔢", "Validating inputs"),
    ("⚙️",  "Running feature engineering"),
    ("🌲", "XGBoost prediction"),
    ("🔍", "Computing SHAP values"),
    ("🤖", f"{OLLAMA_MODEL} generating insight"),
]

def render_steps(current: int, placeholder) -> None:
    """Renders step list. current = active step index. current >= len = all done."""
    rows = ""
    for i, (icon, label) in enumerate(STEPS):
        if i < current:
            rows += f'<div class="step-row"><span class="step-done">✅</span> {label}</div>'
        elif i == current:
            rows += f'<div class="step-row"><span class="step-active">⏳</span> <b>{label}...</b></div>'
        else:
            rows += f'<div class="step-row"><span class="step-pending">○</span> <span style="color:#555">{label}</span></div>'
    placeholder.markdown(rows, unsafe_allow_html=True)


# ── Sidebar — Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧑 User Profile")

    gender     = st.selectbox("Gender",      ["Male", "Female", "Non-binary"])
    occupation = st.selectbox("Occupation",  ["Engineer", "Student", "Manager", "Healthcare", "Creative", "Sales"])
    device     = st.selectbox("Device Type", ["Android", "iOS", "Both"])

    st.markdown("---")
    st.markdown("## 📱 Screen Time")

    daily_phone    = st.slider("Daily Phone Hours",           0.0, 16.0, 6.0, 0.5)
    social_media   = st.slider("Social Media Hours",          0.0, daily_phone, min(3.0, daily_phone), 0.5)
    weekend_screen = st.slider("Weekend Screen Time (hours)", 0.0, 16.0, 7.0, 0.5)
    app_count      = st.slider("Apps Used Daily",             1,   50,   18)

    st.markdown("---")
    st.markdown("## 😴 Sleep & Stress")

    sleep    = st.slider("Sleep Hours",         3.0, 10.0, 6.5, 0.5)
    stress   = st.slider("Stress Level",        1.0, 10.0, 6.5, 0.5)
    caffeine = st.slider("Caffeine (cups/day)", 0.0,  8.0, 2.0, 0.5)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Productivity", use_container_width=True, type="primary")


# ── Main Panel ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Work Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by XGBoost + ministral · Insight generated locally via Ollama</div>',
    unsafe_allow_html=True,
)

if not predict_btn:
    st.info("👈 Configure your profile in the sidebar and click **Predict Productivity** to begin.")
    st.stop()

# ── Build payload ──────────────────────────────────────────────────────────────
payload = {
    "Daily_Phone_Hours":         daily_phone,
    "Social_Media_Hours":        social_media,
    "Sleep_Hours":               sleep,
    "Stress_Level":              stress,
    "App_Usage_Count":           app_count,
    "Caffeine_Intake_Cups":      caffeine,
    "Weekend_Screen_Time_Hours": weekend_screen,
    "Gender":                    gender,
    "Occupation":                occupation,
    "Device_Type":               device,
}

# ── Progress stepper ───────────────────────────────────────────────────────────
st.markdown("#### ⚙️ Processing")
progress_placeholder = st.empty()
progress_bar         = st.progress(0)

# Animate steps 0–3 locally with short delays (these are fast server-side)
for i in range(4):
    render_steps(i, progress_placeholder)
    progress_bar.progress(int((i / len(STEPS)) * 100))
    time.sleep(0.4)

# Step 4 — LLM — this is the real wait, hold here until response
render_steps(4, progress_placeholder)
progress_bar.progress(80)

result = call_predict(payload)   # ← blocks here until FastAPI + Ollama respond

# All done
progress_bar.progress(100)
render_steps(len(STEPS), progress_placeholder)
time.sleep(0.5)

# Clear progress UI cleanly
progress_placeholder.empty()
progress_bar.empty()
st.markdown("")   # spacer

if result is None:
    st.stop()

# ── Unpack ─────────────────────────────────────────────────────────────────────
score        = result["score"]
category     = result["score_category"]
contributors = result["top_contributors"]
insight      = result["insight"]

st.success(f"✅ Prediction complete — Score: **{score}/100** ({category})")

# ── Results Layout ─────────────────────────────────────────────────────────────
col_gauge, col_shap = st.columns([1, 1], gap="large")

with col_gauge:
    st.plotly_chart(make_gauge(score, category), use_container_width=True)
    css_class = f"score-{category.lower()}"
    st.markdown(
        f'<div class="{css_class}">Category: {category} Productivity</div>',
        unsafe_allow_html=True,
    )

with col_shap:
    st.plotly_chart(make_shap_bar(contributors), use_container_width=True)

st.markdown("---")

# ── LLM Insight ────────────────────────────────────────────────────────────────
# ── LLM Insight ────────────────────────────────────────────────────────────────
st.markdown("### 🤖 AI Insight (ministral 3b)")

if insight.startswith("⚠️"):
    st.warning(insight)
else:
    with st.container(border=True):
        st.markdown(insight)   # ← Streamlit renders markdown natively here

# ── SHAP detail table ──────────────────────────────────────────────────────────
with st.expander("📋 SHAP Contributor Details"):
    df_shap = pd.DataFrame(contributors)
    df_shap["direction"] = df_shap["shap_value"].apply(
        lambda x: "⬆️ Boosts" if x > 0 else "⬇️ Reduces"
    )
    df_shap = df_shap.rename(columns={
        "feature":    "Feature",
        "value":      "Your Value",
        "shap_value": "SHAP Impact",
        "direction":  "Effect",
    })
    st.dataframe(
        df_shap[["Feature", "Your Value", "SHAP Impact", "Effect"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Raw payload debug ──────────────────────────────────────────────────────────
with st.expander("🔧 Raw Input Sent to API"):
    st.json(payload)