import sys
import os
import streamlit as st
import httpx
import plotly.graph_objects as go
import pandas as pd
import time
from api.api_config import FASTAPI_BASE_URL, OLLAMA_MODEL

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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* Hide default streamlit header chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0a0a18 100%);
        border: 1px solid #1e1e3a;
        border-radius: 16px;
        padding: 2rem 2.5rem 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 300px; height: 120px;
        background: radial-gradient(ellipse, rgba(52,211,153,0.08) 0%, transparent 70%);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f0f0f8;
        margin: 0 0 0.3rem;
        line-height: 1.2;
    }
    .hero-title span {
        background: linear-gradient(90deg, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 0.85rem;
        color: #6b7280;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.02em;
    }

    /* ── Idle state card ── */
    .idle-card {
        background: #0d0d1a;
        border: 1px dashed #2a2a4a;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        color: #4b5563;
    }
    .idle-card .icon { font-size: 3rem; margin-bottom: 1rem; }
    .idle-card p { font-size: 0.95rem; margin: 0; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #08080f;
        border-right: 1px solid #1a1a2e;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4c6ef5;
        margin: 1.2rem 0 0.4rem;
        border-bottom: 1px solid #1a1a2e;
        padding-bottom: 0.4rem;
    }

    /* ── Step progress ── */
    .steps-wrap {
        background: #0d0d1a;
        border: 1px solid #1e1e3a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .step-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0;
        font-size: 0.9rem;
        color: #6b7280;
        font-family: 'IBM Plex Mono', monospace;
        transition: color 0.3s;
    }
    .step-done   { color: #34d399; }
    .step-active { color: #fbbf24; animation: pulse-text 1.2s ease-in-out infinite; }
    .step-label-done   { color: #9ca3af; }
    .step-label-active { color: #f9fafb; font-weight: 600; }
    .step-label-pending{ color: #374151; }

    @keyframes pulse-text {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }

    /* ── Score pill ── */
    .score-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 100px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-top: 0.5rem;
    }
    .pill-low      { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .pill-moderate { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .pill-high     { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

    /* ── Insight box ── */
    .insight-wrap {
        background: #0d0d1a;
        border: 1px solid #1e1e3a;
        border-left: 3px solid #818cf8;
        border-radius: 0 12px 12px 0;
        padding: 1.4rem 1.6rem;
        font-size: 0.93rem;
        line-height: 1.75;
        color: #d1d5db;
        box-shadow: 0 0 30px rgba(99,102,241,0.06);
    }
    .insight-header {
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.8rem;
    }

    /* ── Section divider ── */
    .section-label {
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #374151;
        margin: 1.5rem 0 0.8rem;
    }

    /* ── Dataframe overrides ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#0d0d1a",
    plot_bgcolor="#0d0d1a",
    font=dict(color="#e0e0e0", size=12),
)


def make_gauge(score: float, category: str) -> go.Figure:
    color_map = {"Low": "#f87171", "Moderate": "#fbbf24", "High": "#34d399"}
    color = color_map.get(category, "#818cf8")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 40, "color": color, "family": "Syne"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#374151",
                "tickfont": {"color": "#6b7280", "size": 10},
            },
            "bar":  {"color": color, "thickness": 0.2},
            "bgcolor": "#0d0d1a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "#1a0e0e"},
                {"range": [40, 70],  "color": "#1a1608"},
                {"range": [70, 100], "color": "#0a1a12"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={
            "text": "Productivity Score",
            "font": {"color": "#9ca3af", "size": 13, "family": "IBM Plex Mono"},
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=10, l=20, r=20),
        **CHART_LAYOUT,
    )
    return fig


def make_shap_bar(contributors: list[dict]) -> go.Figure:
    features  = [c["feature"] for c in contributors]
    shap_vals = [c["shap_value"] for c in contributors]
    colors    = ["#34d399" if v > 0 else "#f87171" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.9,
        text=[f"{v:+.2f}" for v in shap_vals],
        textposition="outside",
        textfont={"color": "#9ca3af", "size": 11, "family": "IBM Plex Mono"},
    ))
    fig.update_layout(
        title=dict(
            text="Feature Contributions (SHAP)",
            font=dict(color="#9ca3af", size=12, family="IBM Plex Mono")
        ),
        xaxis=dict(
            title=None,
            tickfont=dict(color="#6b7280", size=10, family="IBM Plex Mono"),
            gridcolor="#1a1a2e",
            zerolinecolor="#2a2a4a",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#d1d5db", size=11, family="Syne"),
            gridcolor="#1a1a2e",
        ),
        height=280,
        margin=dict(t=45, b=10, l=10, r=55),
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

            # 2️⃣ Poll with backoff
            wait = 1.5
            while True:
                r = client.get(f"{FASTAPI_BASE_URL}/result/{task_id}")
                if r.status_code == 202:
                    time.sleep(wait)
                    wait = min(wait * 1.4, 5.0)   # cap at 5s
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
    ("⚙️",  "Feature engineering"),
    ("🌲", "XGBoost inference"),
    ("🔍", "SHAP attribution"),
    ("🤖", f"{OLLAMA_MODEL} · generating insight"),
]

def render_steps(current: int, placeholder) -> None:
    rows = ""
    for i, (icon, label) in enumerate(STEPS):
        if i < current:
            rows += (
                f'<div class="step-row">'
                f'<span class="step-done">✓</span>'
                f'<span class="step-label-done">{label}</span>'
                f'</div>'
            )
        elif i == current:
            rows += (
                f'<div class="step-row">'
                f'<span class="step-active">⏳</span>'
                f'<span class="step-label-active">{label}</span>'
                f'</div>'
            )
        else:
            rows += (
                f'<div class="step-row">'
                f'<span style="color:#1f2937">○</span>'
                f'<span class="step-label-pending">{label}</span>'
                f'</div>'
            )
    placeholder.markdown(
        f'<div class="steps-wrap">{rows}</div>',
        unsafe_allow_html=True
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.2rem;font-weight:800;color:#f0f0f8;letter-spacing:-0.01em;'
        'padding-bottom:0.8rem;border-bottom:1px solid #1a1a2e;">⚡ Productivity Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">Profile</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",      ["Male", "Female", "Non-binary"])
    occupation = st.selectbox("Occupation",  ["Engineer", "Student", "Manager", "Healthcare", "Creative", "Sales"])
    device     = st.selectbox("Device Type", ["Android", "iOS", "Both"])

    st.markdown('<div class="sidebar-section">Screen Time</div>', unsafe_allow_html=True)
    daily_phone    = st.slider("Daily Phone Hours",           0.0, 16.0, 6.0, 0.5)
    social_media   = st.slider("Social Media Hours",          0.0, daily_phone, min(3.0, daily_phone), 0.5)
    weekend_screen = st.slider("Weekend Screen Time (hours)", 0.0, 16.0, 7.0, 0.5)
    app_count      = st.slider("Apps Used Daily",             1,   50,   18)

    st.markdown('<div class="sidebar-section">Sleep & Stress</div>', unsafe_allow_html=True)
    sleep    = st.slider("Sleep Hours",         3.0, 10.0, 6.5, 0.5)
    stress   = st.slider("Stress Level",        1.0, 10.0, 6.5, 0.5)
    caffeine = st.slider("Caffeine (cups/day)", 0.0,  8.0, 2.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction →", use_container_width=True, type="primary")


# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Work Productivity <span>Predictor</span></div>
    <div class="hero-sub">XGBoost · SHAP Attribution · {model} · Local Inference</div>
</div>
""".replace("{model}", OLLAMA_MODEL), unsafe_allow_html=True)


# ── Idle state ─────────────────────────────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div class="idle-card">
        <div class="icon">📊</div>
        <p>Configure your profile in the sidebar and click <strong>Run Prediction →</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Payload ────────────────────────────────────────────────────────────────────
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


# ── Progress UI — show steps, then immediately hit the API ─────────────────────
st.markdown('<div class="section-label">Processing</div>', unsafe_allow_html=True)
progress_placeholder = st.empty()
progress_bar         = st.progress(0)

# Show step 0 briefly, then step 4 (LLM), then block on real API call
render_steps(0, progress_placeholder)
progress_bar.progress(10)

render_steps(4, progress_placeholder)
progress_bar.progress(30)

result = call_predict(payload)   # ← only real wait

progress_bar.progress(100)
render_steps(len(STEPS), progress_placeholder)

# Clean up progress UI
progress_placeholder.empty()
progress_bar.empty()

if result is None:
    st.stop()


# ── Unpack ─────────────────────────────────────────────────────────────────────
score        = result["score"]
category     = result["score_category"]
contributors = result["top_contributors"]
insight      = result["insight"]

pill_class = f"pill-{category.lower()}"
dot = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}.get(category, "●")

st.markdown(
    f'<div class="score-pill {pill_class}">{dot} {score}/100 · {category} Productivity</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# ── Charts ─────────────────────────────────────────────────────────────────────
col_gauge, col_shap = st.columns([1, 1], gap="large")

with col_gauge:
    st.plotly_chart(make_gauge(score, category), use_container_width=True)

with col_shap:
    st.plotly_chart(make_shap_bar(contributors), use_container_width=True)


# ── LLM Insight ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">AI Insight</div>', unsafe_allow_html=True)

if insight.startswith("⚠️"):
    st.warning(insight)
else:
    st.markdown(
        f'<div class="insight-wrap">'
        f'<div class="insight-header">🤖 {OLLAMA_MODEL}</div>'
        f'{insight}'
        f'</div>',
        unsafe_allow_html=True
    )


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