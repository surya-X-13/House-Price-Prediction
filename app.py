import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

model = None
try:
    model = joblib.load("best_random_forest_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'best_random_forest_model.pkl' not found.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1300px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1629; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 3px; }

/* ── Hero Section ── */
.hero {
    background: linear-gradient(135deg, #0f1629 0%, #1a1040 40%, #0d1f3c 100%);
    border-radius: 24px;
    padding: 3.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 25px 80px rgba(79, 70, 229, 0.15);
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.8rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    max-width: 500px;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: center;
}
.hero-stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1.2rem 0;
}
.section-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(99,102,241,0.3), transparent);
}

/* ── Glass Cards – target Streamlit columns natively ── */
[data-testid="stColumn"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1rem !important;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stColumn"]:hover {
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 8px 32px rgba(79,70,229,0.12);
}

/* ── Streamlit inputs – restyle ── */
.stNumberInput > label,
.stSelectbox > label,
.stSlider > label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    margin-bottom: 4px !important;
}
.stNumberInput input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.2) !important;
}
.stSlider > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 30px rgba(79,70,229,0.4) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(79,70,229,0.55) !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #3b82f6 100%) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(79,70,229,0.1) 100%);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
    margin-top: 1.5rem;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #34d399;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-value {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, #34d399, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.result-note {
    font-size: 0.8rem;
    color: #64748b;
}
.result-glow {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,22,41,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(99,102,241,0.3), transparent);
    margin: 2rem 0;
}

/* ── Info chips ── */
.chip {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 AI-Powered Real Estate</div>
    <div class="hero-title">House Price<br>Predictor</div>
    <div class="hero-sub">Get accurate property valuations in seconds using our trained Random Forest model.</div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">19</div>
            <div class="hero-stat-label">Features</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">Random Forest</div>
            <div class="hero-stat-label">Algorithm</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">~1s</div>
            <div class="hero-stat-label">Prediction Time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Section 1 – Property Basics ───────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">🏡</div>
    <div class="section-title">Property Basics</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms  = st.number_input("🛏 Bedrooms",  min_value=1, max_value=20, value=3)
        bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)
        floors    = st.number_input("🏢 Floors",    min_value=1, max_value=5,  value=1)

    with col2:
        living_area = st.number_input("📐 Living Area (sq ft)",         min_value=200, value=1500)
        lot_area    = st.number_input("🌿 Lot Area (sq ft)",            min_value=500, value=4000)
        area_house  = st.number_input("🏠 House Area (excl. basement)", min_value=200, value=1200)

    with col3:
        basement   = st.number_input("🏚 Basement Area (sq ft)",       min_value=0, value=300)
        built_year = st.number_input("📅 Year Built",                   min_value=1900, max_value=2025, value=2005)
        renovation = st.number_input("🔨 Renovation Year (0 = none)",  min_value=0,    max_value=2025, value=0)

# ── Section 2 – Quality & Features ────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">⭐</div>
    <div class="section-title">Quality & Features</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col4, col5 = st.columns(2)

    with col4:
        waterfront = st.selectbox("🌊 Waterfront Present", options=[0, 1],
                                   format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
        views     = st.slider("👁 Number of Views",       0, 5, 0)
        condition = st.slider("🔧 Condition of House (1–5)", 1, 5, 3)

    with col5:
        grade   = st.slider("🏅 Grade of House (1–13)", 1, 13, 7)
        schools = st.number_input("🏫 Schools Nearby",              min_value=0, value=2)
        airport = st.number_input("✈ Distance from Airport (km)",  min_value=0.0, value=15.0, step=0.1)

# ── Section 3 – Location & Renovation ─────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">📍</div>
    <div class="section-title">Location & Renovation Info</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col6, col7, col8, col9 = st.columns(4)

    with col6:
        latitude  = st.number_input("🌐 Latitude",  value=28.61, format="%.4f")
    with col7:
        longitude = st.number_input("🌐 Longitude", value=77.20, format="%.4f")
    with col8:
        living_area_renov = st.number_input("📐 Living Area (Renovated)", value=1500)
    with col9:
        lot_area_renov    = st.number_input("🌿 Lot Area (Renovated)",    value=4000)

# ── Predict Button ─────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


predict = st.button("🔮  Predict House Price")

# ── Prediction Logic ───────────────────────────────────────────
if predict:
    if model is None:
        st.error("❌ Model not loaded. Please add 'best_random_forest_model.pkl'.")
    else:
        input_data = np.array([[
            bedrooms, bathrooms, living_area, lot_area, floors,
            waterfront, views, condition, grade, area_house,
            basement, built_year, renovation,
            latitude, longitude,
            living_area_renov, lot_area_renov,
            schools, airport
        ]], dtype=np.float64)

        if hasattr(model, 'n_features_in_') and input_data.shape[1] != model.n_features_in_:
            st.error(
                f"⚠️ Feature mismatch: model expects {model.n_features_in_} "
                f"but {input_data.shape[1]} were provided."
            )
        else:
            try:
                prediction = model.predict(input_data)
                price = prediction[0]

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-glow">🏡</div>
                    <div class="result-label">Estimated Property Value</div>
                    <div class="result-value">₹ {price:,.0f}</div>
                    <div class="result-note">Powered by Random Forest · 19 features analyzed</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="text-align:center; margin-top:1rem;">
                    <span class="chip">🛏 {bedrooms} Beds</span>
                    <span class="chip">🚿 {bathrooms} Baths</span>
                    <span class="chip">📐 {living_area} sq ft</span>
                    <span class="chip">🏢 {floors} Floor(s)</span>
                    <span class="chip">🏅 Grade {grade}</span>
                    <span class="chip">🔧 Condition {condition}/5</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1rem 0; color: #334155; font-size: 0.8rem;">
    Built with ❤️ using <strong style="color:#6366f1">Streamlit</strong> &amp;
    <strong style="color:#34d399">scikit-learn</strong>
    <strong style="color:#6366f1">By Suryadip</strong>
</div>
""", unsafe_allow_html=True)