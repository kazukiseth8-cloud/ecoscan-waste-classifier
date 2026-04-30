"""
Intelligent Waste Classification System — Mobile App
Kazuki Seth | B.Tech CSE (AI & ML) Sem IV | Graphic Era Hill University
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

WASTE_DATA = {
    'cardboard': {
        'emoji': '📦',
        'color': '#D4845A',
        'bg': '#2D1F14',
        'label': 'Cardboard',
        'recyclable': True,
        'bin': 'Blue Recycling Bin',
        'tip': 'Flatten all boxes before recycling. Remove tape and staples if possible.',
        'facts': ['Cardboard can be recycled 5–7 times', '1 ton of recycled cardboard saves 17 trees', 'Takes 2 months to decompose if composted'],
        'do': ['Flatten boxes completely', 'Remove bubble wrap & foam', 'Keep it dry — wet cardboard is rejected'],
        'dont': ["Don't recycle if greasy (e.g. pizza boxes)", "Don't include wax-coated cardboard"],
    },
    'glass': {
        'emoji': '🫙',
        'color': '#5AB8D4',
        'bg': '#0F2028',
        'label': 'Glass',
        'recyclable': True,
        'bin': 'Glass Recycling Bin',
        'tip': 'Rinse thoroughly. Never mix broken glass with regular recycling.',
        'facts': ['Glass can be recycled indefinitely without quality loss', 'Recycling 1 glass bottle saves enough energy to power a light bulb for 4 hours', 'Takes over 1 million years to decompose in landfill'],
        'do': ['Rinse out food and liquid residue', 'Remove metal lids separately', 'Sort by colour where required'],
        'dont': ["Don't include broken glass in regular bins", "Don't recycle mirrors, windows, or Pyrex"],
    },
    'metal': {
        'emoji': '🥫',
        'color': '#A8A8B8',
        'bg': '#1A1A22',
        'label': 'Metal',
        'recyclable': True,
        'bin': 'Metal / Mixed Recycling Bin',
        'tip': 'Rinse cans. Aluminium cans are especially valuable — always recycle them.',
        'facts': ['Aluminium can be recycled in 60 days and back on the shelf', 'Recycling aluminium uses 95% less energy than making new aluminium', 'Steel is the most recycled material on Earth'],
        'do': ['Rinse cans of food residue', 'Crush cans to save space', 'Include foil trays and bottle caps'],
        'dont': ["Don't include paint tins with dried paint", "Don't recycle aerosol cans that aren't empty"],
    },
    'paper': {
        'emoji': '📄',
        'color': '#C8D46A',
        'bg': '#1E2208',
        'label': 'Paper',
        'recyclable': True,
        'bin': 'Paper / Blue Recycling Bin',
        'tip': 'Keep paper dry. Shredded paper should be bagged separately.',
        'facts': ['Paper can be recycled 4–6 times before fibres degrade', 'Recycling 1 tonne of paper saves 13 trees and 2.5 barrels of oil', 'Paper makes up ~26% of landfill waste globally'],
        'do': ['Include newspapers, magazines, envelopes', 'Remove plastic windows from envelopes', 'Bundle or bag shredded paper'],
        'dont': ["Don't include wax paper or laminated paper", "Don't recycle paper soiled with food"],
    },
    'plastic': {
        'emoji': '🧴',
        'color': '#9B7DD4',
        'bg': '#1A0F2E',
        'label': 'Plastic',
        'recyclable': True,
        'bin': 'Check Recycling Number First',
        'tip': 'Check the number in the triangle symbol. #1 (PET) and #2 (HDPE) are most recyclable.',
        'facts': ['Only 9% of all plastic ever produced has been recycled', 'Plastic takes 400–1000 years to decompose', 'Over 8 million tonnes of plastic enter oceans every year'],
        'do': ['Rinse containers before recycling', 'Check the resin code (number in triangle)', 'Recycle #1 PET and #2 HDPE freely'],
        'dont': ["Don't recycle black plastic (hard to sort)", "Don't include plastic bags in kerbside bins — take to store drop-off"],
    },
    'trash': {
        'emoji': '🗑️',
        'color': '#E05555',
        'bg': '#280F0F',
        'label': 'General Waste',
        'recyclable': False,
        'bin': 'General Waste Bin (Landfill)',
        'tip': 'This item cannot be recycled through standard streams. Dispose responsibly.',
        'facts': ['General waste in landfill produces methane — a potent greenhouse gas', 'Reducing waste at source is always better than disposal', 'Some "trash" items can be taken to specialist recycling points'],
        'do': ['Bag securely to prevent littering', 'Check if local specialist recycling applies', 'Consider if the item can be repaired or reused first'],
        'dont': ["Don't contaminate recycling bins with non-recyclables", "Don't dump illegally — always use designated bins"],
    }
}

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & Base ─────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #0A0F0A;
        color: #E8EDE8;
    }
    .block-container {
        max-width: 480px !important;
        padding: 1rem 1.2rem 4rem !important;
        margin: 0 auto;
    }

    /* ── Header ───────────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        animation: fadeDown 0.6s ease both;
    }
    .app-header .logo {
        font-size: 3.2rem;
        display: block;
        margin-bottom: 0.3rem;
        animation: pulse-logo 3s ease-in-out infinite;
    }
    @keyframes pulse-logo {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.06); }
    }
    .app-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #5DCC7A;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #6B7F6B;
        font-size: 0.82rem;
        margin-top: 0.3rem;
        font-weight: 300;
    }

    /* ── Input Tabs ───────────────────────────────────────────────── */
    .tab-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .tab-btn {
        flex: 1;
        padding: 0.65rem;
        border-radius: 10px;
        border: 1.5px solid #2A3A2A;
        background: #111811;
        color: #5DCC7A;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.82rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .tab-btn.active {
        background: #5DCC7A;
        color: #0A0F0A;
        border-color: #5DCC7A;
    }

    /* ── Upload Zone ──────────────────────────────────────────────── */
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #2A3A2A !important;
        border-radius: 16px !important;
        background: #111811 !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #5DCC7A !important;
    }
    [data-testid="stFileUploader"] label {
        color: #5DCC7A !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── Camera ───────────────────────────────────────────────────── */
    [data-testid="stCameraInput"] > div {
        border: 2px dashed #2A3A2A !important;
        border-radius: 16px !important;
        background: #111811 !important;
    }

    /* ── Analyse Button ───────────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #5DCC7A, #2E8B45) !important;
        color: #0A0F0A !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.85rem !important;
        border-radius: 14px !important;
        border: none !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 20px rgba(93, 204, 122, 0.25) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(93, 204, 122, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Result Card ──────────────────────────────────────────────── */
    .result-card {
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        animation: slideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
        pointer-events: none;
    }
    .result-emoji {
        font-size: 3.5rem;
        display: block;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: bounceIn 0.6s 0.2s both;
    }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .result-bin {
        text-align: center;
        font-size: 0.82rem;
        opacity: 0.75;
        margin-top: 0.3rem;
        font-weight: 300;
    }
    .recyclable-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Syne', sans-serif;
        margin-top: 0.6rem;
    }

    /* ── Confidence Bar ───────────────────────────────────────────── */
    .conf-section {
        margin: 1rem 0;
        animation: fadeUp 0.5s 0.3s both;
    }
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #6B7F6B;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    .conf-track {
        height: 8px;
        background: #1A2A1A;
        border-radius: 4px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        animation: growBar 1s 0.5s both;
    }
    @keyframes growBar {
        from { width: 0 !important; }
    }

    /* ── All-class breakdown ──────────────────────────────────────── */
    .breakdown-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: #4A5E4A;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1.4rem 0 0.7rem;
        animation: fadeUp 0.5s 0.4s both;
    }
    .breakdown-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.5rem;
        animation: fadeUp 0.4s both;
    }
    .breakdown-name {
        width: 75px;
        font-size: 0.78rem;
        color: #8A9E8A;
        flex-shrink: 0;
    }
    .breakdown-bar-track {
        flex: 1;
        height: 6px;
        background: #1A2A1A;
        border-radius: 3px;
        overflow: hidden;
    }
    .breakdown-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    .breakdown-pct {
        width: 38px;
        text-align: right;
        font-size: 0.76rem;
        color: #6B7F6B;
        font-variant-numeric: tabular-nums;
    }

    /* ── Info Sections ────────────────────────────────────────────── */
    .info-block {
        background: #111811;
        border: 1px solid #1E2E1E;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        animation: fadeUp 0.5s both;
    }
    .info-block h4 {
        font-family: 'Syne', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #4A5E4A;
        margin: 0 0 0.5rem;
    }
    .info-block ul {
        margin: 0;
        padding-left: 1rem;
        color: #A8C0A8;
        font-size: 0.84rem;
        line-height: 1.7;
    }
    .info-block .fact-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
        font-size: 0.83rem;
        color: #A8C0A8;
        line-height: 1.5;
    }
    .fact-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        margin-top: 0.45rem;
        flex-shrink: 0;
    }

    /* ── Low confidence warning ───────────────────────────────────── */
    .low-conf-warn {
        background: #2A1E0A;
        border: 1px solid #6B4E1A;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        font-size: 0.83rem;
        color: #D4A847;
        margin-bottom: 0.8rem;
        animation: fadeUp 0.4s both;
    }

    /* ── Footer ───────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 0.5rem;
        color: #3A4E3A;
        font-size: 0.72rem;
        line-height: 1.8;
    }

    /* ── Animations ───────────────────────────────────────────────── */
    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px) scale(0.96); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes bounceIn {
        0%   { opacity: 0; transform: scale(0.5); }
        70%  { transform: scale(1.12); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* ── Streamlit overrides ──────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stRadio > div { display: none; }
    [data-testid="stImage"] img {
        border-radius: 14px;
        width: 100%;
    }
    div[data-testid="stSpinner"] {
        color: #5DCC7A !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Update this path to wherever waste_model_v2.h5 lives
    # On Colab: /content/drive/MyDrive/DL_Waste_Project/models/waste_model_v2.h5
    # Locally:  waste_model_v2.h5 (same folder as app_mobile.py)
    try:
        model = tf.keras.models.load_model("waste_model_v2.h5")
        return model
    except Exception:
        try:
            model = tf.keras.models.load_model(
                "/content/drive/MyDrive/DL_Waste_Project/models/waste_model_v2.h5"
            )
            return model
        except Exception as e:
            return None

# ── PREDICTION ────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model) -> tuple:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds

# ── RESULT CARD HTML ──────────────────────────────────────────────────────────
def result_card_html(cls: str, confidence: float) -> str:
    d = WASTE_DATA[cls]
    badge_style = (
        f"background: rgba(93,204,122,0.15); color: #5DCC7A;"
        if d['recyclable'] else
        "background: rgba(224,85,85,0.15); color: #E05555;"
    )
    badge_text = "✓ Recyclable" if d['recyclable'] else "✗ General Waste"
    return f"""
    <div class="result-card" style="background:{d['bg']}; border: 1.5px solid {d['color']}30;">
        <span class="result-emoji">{d['emoji']}</span>
        <p class="result-label" style="color:{d['color']};">{d['label']}</p>
        <p class="result-bin">🗂️ {d['bin']}</p>
        <div style="text-align:center;">
            <span class="recyclable-badge" style="{badge_style}">{badge_text}</span>
        </div>
    </div>
    """

def confidence_bar_html(confidence: float, color: str) -> str:
    level = "High" if confidence > 0.75 else "Medium" if confidence > 0.55 else "Low"
    pct   = confidence * 100
    return f"""
    <div class="conf-section">
        <div class="conf-label">
            <span>Confidence — {level}</span>
            <span style="color:{color}; font-weight:700;">{pct:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{pct}%; background:linear-gradient(90deg,{color}80,{color});"></div>
        </div>
    </div>
    """

def breakdown_html(preds, predicted_cls: str) -> str:
    html = '<p class="breakdown-title">All Classes</p>'
    sorted_pairs = sorted(zip(CLASS_NAMES, preds), key=lambda x: x[1], reverse=True)
    for i, (name, prob) in enumerate(sorted_pairs):
        d    = WASTE_DATA[name]
        pct  = prob * 100
        bold = "font-weight:700; color:#E8EDE8;" if name == predicted_cls else ""
        delay= f"animation-delay:{0.05*i}s"
        html += f"""
        <div class="breakdown-row" style="{delay}">
            <span class="breakdown-name" style="{bold}">{d['emoji']} {name}</span>
            <div class="breakdown-bar-track">
                <div class="breakdown-bar-fill"
                     style="width:{pct}%; background:{d['color']}; opacity:{'1' if name==predicted_cls else '0.45'};"></div>
            </div>
            <span class="breakdown-pct">{pct:.1f}%</span>
        </div>
        """
    return html

def info_blocks_html(cls: str) -> str:
    d    = WASTE_DATA[cls]
    col  = d['color']
    html = ""

    # Tip
    html += f"""
    <div class="info-block" style="border-left: 3px solid {col};">
        <h4>💡 Disposal Tip</h4>
        <p style="color:#A8C0A8; font-size:0.84rem; margin:0; line-height:1.6;">{d['tip']}</p>
    </div>
    """

    # Do / Don't
    dos   = "".join(f"<li>{x}</li>" for x in d['do'])
    donts = "".join(f"<li>{x}</li>" for x in d['dont'])
    html += f"""
    <div class="info-block">
        <h4 style="color:#5DCC7A;">✓ Do</h4>
        <ul>{dos}</ul>
        <h4 style="color:#E05555; margin-top:0.7rem;">✗ Don't</h4>
        <ul>{donts}</ul>
    </div>
    """

    # Facts
    facts = "".join(
        f'<div class="fact-item"><span class="fact-dot" style="background:{col};"></span><span>{f}</span></div>'
        for f in d['facts']
    )
    html += f"""
    <div class="info-block">
        <h4>🌍 Did You Know?</h4>
        {facts}
    </div>
    """
    return html

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Header
    st.markdown("""
    <div class="app-header">
        <span class="logo">♻️</span>
        <h1>EcoScan AI</h1>
        <p>Intelligent Waste Classification · MobileNetV2 · 78.7% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.error("❌ Model not found. Check the path in `load_model()` and try again.")
        st.info("Expected: `waste_model_v2.h5` in the same folder, or at the Colab Drive path.")
        return

    # Input mode toggle
    mode = st.radio("Input mode", ["📸 Camera", "📁 Upload"], horizontal=True)

    img_pil = None

    if mode == "📸 Camera":
        cam_img = st.camera_input("Point camera at waste item")
        if cam_img:
            img_pil = Image.open(cam_img)

    else:
        uploaded = st.file_uploader(
            "Choose a waste image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            img_pil = Image.open(uploaded)

    # Analyse button
    if img_pil is not None:
        st.image(img_pil, use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Analyse Waste"):
            with st.spinner("Scanning..."):
                time.sleep(0.3)  # brief pause for animation feel
                cls, confidence, all_preds = predict(img_pil, model)

            data = WASTE_DATA[cls]

            # Low confidence warning
            if confidence < 0.55:
                st.markdown("""
                <div class="low-conf-warn">
                    ⚠️ <strong>Low Confidence</strong> — The model is unsure about this image.
                    Try a clearer, better-lit photo of a single item.
                </div>
                """, unsafe_allow_html=True)

            # Result card
            st.markdown(result_card_html(cls, confidence), unsafe_allow_html=True)

            # Confidence bar
            st.markdown(confidence_bar_html(confidence, data['color']), unsafe_allow_html=True)

            # All-class breakdown
            st.markdown(breakdown_html(all_preds, cls), unsafe_allow_html=True)

            # Divider
            st.markdown("<hr style='border:none; border-top:1px solid #1E2E1E; margin:1.2rem 0;'>", unsafe_allow_html=True)

            # Info blocks
            st.markdown(info_blocks_html(cls), unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
        EcoScan AI · B.Tech CSE (AI & ML) Sem IV<br>
        Graphic Era Hill University · April 2026<br>
        MobileNetV2 Transfer Learning · TrashNet Dataset
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
