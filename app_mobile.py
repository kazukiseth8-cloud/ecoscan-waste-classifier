import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

CLASS_CONFIG = {
    'cardboard': {
        'emoji': '📦', 'color': '#e8a87c', 'bin': 'Cardboard Recycling Bin',
        'recyclable': True,
        'tip': 'Flatten all boxes. Remove tape and staples before recycling.',
        'do': ['Flatten boxes completely', 'Remove all tape and staples', 'Keep dry — wet cardboard is rejected'],
        'dont': ["Don't recycle greasy pizza boxes", "Don't include wax-coated cardboard"],
        'facts': ['Recycling 1 ton of cardboard saves 17 trees', 'Cardboard can be recycled up to 7 times', 'It takes just 75% of the energy to make recycled cardboard vs new']
    },
    'glass': {
        'emoji': '🫙', 'color': '#7ecdc8', 'bin': 'Glass Recycling Bin',
        'recyclable': True,
        'tip': 'Rinse thoroughly. Never mix broken glass with regular recycling.',
        'do': ['Rinse out food and liquid residue', 'Remove metal lids separately', 'Sort by colour where required'],
        'dont': ["Don't include broken glass in regular bins", "Don't recycle mirrors, windows, or Pyrex"],
        'facts': ['Glass can be recycled indefinitely without quality loss', 'Recycling 1 glass bottle saves enough energy to power a light bulb for 4 hours', 'Takes over 1 million years to decompose in landfill']
    },
    'metal': {
        'emoji': '🥫', 'color': '#a8b8c8', 'bin': 'Metal Recycling Bin',
        'recyclable': True,
        'tip': 'Rinse cans well. Aluminium and steel are both recyclable.',
        'do': ['Rinse cans and tins', 'Crush cans to save space', 'Include aluminium foil if clean'],
        'dont': ["Don't include paint or chemical cans", "Don't mix with general waste"],
        'facts': ['Aluminium can be recycled in as little as 60 days', 'Recycling aluminium uses 95% less energy than making it new', 'Steel is the most recycled material in the world']
    },
    'paper': {
        'emoji': '📄', 'color': '#98c89a', 'bin': 'Paper Recycling Bin',
        'recyclable': True,
        'tip': 'Keep dry. Remove plastic windows from envelopes before recycling.',
        'do': ['Keep paper clean and dry', 'Include newspapers, magazines, cardboard', 'Shred sensitive documents first'],
        'dont': ["Don't recycle greasy or food-soiled paper", "Don't include tissue or paper towels"],
        'facts': ['Paper can be recycled 5-7 times before fibres break down', 'Recycling 1 tonne of paper saves 17 trees', 'Paper makes up about 25% of landfill waste']
    },
    'plastic': {
        'emoji': '🧴', 'color': '#b8a8d8', 'bin': 'Plastic Recycling Bin',
        'recyclable': True,
        'tip': 'Check the recycling number. Rinse containers before disposal.',
        'do': ['Check recycling symbol (1-7)', 'Rinse containers clean', 'Flatten bottles to save space'],
        'dont': ["Don't include plastic bags in kerbside bins", "Don't recycle greasy or dirty plastic"],
        'facts': ['Only 9% of plastic ever made has been recycled', 'Plastic takes 400-1000 years to decompose', 'Recycling 1 plastic bottle saves enough energy to power a laptop for 25 minutes']
    },
    'trash': {
        'emoji': '🗑️', 'color': '#e87c7c', 'bin': 'General Waste Bin',
        'recyclable': False,
        'tip': 'This item cannot be recycled. Dispose in your general waste bin.',
        'do': ['Seal waste in a bag to prevent odour', 'Check local guidelines for bulky items', 'Consider if any parts can be separated for recycling'],
        'dont': ["Don't contaminate recycling with general waste", "Don't dump illegally — use council services"],
        'facts': ['The average person generates 4.5 lbs of trash per day', 'Landfill produces methane — a potent greenhouse gas', 'Reducing consumption is more effective than recycling']
    }
}

CLASS_COLORS = {
    'cardboard': '#e8a87c', 'glass': '#7ecdc8', 'metal': '#a8b8c8',
    'paper': '#98c89a', 'plastic': '#b8a8d8', 'trash': '#e87c7c'
}

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #0a0f0a !important;
    color: #e8f5e8 !important;
}
.stApp { background-color: #0a0f0a; }
#MainMenu, footer, header { visibility: hidden; }

.eco-header { text-align: center; padding: 2rem 0 1rem; animation: fadeInDown 0.6s ease; }
.eco-logo { font-size: 3.5rem; display: block; animation: pulse 2s infinite; }
.eco-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px;
}
.eco-subtitle { color: #6b8f6b; font-size: 0.85rem; letter-spacing: 0.5px; }

.result-card {
    border-radius: 20px; padding: 2rem; text-align: center;
    margin: 1.5rem 0; animation: slideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.result-emoji { font-size: 4rem; display: block; margin-bottom: 0.5rem; }
.result-class { font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0; }
.result-bin { font-size: 0.95rem; opacity: 0.85; margin-bottom: 1rem; }
.badge { display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-recyclable { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
.badge-trash { background: #3a1a1a; color: #f87171; border: 1px solid #f87171; }

.tip-box { background: #0f1e12; border-left: 4px solid #4ade80; border-radius: 0 14px 14px 0; padding: 1.2rem 1.4rem; margin: 1rem 0; }
.tip-title { color: #4ade80; font-size: 0.75rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.5rem; }
.do-box { background: #0f1e12; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 0.5rem 0; }
.dont-box { background: #1e0f0f; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 0.5rem 0; }
.do-title { color: #4ade80; font-weight: 700; margin-bottom: 0.6rem; }
.dont-title { color: #f87171; font-weight: 700; margin-bottom: 0.6rem; }
.fact-box { background: #0d1520; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1rem 0; }
.fact-title { color: #38bdf8; font-size: 0.75rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.6rem; }
.warn-box { background: #1e1600; border: 1px solid #ca8a04; border-radius: 12px; padding: 1rem 1.2rem; color: #fbbf24; margin: 1rem 0; font-size: 0.9rem; }

.section-label { color: #6b8f6b; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }

.stat-row { display: flex; gap: 1rem; margin: 1rem 0; }
.stat-card { flex: 1; background: #0f1a0f; border: 1px solid #1a3a1a; border-radius: 14px; padding: 1rem; text-align: center; }
.stat-num { font-size: 1.8rem; font-weight: 700; color: #4ade80; }
.stat-label { font-size: 0.75rem; color: #6b8f6b; margin-top: 0.2rem; }

.scan-row {
    display: flex; align-items: center; gap: 0.8rem;
    background: #0f1a0f; border: 1px solid #1a3a1a;
    border-radius: 12px; padding: 0.8rem 1rem; margin: 0.5rem 0;
}
.scan-num { color: #2d4a2d; font-size: 0.75rem; font-weight: 600; min-width: 1.5rem; }
.scan-emoji { font-size: 1.4rem; }
.scan-class { font-weight: 600; font-size: 0.95rem; flex: 1; }
.scan-badge { font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 10px; }
.scan-conf { font-size: 0.8rem; color: #6b8f6b; text-align: right; min-width: 3.5rem; }
.scan-time { font-size: 0.72rem; color: #2d4a2d; text-align: right; min-width: 4rem; }

.eco-footer { text-align: center; color: #2d4a2d; font-size: 0.78rem; padding: 2rem 0 1rem; line-height: 1.8; }

.stButton > button {
    background: linear-gradient(135deg, #4ade80, #22d3ee) !important;
    color: #0a0f0a !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4ade80, #22d3ee) !important;
}

@keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-header">
    <span class="eco-logo">♻️</span>
    <div class="eco-title">EcoScan AI</div>
    <div class="eco-subtitle">Intelligent Waste Classification · MobileNetV2 · 78.7% accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    paths = [
        "waste_model_v2.keras",
        "waste_model_v2.h5",
        "/content/drive/MyDrive/DL_Waste_Project/models/waste_model_v2.keras",
        "/content/drive/MyDrive/DL_Waste_Project/models/waste_model_v2.h5",
    ]
    for path in paths:
        try:
            m = tf.keras.models.load_model(path, compile=False)
            m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return m
        except Exception:
            continue
    return None

with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.markdown("""
    <div style="background:#1e0f0f;border:1px solid #f87171;border-radius:12px;padding:1.2rem;color:#f87171;margin:1rem 0;">
        ✗ <strong>Model not found.</strong> Ensure <code>waste_model_v2.keras</code> is in the repo root.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── IMAGE INPUT ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])
with tab1:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with tab2:
    camera_image = st.camera_input("")

image_source = uploaded_file or camera_image

if image_source:
    image = Image.open(image_source).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🔍 Analyse Waste", use_container_width=True):

        img_array = np.expand_dims(np.array(image.resize((224, 224))) / 255.0, axis=0)

        with st.spinner("Classifying..."):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(predictions[predicted_idx])
        cfg             = CLASS_CONFIG[predicted_class]
        timestamp       = datetime.now().strftime("%H:%M:%S")

        # Save to session history
        st.session_state.history.append({
            'class':      predicted_class,
            'confidence': confidence,
            'time':       timestamp,
            'recyclable': cfg['recyclable']
        })

        # ── LOW CONFIDENCE WARNING ────────────────────────────────────────────
        if confidence < 0.55:
            st.markdown('<div class="warn-box">⚠️ <strong>Low Confidence</strong> — Try a clearer, better-lit photo of a single item.</div>', unsafe_allow_html=True)

        # ── RESULT CARD ───────────────────────────────────────────────────────
        badge_cls  = "badge-recyclable" if cfg['recyclable'] else "badge-trash"
        badge_text = "✓ Recyclable"     if cfg['recyclable'] else "✗ Not Recyclable"
        st.markdown(f"""
        <div class="result-card" style="background:linear-gradient(135deg,#0f1a0f,#0d1520);border:1px solid {cfg['color']}33;">
            <span class="result-emoji">{cfg['emoji']}</span>
            <div class="result-class" style="color:{cfg['color']};">{predicted_class.capitalize()}</div>
            <div class="result-bin">🗂️ {cfg['bin']}</div>
            <span class="badge {badge_cls}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE BAR ────────────────────────────────────────────────────
        conf_label = "High Confidence" if confidence >= 0.75 else ("Medium Confidence" if confidence >= 0.55 else "Low Confidence")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.caption(f"Confidence — {conf_label}")
        with col_r:
            st.caption(f"**{confidence*100:.1f}%**")
        st.progress(float(confidence))

        # ── ALL CLASSES BREAKDOWN ─────────────────────────────────────────────
        st.markdown('<div class="section-label">All Classes</div>', unsafe_allow_html=True)
        for cls, prob in sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True):
            is_top = cls == predicted_class
            c1, c2, c3 = st.columns([2, 6, 1])
            with c1:
                st.markdown(
                    f"<span style='color:{'#4ade80' if is_top else '#6b8f6b'};"
                    f"font-weight:{'700' if is_top else '400'};font-size:0.85rem;'>"
                    f"{CLASS_CONFIG[cls]['emoji']} {cls}</span>",
                    unsafe_allow_html=True
                )
            with c2:
                st.progress(float(prob))
            with c3:
                st.caption(f"{prob*100:.1f}%")

        # ── DISPOSAL TIP ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="tip-box">
            <div class="tip-title">💡 Disposal Tip</div>
            <div style="color:#c8e6c8;">{cfg['tip']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── DO / DON'T ────────────────────────────────────────────────────────
        col_do, col_dont = st.columns(2)
        with col_do:
            items = "".join([f"<div style='color:#c8e6c8;margin:0.3rem 0;font-size:0.88rem;'>✓ {i}</div>" for i in cfg['do']])
            st.markdown(f'<div class="do-box"><div class="do-title">✓ DO</div>{items}</div>', unsafe_allow_html=True)
        with col_dont:
            items = "".join([f"<div style='color:#f4c4c4;margin:0.3rem 0;font-size:0.88rem;'>{i}</div>" for i in cfg['dont']])
            st.markdown(f'<div class="dont-box"><div class="dont-title">✗ DON\'T</div>{items}</div>', unsafe_allow_html=True)

        # ── DID YOU KNOW ──────────────────────────────────────────────────────
        facts = "".join([f"<div style='color:#94b8d4;margin:0.4rem 0;font-size:0.9rem;'>• {f}</div>" for f in cfg['facts']])
        st.markdown(f'<div class="fact-box"><div class="fact-title">🌍 Did You Know?</div>{facts}</div>', unsafe_allow_html=True)


# ── SESSION STATISTICS ────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-label">📊 Session Statistics</div>', unsafe_allow_html=True)

    history      = st.session_state.history
    total_scans  = len(history)
    avg_conf     = np.mean([h['confidence'] for h in history]) * 100
    recyclable_n = sum(1 for h in history if h['recyclable'])

    # ── STAT CARDS ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">{total_scans}</div>
            <div class="stat-label">Items Scanned</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{avg_conf:.1f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{recyclable_n}</div>
            <div class="stat-label">Recyclable</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PIE CHART ─────────────────────────────────────────────────────────────
    class_counts = {c: 0 for c in CLASS_NAMES}
    for h in history:
        class_counts[h['class']] += 1
    active = {k: v for k, v in class_counts.items() if v > 0}

    if len(active) >= 1:
        st.markdown('<div class="section-label">Items Classified by Type</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0f0a')
        ax.set_facecolor('#0a0f0a')
        wedges, texts, autotexts = ax.pie(
            list(active.values()),
            labels=[f"{CLASS_CONFIG[c]['emoji']} {c}" for c in active],
            colors=[CLASS_COLORS[c] for c in active],
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=2, edgecolor='#0a0f0a')
        )
        for t in texts:
            t.set_color('#c8e6c8')
            t.set_fontsize(9)
        for a in autotexts:
            a.set_color('#0a0f0a')
            a.set_fontweight('bold')
            a.set_fontsize(9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── AVG CONFIDENCE PER CLASS ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Avg Confidence per Waste Type</div>', unsafe_allow_html=True)
    class_conf = {}
    for h in history:
        class_conf.setdefault(h['class'], []).append(h['confidence'])

    for cls, confs in sorted(class_conf.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(confs) * 100
        c1, c2, c3 = st.columns([2, 5, 1])
        with c1:
            st.markdown(
                f"<span style='color:{CLASS_CONFIG[cls]['color']};font-size:0.85rem;font-weight:600;'>"
                f"{CLASS_CONFIG[cls]['emoji']} {cls}</span>",
                unsafe_allow_html=True
            )
        with c2:
            st.progress(float(avg / 100))
        with c3:
            st.caption(f"{avg:.1f}%")

    # ── PREVIOUS SCANS LOG ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🕓 Previous Scans</div>', unsafe_allow_html=True)

    for i, h in enumerate(reversed(history), 1):
        cfg_h       = CLASS_CONFIG[h['class']]
        badge_bg    = '#1a3a1a' if h['recyclable'] else '#3a1a1a'
        badge_color = '#4ade80' if h['recyclable'] else '#f87171'
        badge_text  = '♻ Recyclable' if h['recyclable'] else '✗ General'
        conf_color  = '#4ade80' if h['confidence'] >= 0.75 else ('#fbbf24' if h['confidence'] >= 0.55 else '#f87171')

        st.markdown(f"""
        <div class="scan-row">
            <div class="scan-num">#{total_scans - i + 1}</div>
            <div class="scan-emoji">{cfg_h['emoji']}</div>
            <div class="scan-class" style="color:{cfg_h['color']};">{h['class'].capitalize()}</div>
            <span class="scan-badge" style="background:{badge_bg};color:{badge_color};border:1px solid {badge_color}40;">
                {badge_text}
            </span>
            <div class="scan-conf" style="color:{conf_color};font-weight:600;">{h['confidence']*100:.1f}%</div>
            <div class="scan-time">{h['time']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── CLEAR BUTTON ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Session History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-footer">
    EcoScan AI · B.Tech CSE (AI &amp; ML) Sem IV<br>
    Graphic Era Hill University · April 2026<br>
    MobileNetV2 Transfer Learning · TrashNet Dataset
</div>
""", unsafe_allow_html=True)
