"""
Intelligent Waste Classification System — Desktop App (Updated)
Kazuki Seth | B.Tech CSE (AI & ML) Sem IV | Graphic Era Hill University
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

DISPOSAL_TIPS = {
    'cardboard': ('📦', '#D4845A', 'Flatten and keep dry — Blue Recycling Bin.'),
    'glass':     ('🫙', '#5AB8D4', 'Rinse thoroughly — Glass Recycling Bin only.'),
    'metal':     ('🥫', '#A8A8B8', 'Rinse cans — Metal / Mixed Recycling Bin.'),
    'paper':     ('📄', '#C8D46A', 'Keep dry, remove plastic windows — Paper Bin.'),
    'plastic':   ('🧴', '#9B7DD4', 'Check recycling number, rinse — Recycling Bin.'),
    'trash':     ('🗑️', '#E05555', 'Cannot be recycled — General Waste Bin.')
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0A0F0A; color: #E8EDE8; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #5DCC7A, #2E8B45) !important;
    color: #0A0F0A !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.75rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
[data-testid="stImage"] img { border-radius: 12px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return tf.keras.models.load_model("waste_model_v2.h5")
    except Exception:
        try:
            return tf.keras.models.load_model(
                "/content/drive/MyDrive/DL_Waste_Project/models/waste_model_v2.h5"
            )
        except Exception:
            return None

def predict(image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 1rem;">
    <div style="font-size:2.8rem;">♻️</div>
    <h1 style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
               color:#5DCC7A; margin:0.2rem 0 0; letter-spacing:-0.5px;">
        Intelligent Waste Classifier
    </h1>
    <p style="color:#4A5E4A; font-size:0.82rem; margin:0.2rem 0 0;">
        MobileNetV2 · Transfer Learning · TrashNet · 78.7% Accuracy
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("❌ Model file not found. Update the path in `load_model()`.")
    st.stop()

# Input mode
mode = st.radio("Input", ["📁 Upload Image", "📸 Camera"], horizontal=True)

img_pil = None
if mode == "📁 Upload Image":
    f = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if f:
        img_pil = Image.open(f)
else:
    cam = st.camera_input("Take a photo of your waste item")
    if cam:
        img_pil = Image.open(cam)

if img_pil:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(img_pil, use_column_width=True)

    if st.button("🔍 Classify Waste"):
        with st.spinner("Analysing..."):
            time.sleep(0.2)
            cls, confidence, all_preds = predict(img_pil, model)

        emoji, color, tip = DISPOSAL_TIPS[cls]

        if confidence < 0.55:
            st.warning("⚠️ Low confidence — try a clearer photo of a single item.")

        st.markdown(f"""
        <div style="background:#111811; border:1.5px solid {color}40;
                    border-radius:16px; padding:1.5rem; text-align:center; margin:0.8rem 0;">
            <div style="font-size:3rem; margin-bottom:0.3rem;">{emoji}</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem;
                        font-weight:800; color:{color}; margin-bottom:0.2rem;">
                {cls.upper()}
            </div>
            <div style="font-size:0.82rem; color:#6B7F6B;">{tip}</div>
            <div style="margin-top:0.8rem; font-size:1.3rem; font-weight:700;
                        color:{color};">{confidence*100:.1f}%</div>
            <div style="font-size:0.75rem; color:#4A5E4A;">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div style="margin:0.5rem 0 1rem;">
            <div style="height:8px; background:#1A2A1A; border-radius:4px; overflow:hidden;">
                <div style="width:{confidence*100:.0f}%; height:100%;
                            background:linear-gradient(90deg,{color}80,{color});
                            border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        st.markdown("**Confidence Breakdown**")
        prob_dict = {
            f"{DISPOSAL_TIPS[n][0]} {n}": float(p)
            for n, p in zip(CLASS_NAMES, all_preds)
        }
        st.bar_chart(prob_dict)

st.markdown("""
<div style="text-align:center; padding:2rem 0 0.5rem; color:#2A3A2A; font-size:0.72rem;">
    B.Tech CSE (AI & ML) · Graphic Era Hill University · April 2026
</div>
""", unsafe_allow_html=True)
