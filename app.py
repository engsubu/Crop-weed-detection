import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Smart Sprayer")
st.title("🌱 Crop vs Weed Detection")

@st.cache_resource
def load_model():
    return YOLO('models/crop_weed_yolov8n/weights/best.pt')

model = load_model()
uploaded_file = st.file_uploader("Upload Field Image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input")
    results = model.predict(np.array(image), conf=0.4)
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Output")
    
    weeds = sum(1 for box in results[0].boxes if int(box.cls) == 1)
    if weeds > 0:
        st.error(f"⚠️ SPRAY REQUIRED - {weeds} Weeds Detected")
    else:
        st.success("✅ NO SPRAY - Only Crops")
