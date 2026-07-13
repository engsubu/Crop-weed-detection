import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Smart Sprayer")
st.title("🌱 Crop vs Weed Detection")
st.write("Upload a field image to detect weeds and crops")

@st.cache_resource
def load_model():
    model = YOLO('models/crop_weed_yolov8n/weights/best.pt')
    return model

model = load_model()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    with st.spinner('Detecting...'):
        results = model.predict(img_array, conf=0.4, verbose=False)
    
    annotated_img = results[0].plot() # this uses cv2 internally but headless handles it
    annotated_img = Image.fromarray(annotated_img[..., ::-1]) # BGR to RGB
    
    weed_count = sum(1 for box in results[0].boxes if int(box.cls) == 1)
    
    with col2:
        st.subheader("Results")
        st.metric("Weeds Detected", weed_count)
        if weed_count > 0:
            st.error("⚠️ SPRAY REQUIRED")
        else:
            st.success("✅ NO SPRAY NEEDED")
    
    st.image(annotated_img, caption="Detection Output", use_column_width=True)
