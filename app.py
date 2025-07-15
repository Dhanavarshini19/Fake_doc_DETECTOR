import streamlit as st
import joblib
import pandas as pd
import tempfile
from utils.ocr_utils import extract_text_from_image
from utils.image_utils import get_image_features
from utils.nlp_utils import clean_text, get_text_features

st.set_page_config(page_title="Fake Document Detector", layout="centered")
st.title("🕵️‍♂️ Fake Document Detection using AI")

uploaded_file = st.file_uploader("📄 Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    st.image(file_path, caption="Uploaded Document", use_column_width=True)

    model = joblib.load("model/fake_doc_model.pkl")

    text = extract_text_from_image(file_path)
    st.subheader("📝 OCR Extracted Text:")
    st.text_area("Text", value=text, height=150)

    cleaned = clean_text(text)
    features = {**get_text_features(cleaned), **get_image_features(file_path)}

    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]

    st.subheader("🔍 Prediction:")
    st.success("✅ REAL Document" if prediction == 0 else "❌ FAKE Document")