import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Package Predictor")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="PK3010/tourism-package-model",
        filename="tourism_model.pkl"  # Use exact filename
    )
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")