"""
Quick test to verify Streamlit app can start
"""
import streamlit as st

st.set_page_config(page_title="Test", page_icon="🧪")

st.title("Test App")
st.write("If you see this, Streamlit is working!")

try:
    from src.data_loader import load_json_files
    st.success("✅ Imports working!")
except Exception as e:
    st.error(f"❌ Import error: {e}")

