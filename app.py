import streamlit as st
import traceback
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

st.set_page_config(page_title="Customer Assistance Tool", layout="wide")

try:
    from src.generate_steps import (
        predict_category_with_gpt, generate_steps_few_shot
    )
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

st.title("SymTrain Customer Assistance Tool")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

with open("reason_step_categories_openai.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with st.form(key="input_form"):
    user_text = st.text_area(
        "Customer request:",
        placeholder="Hi, I need help updating the payment method for my recent order...",
        height=120
    )

    submitted = st.form_submit_button("Run") 

if submitted and user_text.strip():
    try:
        category = predict_category_with_gpt(client, user_text)
        examples = [x for x in data if x.get("category") == category][:3] or data[:3]

        result = generate_steps_few_shot(client, user_text, examples)
        result["category"] = category

        st.subheader("Generated Response")
        st.metric("Predicted Category", category)
        st.metric("Number of Steps", len(result.get("steps", [])))

        st.write("### Steps")
        for i, step in enumerate(result.get("steps", []), 1):
            st.write(f"{i}. {step}")

        with st.expander("JSON Output"):
            st.json(result)

    except Exception as e:
        st.error("An error occurred.")
        st.code(traceback.format_exc())
