import streamlit as st
import traceback
from openai import OpenAI
st.set_page_config(page_title="Customer Assistance Tool", page_icon="🤖", layout="wide")

# ---------------- IMPORT PIPELINE ----------------
try:
    from src.generate_steps import (predict_category_with_gpt, generate_steps_few_shot)
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# ------------------- MAIN UI -------------------
st.title("SymTrain Customer Assistance Tool")

user_text = st.text_area(
    "Customer request:",
    placeholder="Hi, I need help updating the payment method for my recent order...",
    height=120
)

client = OpenAI()
# Automatically run model when text entered
if user_text.strip():
    try:
        # Predict category
        category = predict_category_with_gpt(client, reason, steps)

        # A few example conversations from same category
        examples = [x for x in data if x.get("category") == category][:3] or data[:3]

        # Generate steps
        result = generate_steps_few_shot(user_text, examples)
        result["category"] = category

        # ---------- DISPLAY ----------
        st.subheader("Generated Response")

        st.metric("Predicted Category", category)
        st.metric("Number of Steps", len(result.get("steps", [])))

        st.write("### Reason")
        st.info(result.get("reason", ""))

        st.write("### Steps")
        for i, step in enumerate(result.get("steps", []), 1):
            st.write(f"{i}. {step}")

        with st.expander("JSON Output"):
            st.json(result)

    except Exception as e:
        st.error("An error occurred.")
        st.code(traceback.format_exc())
