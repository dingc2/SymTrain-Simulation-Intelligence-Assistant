import streamlit as st
import traceback

st.set_page_config(page_title="Customer Assistance MVP", page_icon="🤖", layout="wide")

# ---------------- IMPORT PIPELINE ----------------
try:
    from src.data_loader import load_json_files
    from src.dialogue_merger import merge_all_dialogues
    from src.reason_extractor import extract_reason_and_steps
    from src.categorizer import categorize_all
    from src.few_shot_pipeline import (
        predict_category_with_gpt,
        generate_steps_few_shot
    )
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()


# ---------------- LOAD + PROCESS TRAINING DATA ----------------
@st.cache_data
def load_pipeline_data():
    files = load_json_files("data/jsons")
    merged = merge_all_dialogues(files)

    processed = []
    for item in merged:
        try:
            reason, steps = extract_reason_and_steps(item["audioContentItems"], method="gpt")
            item.update({"reason": reason, "steps": steps})
            processed.append(item)
        except:
            pass

    try:
        return categorize_all(processed, method="gpt")
    except:
        for x in processed:
            x.setdefault("category", "Other")
        return processed


data = load_pipeline_data()
st.sidebar.success(f"Training items loaded: {len(data)}")


# ------------------- MAIN UI -------------------
st.title("SymTrain Customer Assistance Tool")

user_text = st.text_area(
    "Customer request:",
    placeholder="Hi, I need help updating the payment method for my recent order...",
    height=120
)

# Automatically run model when text entered
if user_text.strip():
    try:
        # Predict category
        category = predict_category_with_gpt(user_text, data)

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
