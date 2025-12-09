"""
Task 6: Streamlit application for customer assistance pipeline
"""
import streamlit as st
import json
import os
import sys
import traceback
from pathlib import Path

# Page config must be first
st.set_page_config(
    page_title="Customer Assistance Pipeline",
    page_icon="🤖",
    layout="wide"
)

# Import after page config
try:
    from src.data_loader import load_json_files
    from src.dialogue_merger import merge_all_dialogues
    from src.reason_extractor import extract_reason_and_steps
    from src.categorizer import categorize_all
    from src.few_shot_pipeline import process_test_inputs, predict_category_with_gpt, generate_steps_few_shot
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please install missing dependencies: `pip install -r requirements.txt`")
    st.stop()

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'categorized_data' not in st.session_state:
    st.session_state.categorized_data = None
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None


@st.cache_data
def load_and_process_data():
    """Load and process all training data."""
    try:
        # Load data
        files = load_json_files("data/jsons")
        if not files:
            st.warning("No JSON files found in data/jsons directory")
            return []
        
        merged = merge_all_dialogues(files)
        if not merged:
            st.warning("No dialogues could be merged")
            return []
        
        # Extract reasons and steps
        processed = []
        for item in merged:
            try:
                reason, steps = extract_reason_and_steps(item['audioContentItems'], method="rule_based")
                item['reason'] = reason
                item['steps'] = steps
                processed.append(item)
            except Exception as e:
                st.warning(f"Error processing item {item.get('name', 'unknown')}: {e}")
                continue
        
        if not processed:
            st.warning("No items could be processed")
            return []
        
        # Categorize - use rule-based first (faster, more reliable)
        try:
            categorized = categorize_all(processed, method="rule_based")
            return categorized
        except Exception as e:
            st.warning(f"Categorization error: {e}")
            # If rule-based fails, just add "Other" category
            for item in processed:
                if 'category' not in item:
                    item['category'] = "Other"
            return processed
            
    except Exception as e:
        error_msg = f"Error loading data: {e}\n\n{traceback.format_exc()}"
        st.error(error_msg)
        st.code(traceback.format_exc())
        return []


def main():
    st.title("🤖 Customer Assistance Pipeline")
    st.markdown("Generate step-by-step instructions for customer service requests")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        if st.button("Load Training Data"):
            with st.spinner("Loading and processing data..."):
                st.session_state.categorized_data = load_and_process_data()
                st.success(f"Loaded {len(st.session_state.categorized_data)} simulations")
        
        if st.session_state.categorized_data:
            st.info(f"✅ {len(st.session_state.categorized_data)} simulations loaded")
            
            # Show category distribution
            categories = {}
            for item in st.session_state.categorized_data:
                cat = item.get('category', 'Other')
                categories[cat] = categories.get(cat, 0) + 1
            
            st.subheader("Categories")
            for cat, count in sorted(categories.items()):
                st.write(f"- {cat}: {count}")
    
    # Main content
    tab1, tab2 = st.tabs(["Generate Steps", "Test Data"])
    
    with tab1:
        st.header("Customer Request Input")
        
        user_input = st.text_area(
            "Enter customer request:",
            height=100,
            placeholder="Hi, I need to update my payment method for my recent order. Can you help me?"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Steps", type="primary"):
                if not user_input:
                    st.warning("Please enter a customer request")
                elif not st.session_state.categorized_data:
                    st.warning("Please load training data first")
                else:
                    with st.spinner("Generating steps..."):
                        # Predict category
                        category = predict_category_with_gpt(
                            user_input, 
                            st.session_state.categorized_data
                        )
                        
                        # Find examples from same category
                        examples = [
                            item for item in st.session_state.categorized_data 
                            if item.get('category') == category
                        ]
                        if not examples:
                            examples = st.session_state.categorized_data[:3]
                        
                        # Generate steps
                        result = generate_steps_few_shot(
                            user_input, 
                            examples[:3]
                        )
                        result['category'] = category
                        
                        st.session_state.current_result = result
        
        with col2:
            if st.button("Clear"):
                st.session_state.current_result = None
                st.rerun()
        
        # Display results
        if 'current_result' in st.session_state and st.session_state.current_result:
            result = st.session_state.current_result
            
            st.subheader("Generated Response")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", result.get('category', 'N/A'))
            with col2:
                st.metric("Steps Count", len(result.get('steps', [])))
            
            st.markdown("### Reason")
            st.info(result.get('reason', 'N/A'))
            
            st.markdown("### Steps")
            steps = result.get('steps', [])
            if steps:
                for i, step in enumerate(steps, 1):
                    st.write(f"{i}. {step}")
            else:
                st.warning("No steps generated")
            
            # Show JSON output
            with st.expander("JSON Output"):
                st.json(result)
    
    with tab2:
        st.header("Test Data Evaluation")
        
        test_inputs = [
            "Hi, I ordered a shirt last week and paid with my American Express card. I need to update the payment method because there is an issue with that card. Can you help me?",
            "Hi, I need to update the payment method for one of my recent orders. Can you help me with that?",
            "Hi, I am Sam. I was in a car accident this morning and need to file an insurance claim. Can you help me?",
            "Hi, can you help me file a claim?",
            "Hi, I recently ordered a book online. Can you give me an update on the order status?",
            "Hi, I have been waiting for two weeks for the book I ordered. What is going on with it? Can you give me an update?"
        ]
        
        if st.button("Process All Test Inputs"):
            if not st.session_state.categorized_data:
                st.warning("Please load training data first")
            else:
                with st.spinner("Processing test inputs..."):
                    results = process_test_inputs(
                        test_inputs,
                        st.session_state.categorized_data
                    )
                    st.session_state.test_results = results
        
        if 'test_results' in st.session_state and st.session_state.test_results is not None:
            st.subheader("Test Results")
            
            if len(st.session_state.test_results) == len(test_inputs):
                for i, (test_input, result) in enumerate(zip(test_inputs, st.session_state.test_results), 1):
                    with st.expander(f"Test {i}: {test_input[:50]}..."):
                        st.json(result)
            else:
                st.warning(f"Results count ({len(st.session_state.test_results)}) doesn't match test inputs count ({len(test_inputs)})")
                st.json(st.session_state.test_results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please check the terminal/console for more details.")

