"""
Task 5: Generate steps for test data using GPT few-shot learning
"""
import json
import openai
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def find_similar_examples(test_input: str, categorized_data: List[Dict[str, Any]], 
                         n_examples: int = 3) -> List[Dict[str, Any]]:
    """
    Find similar examples from the same category for few-shot learning.
    Uses simple keyword matching - could be improved with embeddings.
    """
    # Simple keyword-based category prediction
    test_lower = test_input.lower()
    
    if 'payment' in test_lower or 'card' in test_lower:
        category = "Payment Updates"
    elif 'claim' in test_lower or 'accident' in test_lower or 'insurance' in test_lower:
        category = "Insurance Claims"
    elif 'order' in test_lower or 'book' in test_lower or 'status' in test_lower:
        category = "Order Management"
    else:
        category = "Other"
    
    # Find examples from same category
    examples = [item for item in categorized_data if item.get('category') == category]
    
    # Return first n examples
    return examples[:n_examples]


def predict_category_with_gpt(test_input: str, categorized_data: List[Dict[str, Any]]) -> str:
    """
    Predict category using GPT.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to keyword-based prediction
        test_lower = test_input.lower()
        if 'payment' in test_lower or 'card' in test_lower:
            return "Payment Updates"
        elif 'claim' in test_lower or 'accident' in test_lower or 'insurance' in test_lower:
            return "Insurance Claims"
        elif 'order' in test_lower or 'book' in test_lower or 'status' in test_lower:
            return "Order Management"
        return "Other"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Get unique categories
        categories = list(set([item.get('category', 'Other') for item in categorized_data]))
        
        prompt = f"""Categorize this customer request into one of these categories: {', '.join(categories)}

Customer request: {test_input}

Respond with just the category name."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Category prediction error: {e}, using fallback")
        # Fallback to keyword-based
        test_lower = test_input.lower()
        if 'payment' in test_lower or 'card' in test_lower:
            return "Payment Updates"
        elif 'claim' in test_lower or 'accident' in test_lower or 'insurance' in test_lower:
            return "Insurance Claims"
        elif 'order' in test_lower or 'book' in test_lower or 'status' in test_lower:
            return "Order Management"
        return "Other"


def generate_steps_few_shot(test_input: str, examples: List[Dict[str, Any]], 
                           use_gpt: bool = True) -> Dict[str, Any]:
    """
    Generate steps using few-shot learning.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not use_gpt or not api_key:
        # Fallback to simple rule-based
        return {
            "category": "Other",
            "reason": test_input,
            "steps": ["1. Listen to customer request", "2. Gather necessary information", "3. Process the request"]
        }
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Build few-shot examples
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            reason = ex.get('reason', '')
            steps = ex.get('steps', [])
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Customer: {reason}\n"
            examples_text += f"Steps: {json.dumps(steps)}\n"
        
        prompt = f"""You are a customer service assistant. Given a customer request, generate the steps needed to help them.

Few-shot examples:
{examples_text}

Now, generate steps for this customer request:
Customer: {test_input}

Respond in JSON format:
{{
    "category": "category name",
    "reason": "brief reason for the call",
    "steps": ["step 1", "step 2", "step 3"]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant that generates step-by-step instructions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        # Try to extract JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        return result
        
    except Exception as e:
        print(f"Few-shot generation error: {e}, using fallback")
        # Fallback to rule-based
        return {
            "category": "Other",
            "reason": test_input,
            "steps": [
                "1. Listen to customer request",
                "2. Gather necessary information", 
                "3. Process the request",
                "4. Confirm completion with customer"
            ]
        }


def process_test_inputs(test_inputs: List[str], categorized_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all test inputs and generate steps.
    """
    results = []
    
    for test_input in test_inputs:
        # Predict category
        category = predict_category_with_gpt(test_input, categorized_data)
        
        # Find examples from same category
        examples = [item for item in categorized_data if item.get('category') == category]
        if not examples:
            examples = categorized_data[:3]  # Fallback
        
        # Generate steps
        result = generate_steps_few_shot(test_input, examples[:3])
        result['category'] = category  # Use predicted category
        results.append(result)
    
    return results

