"""
Task 4: Categorize all simulations
"""
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Optional imports for transformer-based categorization
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import numpy as np
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


def categorize_with_transformer(merged_dialogues: List[Dict[str, Any]], n_categories: int = 8) -> List[Dict[str, Any]]:
    """
    Categorize simulations using sentence transformers and clustering.
    """
    if not TRANSFORMER_AVAILABLE:
        print("sentence_transformers not available, falling back to rule-based")
        return categorize_all(merged_dialogues, method="rule_based")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract text for each simulation (use first part of dialogue)
        texts = []
        for item in merged_dialogues:
            dialogue = item.get('merged_dialogue', '')
            # Use first 500 chars as representation
            text = dialogue[:500] if dialogue else item.get('name', '')
            texts.append(text)
        
        # Generate embeddings
        embeddings = model.encode(texts)
        
        # Cluster into categories
        kmeans = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
        categories = kmeans.fit_predict(embeddings)
        
        # Assign category labels based on cluster centers
        category_labels = []
        for i, cat_id in enumerate(categories):
            # Get representative text from cluster
            cluster_indices = np.where(categories == cat_id)[0]
            cluster_texts = [texts[idx] for idx in cluster_indices[:3]]
            # Simple label generation (in practice, use better method)
            label = f"Category_{cat_id + 1}"
            category_labels.append(label)
        
        # Add categories to dialogues
        result = []
        for i, item in enumerate(merged_dialogues):
            item_copy = item.copy()
            item_copy['category'] = category_labels[i]
            result.append(item_copy)
        
        return result
        
    except Exception as e:
        print(f"Transformer categorization error: {e}")
        return merged_dialogues


def categorize_with_gpt(merged_dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Categorize simulations using GPT.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to rule-based
        return categorize_all(merged_dialogues, method="rule_based")
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # First, get categories from a sample
        sample_dialogues = merged_dialogues[:10]
        
        prompt = f"""Given these customer service simulation dialogues, create meaningful categories and assign each one.

Dialogues:
"""
        for i, item in enumerate(sample_dialogues):
            name = item.get('name', '')
            dialogue_preview = item.get('merged_dialogue', '')[:300]
            prompt += f"\n{i+1}. {name}\nPreview: {dialogue_preview}...\n"
        
        prompt += """
Create 6-10 meaningful categories (e.g., "Payment Updates", "Insurance Claims", "Order Management", etc.) and assign each dialogue to a category.

Respond in JSON format:
{
    "categories": ["Category1", "Category2", ...],
    "assignments": [
        {"index": 0, "category": "Category1"},
        {"index": 1, "category": "Category2"},
        ...
    ]
}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at categorizing customer service interactions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        category_map = {item['index']: item['category'] for item in result.get('assignments', [])}
        
        # Now categorize all dialogues
        result_list = []
        for i, item in enumerate(merged_dialogues):
            if i < len(sample_dialogues):
                category = category_map.get(i, "Other")
            else:
                # For remaining items, use GPT to categorize individually
                category = categorize_single_with_gpt(item, result.get('categories', []), client)
            
            item_copy = item.copy()
            item_copy['category'] = category
            result_list.append(item_copy)
        
        return result_list
        
    except Exception as e:
        print(f"GPT categorization error: {e}, using fallback")
        # Fallback to rule-based categorization
        return categorize_all(merged_dialogues, method="rule_based")


def categorize_single_with_gpt(item: Dict[str, Any], categories: List[str], client) -> str:
    """
    Categorize a single dialogue using GPT.
    """
    name = item.get('name', '')
    dialogue_preview = item.get('merged_dialogue', '')[:500]
    
    prompt = f"""Categorize this customer service simulation into one of these categories: {', '.join(categories)}

Simulation: {name}
Preview: {dialogue_preview}

Respond with just the category name."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except:
        return "Other"


def categorize_all(merged_dialogues: List[Dict[str, Any]], method: str = "transformer") -> List[Dict[str, Any]]:
    """
    Main categorization function.
    
    Args:
        merged_dialogues: List of dialogues with merged_dialogue field
        method: 'transformer' or 'gpt'
        
    Returns:
        List of dialogues with 'category' field added
    """
    if method == "transformer":
        return categorize_with_transformer(merged_dialogues)
    elif method == "gpt":
        return categorize_with_gpt(merged_dialogues)
    else:
        # Default: simple rule-based
        for item in merged_dialogues:
            name = item.get('name', '').lower()
            if 'payment' in name or 'card' in name:
                item['category'] = "Payment Updates"
            elif 'insurance' in name or 'claim' in name:
                item['category'] = "Insurance Claims"
            elif 'order' in name or 'booking' in name:
                item['category'] = "Order Management"
            elif 'health' in name:
                item['category'] = "Health Insurance"
            else:
                item['category'] = "Other"
        return merged_dialogues

