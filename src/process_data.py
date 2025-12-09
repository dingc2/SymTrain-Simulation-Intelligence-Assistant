"""
Main script to process all data and extract reasons/steps
"""
import json
from src.data_loader import load_json_files
from src.dialogue_merger import merge_all_dialogues
from src.reason_extractor import extract_reason_and_steps
from src.categorizer import categorize_all


def process_all_data(save_path: str = "processed_data.json"):
    """
    Process all JSON files: load, merge, extract, and categorize.
    """
    print("Loading JSON files...")
    files = load_json_files()
    print(f"Loaded {len(files)} files")
    
    print("Merging dialogues...")
    merged = merge_all_dialogues(files)
    print(f"Merged {len(merged)} dialogues")
    
    print("Extracting reasons and steps...")
    processed = []
    for item in merged:
        reason, steps = extract_reason_and_steps(item['audioContentItems'], method="rule_based")
        item['reason'] = reason
        item['steps'] = steps
        processed.append(item)
    
    print("Categorizing simulations...")
    categorized = categorize_all(processed, method="transformer")
    
    # Save processed data
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(categorized, f, indent=2, ensure_ascii=False)
    
    print(f"Saved processed data to {save_path}")
    return categorized


if __name__ == "__main__":
    data = process_all_data()

