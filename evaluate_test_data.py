"""
Script to evaluate test data and generate JSON outputs
"""
import json
import os
from src.data_loader import load_json_files
from src.dialogue_merger import merge_all_dialogues
from src.reason_extractor import extract_reason_and_steps
from src.categorizer import categorize_all
from src.few_shot_pipeline import process_test_inputs

# Test inputs from the requirements
TEST_INPUTS = [
    "Hi, I ordered a shirt last week and paid with my American Express card. I need to update the payment method because there is an issue with that card. Can you help me?",
    "Hi, I need to update the payment method for one of my recent orders. Can you help me with that?",
    "Hi, I am Sam. I was in a car accident this morning and need to file an insurance claim. Can you help me?",
    "Hi, can you help me file a claim?",
    "Hi, I recently ordered a book online. Can you give me an update on the order status?",
    "Hi, I have been waiting for two weeks for the book I ordered. What is going on with it? Can you give me an update?"
]


def main():
    print("Loading and processing training data...")
    
    # Load data
    files = load_json_files("data/jsons")
    print(f"Loaded {len(files)} JSON files")
    
    # Merge dialogues
    merged = merge_all_dialogues(files)
    print(f"Merged {len(merged)} dialogues")
    
    # Extract reasons and steps
    processed = []
    for item in merged:
        reason, steps = extract_reason_and_steps(item['audioContentItems'], method="rule_based")
        item['reason'] = reason
        item['steps'] = steps
        processed.append(item)
    
    # Categorize
    categorized = categorize_all(processed, method="transformer")
    print(f"Categorized {len(categorized)} simulations")
    
    # Process test inputs
    print("\nProcessing test inputs...")
    results = process_test_inputs(TEST_INPUTS, categorized)
    
    # Save results
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print("\nResults:")
    for i, (test_input, result) in enumerate(zip(TEST_INPUTS, results), 1):
        print(f"\nTest {i}:")
        print(f"  Input: {test_input[:60]}...")
        print(f"  Category: {result.get('category', 'N/A')}")
        print(f"  Reason: {result.get('reason', 'N/A')[:60]}...")
        print(f"  Steps: {len(result.get('steps', []))} steps")


if __name__ == "__main__":
    main()

