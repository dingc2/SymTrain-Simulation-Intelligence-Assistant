"""
Test script to verify the pipeline works
"""
import sys
from src.data_loader import load_json_files
from src.dialogue_merger import merge_all_dialogues
from src.reason_extractor import extract_reason_and_steps

def test_pipeline():
    """Test basic pipeline functionality."""
    print("Testing Customer Assistance Pipeline...")
    
    # Test 1: Load data
    print("\n1. Testing data loader...")
    try:
        files = load_json_files("data/jsons")
        print(f"   ✓ Loaded {len(files)} JSON files")
        if len(files) == 0:
            print("   ⚠ Warning: No JSON files found")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Merge dialogues
    print("\n2. Testing dialogue merger...")
    try:
        merged = merge_all_dialogues(files[:3])  # Test with first 3 files
        print(f"   ✓ Merged {len(merged)} dialogues")
        if merged:
            print(f"   Sample dialogue length: {len(merged[0].get('merged_dialogue', ''))} chars")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Extract reason and steps
    print("\n3. Testing reason/steps extraction...")
    try:
        if merged:
            reason, steps = extract_reason_and_steps(merged[0]['audioContentItems'], method="rule_based")
            print(f"   ✓ Extracted reason: {reason[:50]}...")
            print(f"   ✓ Extracted {len(steps)} steps")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

