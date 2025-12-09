"""
Task 1: Load the dataset and extract audioContentItems
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_json_files(data_dir: str = "data/jsons") -> List[Dict[str, Any]]:
    """
    Load all JSON files from the data directory and extract audioContentItems.
    
    Args:
        data_dir: Path to the directory containing JSON files
        
    Returns:
        List of dictionaries, each containing the simulation name and audioContentItems
    """
    json_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract audioContentItems
            audio_items = data.get('audioContentItems', [])
            
            json_files.append({
                'name': data.get('name', json_file.stem),
                'audioContentItems': audio_items,
                'full_data': data  # Keep full data for later use
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return json_files


if __name__ == "__main__":
    # Test the loader
    files = load_json_files()
    print(f"Loaded {len(files)} JSON files")
    if files:
        print(f"First file: {files[0]['name']}")
        print(f"Audio items in first file: {len(files[0]['audioContentItems'])}")

