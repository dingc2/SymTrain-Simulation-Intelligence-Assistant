"""
Task 2: Merge dialogue text into continuous string while preserving speaker roles
"""
from typing import List, Dict, Any


def merge_dialogue(audio_items: List[Dict[str, Any]]) -> str:
    """
    Merge transcript lines into one continuous string while preserving speaker roles.
    
    Format: "TRAINEE: [text] SYM: [text] TRAINEE: [text] ..."
    
    Args:
        audio_items: List of audio content items with sequenceNumber, actor, and fileTranscript
        
    Returns:
        Merged dialogue string
    """
    # Sort by sequence number to ensure correct order
    sorted_items = sorted(audio_items, key=lambda x: x.get('sequenceNumber', 0))
    
    dialogue_parts = []
    for item in sorted_items:
        actor = item.get('actor', 'UNKNOWN')
        transcript = item.get('fileTranscript', '').strip()
        
        if transcript:  # Only add non-empty transcripts
            # Map actor names to readable format
            if actor == 'TRAINEE':
                dialogue_parts.append(f"TRAINEE: {transcript}")
            elif actor == 'SYM':
                dialogue_parts.append(f"SYM: {transcript}")
            else:
                dialogue_parts.append(f"{actor}: {transcript}")
    
    return " ".join(dialogue_parts)


def merge_all_dialogues(loaded_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge dialogues for all loaded files.
    
    Args:
        loaded_files: List of loaded JSON file data
        
    Returns:
        List of dictionaries with merged dialogue added
    """
    result = []
    for file_data in loaded_files:
        merged = merge_dialogue(file_data['audioContentItems'])
        result.append({
            'name': file_data['name'],
            'merged_dialogue': merged,
            'audioContentItems': file_data['audioContentItems'],
            'full_data': file_data.get('full_data', {})
        })
    return result


if __name__ == "__main__":
    from src.data_loader import load_json_files
    
    files = load_json_files()
    merged = merge_all_dialogues(files)
    
    if merged:
        print(f"\nMerged dialogue for '{merged[0]['name']}':")
        print(merged[0]['merged_dialogue'][:500] + "...")

