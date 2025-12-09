"""
Task 3: Extract call reasons and steps from agent turns
"""
import re
from typing import List, Dict, Any, Tuple
from transformers import pipeline
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def extract_customer_request(audio_items: List[Dict[str, Any]]) -> str:
    """
    Extract the initial customer request (reason for calling).
    Usually the first SYM actor turn after the greeting.
    """
    sorted_items = sorted(audio_items, key=lambda x: x.get('sequenceNumber', 0))
    
    # Find the first SYM turn that's not a greeting/instruction
    for item in sorted_items:
        actor = item.get('actor', '')
        transcript = item.get('fileTranscript', '').strip()
        
        if actor == 'SYM' and transcript:
            # Skip simulation instructions
            if 'simulation' in transcript.lower() or 'this concludes' in transcript.lower():
                continue
            # Skip very short responses
            if len(transcript.split()) > 5:
                return transcript
    
    return ""


def extract_agent_steps(audio_items: List[Dict[str, Any]]) -> List[str]:
    """
    Extract steps from TRAINEE (agent) turns.
    Steps are typically instructions or actions the agent takes.
    """
    sorted_items = sorted(audio_items, key=lambda x: x.get('sequenceNumber', 0))
    steps = []
    
    for item in sorted_items:
        actor = item.get('actor', '')
        transcript = item.get('fileTranscript', '').strip()
        
        if actor == 'TRAINEE' and transcript:
            # Filter out greetings and short acknowledgments
            if len(transcript.split()) < 5:
                continue
            if any(word in transcript.lower() for word in ['hello', 'thank you for calling', 'how can i help']):
                continue
            
            # Extract actionable steps
            # Look for imperative sentences or instructions
            if any(word in transcript.lower() for word in ['let me', 'i will', 'i can', 'we need to', 'next', 'then']):
                steps.append(transcript)
    
    return steps


def extract_with_transformer(dialogue: str, reason: str = None) -> Tuple[str, List[str]]:
    """
    Attempt to extract reason and steps using a transformer model.
    """
    try:
        # Use a text generation model for extraction
        # This is a simplified approach - in practice, you'd fine-tune or use a more specific model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Extract reason (first part of dialogue)
        if reason:
            extracted_reason = reason
        else:
            # Try to summarize the customer's initial request
            customer_part = dialogue.split("TRAINEE:")[0] if "TRAINEE:" in dialogue else dialogue[:500]
            if len(customer_part) > 50:
                summary = summarizer(customer_part, max_length=50, min_length=10, do_sample=False)
                extracted_reason = summary[0]['summary_text']
            else:
                extracted_reason = customer_part
        
        # Extract steps from agent turns
        agent_parts = re.split(r'TRAINEE:\s*', dialogue)[1:]  # Skip first part (customer)
        agent_text = " ".join(agent_parts[:5])  # Take first few agent turns
        
        if len(agent_text) > 100:
            summary = summarizer(agent_text, max_length=100, min_length=30, do_sample=False)
            steps_text = summary[0]['summary_text']
            # Split into steps (simplified)
            steps = [s.strip() for s in steps_text.split('.') if len(s.strip()) > 10]
        else:
            steps = []
        
        return extracted_reason, steps[:5]  # Limit to 5 steps
        
    except Exception as e:
        print(f"Transformer extraction error: {e}")
        return reason or "", []


def extract_with_gpt(dialogue: str, reason: str = None) -> Tuple[str, List[str]]:
    """
    Extract reason and steps using GPT.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return reason or "", []
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""Given the following customer service dialogue, extract:
1. The reason the customer called (what they need help with)
2. The steps the agent took to help them (as a list)

Dialogue:
{dialogue[:2000]}

Respond in JSON format:
{{
    "reason": "brief reason for the call",
    "steps": ["step 1", "step 2", "step 3"]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from customer service dialogues."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result.get('reason', reason or ''), result.get('steps', [])
        
    except Exception as e:
        print(f"GPT extraction error: {e}")
        # Return empty steps if GPT fails
        return reason or "", []


def extract_reason_and_steps(audio_items: List[Dict[str, Any]], method: str = "rule_based") -> Tuple[str, List[str]]:
    """
    Main function to extract reason and steps.
    
    Args:
        audio_items: List of audio content items
        method: 'rule_based', 'transformer', or 'gpt'
        
    Returns:
        Tuple of (reason, steps)
    """
    # First extract using rule-based approach
    reason = extract_customer_request(audio_items)
    steps = extract_agent_steps(audio_items)
    
    if method == "rule_based":
        return reason, steps
    elif method == "transformer":
        # Merge dialogue for transformer
        from src.dialogue_merger import merge_dialogue
        dialogue = merge_dialogue(audio_items)
        return extract_with_transformer(dialogue, reason)
    elif method == "gpt":
        from src.dialogue_merger import merge_dialogue
        dialogue = merge_dialogue(audio_items)
        return extract_with_gpt(dialogue, reason)
    else:
        return reason, steps

