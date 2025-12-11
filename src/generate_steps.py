import json

def predict_category_with_gpt(client, user_text):
    full_prompt = f'''
Assign a single category based on user's request.

ONLY USE THE 'Other' CATEGORY IF COMPLETELY NECESSARY

USER REQUEST:
{user_text}

PICK A CATEGORY:
"Order Status & Fulfillment",
"Returns, Cancellations & Exchange", 
"Travel & Hospitality Bookings", 
"Insurance Claims & Coverage", 
"Account Management & Billing",
"Technical Support & Troubleshooting", 
"Sales & Quotes", 
"Other"

ONLY OUTPUT THE CATEGORY:
example: Order Status & Fulfillment

'''
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content



def generate_steps_few_shot(client, user_text, examples):
    full_prompt = f'''
Based on the examples provided, generate steps, be detailed but concise.

USER REQUEST:
{user_text}

EXAMPLE:
{examples}

Respond strictly in JSON format:
{{
  "steps": [
    "step 1",
    "step 2",
    "...",
    "step N"
  ]
}}

'''
    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    content_string = response.choices[0].message.content
    
    # Parse the string into a Python dictionary
    try:
        result_json = json.loads(content_string)
        return result_json
    except json.JSONDecodeError:
        # Fallback if something goes wrong
        return {"steps": ["Error parsing steps from LLM"]}
