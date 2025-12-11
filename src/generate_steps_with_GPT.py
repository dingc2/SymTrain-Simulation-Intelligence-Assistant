

def predict_category_with_gpt(client, reason, steps):
    full_prompt = f'''
Assign a single category for the reason and steps.

ONLY USE THE 'Other' CATEGORY IF COMPLETELY NECESSARY

REASON:
{reason}

STEPS:
{steps}

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



def generate_steps_few_shot(user_text, examples):

    return