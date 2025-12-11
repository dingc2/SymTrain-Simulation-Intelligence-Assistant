from transformers import pipeline

# Load an actual HuggingFace model for zero-shot classification
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def categorize_customer_input(text, candidate_labels):
    """
    Categorize the customer's input using a zero-shot transformer.
    
    Args:
        text (str): Customer message or transcript snippet.
        candidate_labels (list[str]): List of possible category names.

    Returns:
        dict: {
            "input": str,
            "predicted_category": str,
            "scores": dict(label -> probability)
        }
    """
    
    result = classifier(
        sequences=text,
        candidate_labels=candidate_labels,
        multi_label=False  # Choose the single best label
    )

    # Clean format
    scores = {label: float(score) for label, score in zip(result["labels"], result["scores"])}
    best = result["labels"][0]

    return {
        "input": text,
        "predicted_category": best,
        "scores": scores
    }


# --------------------
# Example usage
# --------------------

CATEGORIES = [
            "Order Status & Fulfillment",
            "Returns, Cancellations & Exchange",
            "Travel & Hospitality Bookings",
            "Insurance Claims & Coverage",
            "Account Management & Billing",
            "Technical Support & Troubleshooting",
            "Sales & Quotes",
            "Other"
]

customer_text = "I need to change my flight time."

print(categorize_customer_input(customer_text, CATEGORIES))
