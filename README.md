# Customer Assistance Pipeline

A fully automated customer assistance pipeline that generates step-by-step instructions for customer service requests using few-shot learning with GPT.

## Project Structure

```
symtrain/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Task 1: Load JSON files
│   ├── dialogue_merger.py      # Task 2: Merge dialogue text
│   ├── reason_extractor.py    # Task 3: Extract reasons and steps
│   ├── categorizer.py          # Task 4: Categorize simulations
│   ├── few_shot_pipeline.py    # Task 5: Few-shot learning
│   └── process_data.py         # Main processing script
├── data/
│   └── jsons/                  # JSON simulation files
├── app.py                      # Task 6: Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Task 7: Docker configuration
└── README.md                   # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Quick Start (Recommended)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment (optional but recommended for GPT features):**
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

4. **In the app:**
   - Click "Load Training Data" in the sidebar
   - Enter a customer request in the main area
   - Click "Generate Steps"

### Other Scripts

**Test the pipeline:**
```bash
python test_pipeline.py
```

**Pre-process data (optional):**
```bash
python -m src.process_data
```

**Evaluate test data:**
```bash
python evaluate_test_data.py
```

### Docker

Build and run with Docker:

```bash
docker build -t customer-assistance-pipeline .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key customer-assistance-pipeline
```

## Tasks Completed

- ✅ Task 1: Load dataset and extract audioContentItems
- ✅ Task 2: Merge dialogue text preserving speaker roles
- ✅ Task 3: Extract call reasons and steps (transformer + GPT)
- ✅ Task 4: Categorize simulations (transformer + GPT)
- ✅ Task 5: Few-shot learning pipeline with GPT
- ✅ Task 6: Streamlit application
- ✅ Task 7: Dockerfile and packaging

## Test Data

The app includes 6 test inputs for evaluation:
1. Payment method update (detailed)
2. Payment method update (simple)
3. Insurance claim (detailed)
4. Insurance claim (simple)
5. Order status (simple)
6. Order status (with urgency)

