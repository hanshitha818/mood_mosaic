# MoodMosaic: Mapping Emotions and Personality Through Language

MoodMosaic is an end-to-end NLP research prototype that demonstrates how everyday language can reveal patterns of **emotion**, **politeness tone**, and **personality** in a transparent and interpretable way.

Given free text or a CSV of messages, MoodMosaic predicts emotional signals, politeness tone, and Big Five personality traits, and visualizes the results through an interactive Streamlit dashboard. The goal is not clinical diagnosis, but meaningful analytic insight into how language reflects human behavior.

---

## Project Overview

MoodMosaic integrates three transformer-based NLP components into a unified pipeline:

1. **Emotion Classification**  
   A RoBERTa-based multi-label classifier fine-tuned on the **GoEmotions** dataset to detect fine-grained emotions expressed in text.

2. **Politeness Classification**  
   A DistilBERT-based classifier trained on the **Stanford Politeness Corpus** to identify polite vs. impolite tone.

3. **Personality Prediction (Big Five)**  
   A regression model using **Sentence Transformers** (e.g., `all-mpnet-base-v2`) to estimate Big Five personality traits (OCEAN) from aggregated text.

Per-message predictions are aggregated to produce:
- overall emotion distributions
- session-level personality profiles
- message-level tables combining text, emotion, and tone

All outputs are visualized using an interactive **Streamlit dashboard**.

---

## Main Features

- **Two input modes**
  - Free text (one message per line)
  - CSV upload with selectable text column

- **Three prediction tasks**
  - Emotion detection (multi-label)
  - Politeness classification (binary)
  - Personality regression (Big Five traits)

- **Interactive dashboard**
  - Emotion radar chart
  - Big Five personality radar chart
  - Per-message prediction table with filters
  - Downloadable CSV of predictions

- **Reproducible experiments**
  - Training scripts for all models
  - Saved checkpoints under `experiments/checkpoints/`
  - Configuration files under `configs/`

---

## Tech Stack

- **Programming Language:** Python 3.9+
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **NLP Models:** RoBERTa, DistilBERT, Sentence Transformers
- **Data Processing:** Pandas, NumPy
- **Visualization:** Streamlit, Plotly
- **Experiment Management:** YAML configs, saved checkpoints
- **Version Control:** Git, GitHub

---

## Installation

### Clone the repository
git clone https://github.com/hanshitha818/mood_mosaic.git  
cd mood_mosaic  

### Create and activate a virtual environment
python -m venv venv  
source venv/bin/activate    # macOS/Linux  
venv\Scripts\activate       # Windows  

### Install dependencies
pip install -r requirements.txt  

### Running the application
streamlit run app.py  

The application will be available at:  
http://localhost:8501  

## Key Outcomes
- Built a unified NLP pipeline combining emotion detection, politeness tone classification, and personality modeling  
- Translated complex transformer outputs into clear, human-interpretable visual insights  
- Demonstrated how aggregation strategies influence behavioral inference from text  
- Highlighted responsible use cases and limitations of AI-driven personality analysis  
