# MoodMosaic: Mapping Emotions and Personality Through Language

MoodMosaic is a small research prototype that shows how everyday language can reveal patterns of **emotion**, **politeness tone**, and **personality** in a simple, transparent way.

Given free text or a small CSV of messages, MoodMosaic:

- predicts **fine grained emotions** from the GoEmotions label set  
- predicts **politeness tone** (polite or impolite) from the Stanford Politeness Corpus  
- predicts a **Big Five personality profile** (O C E A N) using the Essays Big Five dataset  
- visualizes everything in an interactive **Streamlit dashboard** with radar charts and tables  

The goal is not clinical-level prediction, but a clear analytic view that feels like a mirror for the text you feed in.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Main Features](#main-features)  
3. [Repository Structure](#repository-structure)  
4. [Installation](#installation)  
5. [Models and Data](#models-and-data)  
6. [Training the Models](#training-the-models)  
7. [Running the Dashboard](#running-the-dashboard)  
8. [Using the Dashboard](#using-the-dashboard)  
9. [Evaluation and Metrics](#evaluation-and-metrics)  
10. [Limitations and Ethics](#limitations-and-ethics)  
11. [Planned Extensions](#planned-extensions)  
12. [Citation](#citation)  

---

## Project Overview

MoodMosaic combines three neural components into a single pipeline:

1. **Emotion classifier**  
   Fine-tunes a RoBERTa encoder on the **GoEmotions** dataset to predict multiple emotion labels per message.

2. **Tone classifier (politeness)**  
   Fine-tunes a DistilBERT encoder on the **Stanford Politeness** data (Cleanlab release) to predict **polite** vs **impolite** tone.

3. **Personality regressor (Big Five)**  
   Uses a **Sentence Transformers** encoder (e.g. `all-mpnet-base-v2`) plus a small feed-forward network to predict O C E A N scores from the **Essays Big Five** corpus.

An aggregation layer combines per-message outputs into:

- a global **emotion distribution**  
- a **Big Five personality profile** for the text in the current session  
- a **per-message table** with text, dominant emotion, and tone  

All results are presented in a simple **Streamlit app**.

---

## Main Features

- **Two input modes**
  - Free text input, with one message per line  
  - CSV upload, where you can choose which column contains text  

- **Three prediction heads**
  - Emotion (multi-label, GoEmotions)  
  - Tone (binary politeness)  
  - Personality (five continuous traits: O, C, E, A, N)  

- **Interactive dashboard**
  - Emotion radar chart  
  - Personality radar chart  
  - Per-message table with filters  
  - Downloadable CSV of predictions  

- **Reproducible experiments**
  - Training scripts for all three models  
  - Checkpoints saved under `experiments/checkpoints/...`  
  - Experiment configs under `configs/`  

---

## Screenshots and Visual Overview

### 1. Overall system pipeline

High-level architecture: text input → preprocessing → per-message models → aggregation → personality → dashboard.

```markdown
![System pipeline diagram](docs/img/pipeline_overview.png)
![Dashboard overview with emotion + Big Five radar charts](docs/img/dashboard_overview.png)
![Per-message details table](docs/img/dashboard_message_table.png)
![Emotion radar chart over GoEmotions labels](docs/img/emotion_radar.png)
![Big Five personality radar chart](docs/img/personality_radar.png)
![Big Five personality radar chart](docs/img/personality_radar.png)

