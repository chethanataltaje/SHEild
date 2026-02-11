# SHEild - Women-Centric Hate Speech Detection using NLP & Machine Learning  

**High-Accuracy Sexism Detection • Obfuscated Slur Handling • Dynamic Thresholding • Real-Time Streamlit App**



## Project Overview

SHEild is a hybrid, women-centric hate speech detection system designed to identify **explicit misogyny, obfuscated slurs, and subtle non-slur sexism** in text, while **protecting empowering and reclaimed speech**.

The system analyses raw user-generated text and delivers:

- Accurate classification of sexist content (explicit + implicit)
- Robust handling of evasion tactics (b***h, b1tch, c.u.n.t,s1ut, leetspeak)
- Context-aware detection using deep semantic embeddings
- Low false-positives on affirmative speech
- Near real-time inference through a Streamlit web interface

**No cloud dependency • No unsafe moderation decisions • Designed for ethical AI moderation**

---

## What SHEild Does

Given any text input (e.g., social media comment, chat message, forum post), SHEild can:

- Identify explicit hate speech & slurs  
- Detect subtle non-slur sexism and stereotypes  
- Catch obfuscated or censored abuse  
- Protect empowering or reclaimed language  
- Return a confidence score with CLEAN / SEXIST label  
- Provide real-time predictions via web UI  

All processing is done locally within the ML pipeline.

---

## Key Features

| Feature                         | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| Hybrid NLP Classifier           | SBERT embeddings + CatBoost classifier for high accuracy                    |
| Obfuscated Slur Detection       | Regex-based detection of censored & leetspeak slurs                         |
| Non-Slur Sexism Detection       | Handles subtle bias and stereotypes without explicit slurs                 |
| Dynamic Thresholding            | Stricter rules for reclaimed or ambiguous language                          |
| Affirmative Speech Protection  | Prevents false positives on empowering statements                           |
| Real-Time Streamlit UI          | Live inference with probability scores                                     |
| Lightweight Inference          | Optimized for fast local execution                                          |
| Safety-Oriented Rules Engine   | Deterministic override for heavily obfuscated abuse                         |

---

## Core Design Philosophy

SHEild does **not** rely on naive keyword filtering or unrestricted generative models.

It follows a **Hybrid Safety-First Architecture**:

1. **Rule-based safeguards**  
   Deterministic regex logic ensures explicit abuse is never missed.

2. **Semantic understanding via SBERT**  
   Deep embeddings capture context and subtle intent beyond keywords.

3. **Structured ML decision-making**  
   CatBoost provides stable, interpretable classification.

This ensures **reliability, explainability, and ethical deployment**.

---

## How It Works

1. Raw text is cleaned and normalized  
2. Sentence-BERT generates contextual embeddings  
3. Handcrafted linguistic features are extracted  
4. Features are stacked into a single vector  
5. CatBoost predicts sexism probability  
6. Safety rules & dynamic thresholds finalize decision  
7. Output is shown on Streamlit UI  

---

## Offline & Privacy-First

-No internet required for inference
No cloud APIs or external services
No telemetry or user-data logging
All processing happens locally on the machine
Designed for privacy-preserving content moderation and offline demos



##Intended Use Cases
Social media moderation prototypes
Online community safety tools
Academic demos of ethical AI moderation
NLP research on hate speech & fairness
Student projects and capstone demos
Privacy-first moderation pipelines for forums or platforms

---

## Results & Performance

- Overall Accuracy: **95%+**  
- PR-AUC: **~0.99**  
- High recall on subtle sexism  
- Low false positives on clean & empowering speech  

Validated on multiple hate speech datasets with class imbalance handling.

---

## Tech Stack


## Technology Stack (Offline-Capable)

| Component              | Technology / Tool                               | Purpose |
|-----------------------|--------------------------------------------------|---------|
| User Interface        | Streamlit                                       | Real-time web UI for live inference |
| Text Embeddings       | Sentence-Transformers (all-MiniLM-L6-v2)         | Context-aware sentence embeddings (384-dim) |
| Classifier            | CatBoost                                        | Fast, robust gradient-boosted classifier |
| Safety Rules          | Regex (re)                                      | Obfuscated slur detection & rule-based overrides |
| Deep Learning Backend | PyTorch                                         | Runtime for SBERT models |
| Data Processing       | NumPy, Pandas                                   | Feature stacking & preprocessing |
| Evaluation & Metrics  | Scikit-learn                                    | Accuracy, PR-AUC, ROC, confusion matrix |
| Visualization         | Plotly / Matplotlib (optional)                  | ROC/PR curves & analysis plots |
| App Deployment        | Streamlit (Local)                               | Offline-capable deployment |


## Authors

- Chethana TV
- Ananya Manoharan  
