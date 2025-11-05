# 2025 DS/AI Project – CALL3

## Adaptive Computer-Assisted Language Learning (CALL) for English Learners

Welcome to our CALL project — an AI-powered language learning tool designed to help non-native English speakers improve their grammar through intelligent feedback.  
Built using Natural Language Processing (NLP) and Machine Learning, this web app analyses learner text, identifies grammatical errors, and provides personalised corrections — with a special focus on supporting native Spanish speakers.

---

## Features

- **Grammar Analysis** – Detects and categorises grammatical errors in learner-written English.  
- **Adaptive Feedback** – Provides tailored correction suggestions and improvement hints.  
- **Language-Specific Insights** – Compares error patterns across native languages to identify common difficulties.  
- **Model Training** – Trained on 100k samples from the *C4 200M Grammar Error Correction Dataset*.  
- **Web Application** – Interactive FastAPI platform where users can input text and view corrections in real time.  
- **Ethical AI Design** – Built following GDPR, IEEE, and EU Trustworthy AI principles for fairness, privacy, and transparency.

---

## Core Concepts Applied

This project integrates multiple topics from the *Data Science and Artificial Intelligence* module:

- **Machine Learning for Data Types** – Used to train and evaluate the grammar correction model.  
- **Knowledge Representation & Reasoning (KRR)** – Applied through structured classification of grammatical error types.  
- **Data Quality** – Ensured through dataset cleaning, selection, and balancing.  
- **Explainable AI (XAI)** – Feedback designed to be transparent and interpretable for learners.  
- **Ethics** – Guided by fairness, accountability, and responsible AI design.

---

## Tech Stack

- **Python 3.11**  
- **FastAPI** – Web framework for serving the grammar correction model  
- **LanguageTool (Python)** – Automated grammar correction  
- **ERRANT** – Error annotation and categorisation toolkit  
- **ROUGE** – Evaluation metric for model performance  
- **C4 200M Dataset (subset of 100k)** – Grammar error correction dataset  
- **Pandas / NumPy / Scikit-learn** – Data processing and model training

---

## Getting Started

Clone the repository:
```bash
git clone https://github.com/JoelPerezBermudez04/CALL3.git
cd adaptive-call-project
```
---

## Ethical Framework

The system was developed under the principles of the EU Ethics Guidelines for Trustworthy AI, IEEE Ethically Aligned Design, and GDPR.

Key focuses included:
- No personal data collection
- Transparent model documentation
- Minimising bias across language groups
- Fair and educational use of AI feedback

---

## Contributors
Bokyung Kim, Jiya Kartoidjojo, Katelyn Donovan and Joel Pérez

University of Twente – Data Science and Artificial Intelligence Project 2025

---

## Project Status
Completed – Adaptive CALL Web Application (Version 1.0)

Future work includes larger multilingual datasets, usability testing, and integration of transformer-based models for improved accuracy.

