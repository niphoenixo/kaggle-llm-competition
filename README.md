# 🏆 Kaggle LLM Classification Competition

<div align="center">
  
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

[![GitHub last commit](https://img.shields.io/github/last-commit/niphoenixo/kaggle-llm-competition)](https://github.com/niphoenixo/kaggle-llm-competition)
[![GitHub repo size](https://img.shields.io/github/repo-size/niphoenixo/kaggle-llm-competition)](https://github.com/niphoenixo/kaggle-llm-competition)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-niphoenixo-blue?style=flat&logo=kaggle)](https://www.kaggle.com/niphoenixo)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Score](https://img.shields.io/badge/Score-TBD-orange)]()

**Predicting Human Preferences in LLM Conversations**  
*A Kaggle Getting Started Competition*

</div>

---

## 📋 Table of Contents
- [Competition Overview](#-competition-overview)
- [My Approach](#-my-approach)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Results](#-results)
- [Next Steps](#-next-steps)
- [About Me](#-about-me)
- [Connect](#-connect)

---

## 🎯 Competition Overview

**Competition:** [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)  
**Goal:** Predict which chatbot response users prefer in head-to-head battles  
**Dataset:** Chatbot Arena conversations between multiple LLMs  
**Metric:** Log Loss  
**Type:** Getting Started Competition (Rolling Leaderboard)

### The Challenge
Large Language Models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is critical. This competition uses real-world data from Chatbot Arena, where users chat with two anonymous LLMs and choose the answer they prefer. The task is to predict which response a user will prefer.

This challenge aligns with **Reinforcement Learning from Human Feedback (RLHF)** and **reward modeling** - critical components in modern LLM development.

---

## 💡 My Approach

### Background & Relevance

This competition directly leverages my **LLM evaluation experience at NVIDIA**, where I:
- Designed adversarial SFT datasets for QWEN and NeMo models
- Built 18+ rule-based validator judges
- Achieved 50%+ validator failure rates exposing model weaknesses
- Documented systematic model failure modes

### Submission 1: Baseline Model (February 2026)

| Metric | Detail |
|--------|--------|
| **Method** | Length-based heuristic |
| **Logic** | Longer responses are preferred more often |
| **Features** | Response length comparison |
| **Score** | *Pending first submission* |
| **Rank** | *Pending* |
| **Notebook** | [`notebook-nisha-f8b47a360f.ipynb`](notebooks/notebook-nisha-f8b47a360f.ipynb) |
| **Submission** | [`first_submission.csv`](submissions/first_submission.csv) |

#### Implementation
```python
def get_length_based_probabilities(response_a, response_b):
    """Simple heuristic: longer response gets higher probability"""
    len_a = len(str(response_a))
    len_b = len(str(response_b))
    
    if len_a > len_b:
        return [0.6, 0.2, 0.2]  # Model A wins
    elif len_b > len_a:
        return [0.2, 0.6, 0.2]  # Model B wins
    else:
        return [0.33, 0.33, 0.34]  # Tie