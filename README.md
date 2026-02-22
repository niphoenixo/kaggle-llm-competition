# Kaggle LLM Classification Competition


[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=flat&logo=numpy)](https://numpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()
[![Last Commit](https://img.shields.io/github/last-commit/niphoenixo/kaggle-llm-competition?style=flat)](https://github.com/niphoenixo/kaggle-llm-competition)

**Predicting Human Preferences in LLM Conversations**  
*A Kaggle Getting Started Competition*

---

## Table of Contents
- [Competition Overview](#competition-overview)
- [My Approach](#my-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Next Steps](#next-steps)
- [About Me](#about-me)
- [Connect](#connect)

---

## Competition Overview

**Competition:** [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)  
**Goal:** Predict which chatbot response users prefer  
**Dataset:** Chatbot Arena conversations  
**Metric:** Log Loss  
**Type:** Getting Started Competition

Large Language Models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is critical. This competition uses real-world data from Chatbot Arena, where users chat with two anonymous LLMs and choose the answer they prefer.

---

## My Approach

### Background
This competition leverages my **LLM evaluation experience at NVIDIA** where I:
- Designed adversarial SFT datasets for QWEN and NeMo models
- Built 18+ rule-based validator judges
- Achieved 50%+ validator failure rates

### Submission 1: Baseline Model (Feb 2026)

| Metric | Detail |
|--------|--------|
| **Method** | Length-based heuristic |
| **Logic** | Longer responses are preferred more often |
| **Score** | *Pending* |
| **Notebook** | [`notebook-nisha-f8b47a360f.ipynb`](notebooks/notebook-nisha-f8b47a360f.ipynb) |

```python
def predict_preference(response_a, response_b):
    len_a = len(str(response_a))
    len_b = len(str(response_b))
    
    if len_a > len_b:
        return [0.6, 0.2, 0.2]  # Model A wins
    elif len_b > len_a:
        return [0.2, 0.6, 0.2]  # Model B wins
    else:
        return [0.33, 0.33, 0.34]  # Tie
```
---

## Project Structure

<pre>
kaggle-llm-competition/
│
├── notebooks/
│   └── notebook-nisha-f8b47a360f.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── utils.py
│
├── submissions/
│   └── first_submission.csv
│
├── data/
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
</pre>

### Directory Details

| Directory | Purpose |
|-----------|---------|
| **notebooks/** | Jupyter notebooks with experiments |
| **src/** | Reusable Python modules |
| **submissions/** | Kaggle submission files |
| **data/** | Data download instructions |

---

## Installation

```bash
# Clone repository
git clone https://github.com/niphoenixo/kaggle-llm-competition.git
cd kaggle-llm-competition

# Install dependencies
pip install -r requirements.txt
```

### Data Download

**Option 1: Kaggle API (Recommended)**
```bash
pip install kaggle
kaggle competitions download -c llm-classification-finetuning
unzip llm-classification-finetuning.zip -d data/
```

**Option 2: Manual Download**
1. Visit: https://www.kaggle.com/competitions/llm-classification-finetuning/data
2. Click "Download All"
3. Unzip files into the `data/` folder

---

## Usage

**On Kaggle:**
1. Go to [Competition Page](https://www.kaggle.com/competitions/llm-classification-finetuning)
2. Click "Code" → "New Notebook"
3. Enable GPU: Right sidebar → Accelerator → GPU T4 x2
4. Copy code from [`notebooks/notebook-nisha-f8b47a360f.ipynb`](notebooks/notebook-nisha-f8b47a360f.ipynb)
5. Run all cells → Commit → Submit

**Locally:**
```bash
jupyter notebook notebooks/notebook-nisha-f8b47a360f.ipynb
```

---

## Results

| Date | Submission | Approach | Log Loss | Status |
|------|------------|----------|----------|--------|
| 2026-02-11 | 1 | Length-based heuristic | **1.28203** | ✅ Submitted |
| 2026-02-13 | 3 | comprehensive feature engineering + multi-view ensemble approach | **1.04581** | ✅ Submitted |

---

## Next Steps

- [ ] Get first Kaggle score
- [ ] Add response quality features (bullet points, code blocks, disclaimers)
- [ ] Try Logistic Regression with TF-IDF
- [ ] Fine-tune BERT-based model

---

## About Me

**Nisha Gadhe**  
Senior Backend Engineer | LLM Evaluation Specialist

**Experience:**
- **NVIDIA** (2025-2026): LLM Evaluation Engineer
- **Simplify VMS** (2021-Present): Engineering Manager
- **16+ years** in distributed systems, microservices, AWS

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nisha-g-profile/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/niphoenixo)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/nishatime)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nisha.sonawane@gmail.com)

---

*Last updated: February 11th, 2026*

