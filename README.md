# IMDB Sentiment Analysis with Feedforward Neural Networks

## Project Overview
This project implements a **3-layer feedforward neural network (FFNN)** to classify **IMDB movie reviews** as **positive** or **negative**. The experiments explore the impact of **activation functions** (ReLU vs Tanh) and **regularization** (Dropout vs No Dropout) on model performance and generalization.  

The project demonstrates **early stopping, checkpointing**, and provides **training/validation accuracy curves** along with **classification reports**, highlighting practical deep learning challenges like overfitting, underfitting, and activation sensitivity.

---

## Features / Experiments
- **Activation Functions:** ReLU, Tanh  
- **Regularization Techniques:** Dropout (0.5) vs No Dropout  
- **Training Enhancements:** Early stopping with patience, checkpointing best model  
- **Evaluation:** Accuracy, precision, recall, F1-score, plots of training/validation accuracy  

---

## Repository Structure

IMDB-Sentiment-Analysis/
│
├── data/ # Optional instructions to download IMDB dataset
├── notebooks/ # Step-by-step notebooks
│ ├── 01_Preprocessing.ipynb
│ ├── 02_Model_Setup.ipynb
│ ├── 03_Training_Experiments.ipynb
│ └── 04_Final_Analysis.ipynb
├── src/ # Python modules
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── utils.py
├── results/
│ ├── plots/
│ │ ├── train_val_acc.png
│ │ └── epoch_vs_train_acc.png
│ └── reports/
│ └── classification_reports.txt
├── requirements.txt
├── README.md
└── LICENSE


---

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd IMDB-Sentiment-Analysis
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

Notebooks: Step-by-step workflow from data preprocessing → model setup → training → analysis

Plots & Reports: Saved in results/

## Training Example:

from src.train import train_model
from src.model import FFNN

model = FFNN(vocab_size=tokenizer.vocab_size, activation="relu", regularization="dropout")

train_acc, val_acc = train_model(model, train_loader, test_loader)

## Key Findings
Model	Train Acc	Val Acc	Notes
ReLU + Dropout (0.5)	0.85→0.94	0.83	Underfitting; poor F1 for positive class due to aggressive dropout
ReLU + No Dropout	0.73→0.96	0.82	Stable F1-scores; minor overfitting
Tanh + Dropout (0.5)	0.71→0.95	0.83-0.84	Best class-wise performance; avoids ReLU dead neuron issue

## Observations:

Dropout helps prevent overfitting but can cause underfitting in small networks.

ReLU learns fast but is sensitive to dropout; Tanh is slower but more stable with high dropout.

Activation and regularization choice must be tuned together, not in isolation.


