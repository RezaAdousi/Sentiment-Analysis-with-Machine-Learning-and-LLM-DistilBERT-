# Sentiment Analysis with Machine Learning and DistilBERT

A comparative study of classical machine learning models and a pretrained Transformer model for sentiment classification, followed by a stacking ensemble that combines their predictions.

## Overview

This project investigates sentiment classification on the **Amazon Fine Food Reviews** dataset by comparing traditional machine learning approaches with **DistilBERT**.

The main objectives are to:

* Compare classical ML models for sentiment classification.
* Evaluate a pretrained Transformer model (DistilBERT).
* Combine complementary model predictions using a **stacking ensemble**.
* Analyze whether combining traditional and Transformer-based approaches improves classification performance.

## Dataset

The project uses the **Amazon Fine Food Reviews** dataset, containing over 568,000 reviews.

After removing duplicates, missing values, and neutral reviews (score = 3), the resulting dataset contains approximately **525,000 reviews**.

Sentiment labels are constructed as follows:

* **Positive (1):** Rating ≥ 4
* **Negative (0):** Rating ≤ 2
* **Neutral reviews:** Removed

The data is split into training and test sets using an 80/20 split.

## Methods

### 1. Classical Machine Learning

Reviews are represented using **TF-IDF** features with a maximum of 500 features.

The following models are evaluated:

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes

### 2. DistilBERT

A pretrained `distilbert-base-uncased-finetuned-sst-2-english` model is used for sentiment prediction.

Due to the model's maximum input length, a filtered subset of reviews containing no more than 512 tokens is used for the DistilBERT evaluation.

### 3. Stacking Ensemble

The predictions from:

* Logistic Regression
* SVM
* DistilBERT

are combined as input features to a **Logistic Regression meta-model**.

The goal is to determine whether combining different modeling approaches can improve sentiment classification performance.

## Results

| Model                 |    Accuracy |
| --------------------- | ----------: |
| Logistic Regression   |      89.74% |
| SVM                   |      89.77% |
| Naive Bayes           |      84.57% |
| DistilBERT            |     84.97%* |
| **Stacking Ensemble** | **91.64%*** |

* Evaluated on the filtered subset of test reviews satisfying the ≤512-token constraint.

The stacking ensemble achieved the highest accuracy among the evaluated approaches, outperforming the individual models on the filtered test set.

## Key Findings

* Logistic Regression and SVM achieved very similar performance on TF-IDF features.
* Naive Bayes performed substantially worse than the other classical models.
* DistilBERT provided a strong pretrained Transformer baseline, although its evaluation was limited to shorter reviews.
* Combining classical ML models with DistilBERT through stacking improved performance over the individual models evaluated on the same filtered test subset.

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Hugging Face Transformers
* PyTorch
* Google Colab / Jupyter Notebook

## Project Structure

```text
.
├── Sentiment_Analysis.ipynb
└── README.md
```

## How to Run

Clone the repository and install the required dependencies:

```bash
pip install pandas numpy scikit-learn transformers torch gdown
```

Then open the notebook:

```text
Sentiment_Analysis.ipynb
```

The notebook downloads the dataset, preprocesses the reviews, trains the models, evaluates their performance, and builds the stacking ensemble.

## Author

**Reza Adousi**

Computer Engineering, Khatam University

GitHub: [RezaAdousi](https://github.com/RezaAdousi)
