# Movie Genre Classification using Machine Learning

This project focuses on classifying movies into genres based on their plot summaries using **traditional machine learning algorithms**. The dataset contains movie plot descriptions and their corresponding genres, and the goal is to build a model that can accurately predict the genre of a movie given its plot.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Algorithms Used](#algorithms-used)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

The objective of this project is to develop a machine learning model that can classify movies into genres based on their textual plot descriptions. The project involves:
- Preprocessing the text data (cleaning, tokenization, etc.).
- Converting text into numerical features using **TF-IDF**.
- Training and evaluating multiple machine learning models (e.g., Logistic Regression, Random Forest, SVM, etc.).
- Comparing the performance of different algorithms.

---

## Dataset

The dataset used in this project consists of:
- **Training Data**: Contains movie plot descriptions and their corresponding genres.
- **Test Data**: Contains movie plot descriptions without genres (used for predictions).
- **Test Solution Data**: Contains the correct genres for the test data (used for evaluation).

The dataset is stored in the following files:
- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt`

---

## Approach

1. **Data Preprocessing**:
   - Clean the text data (remove HTML tags, URLs, stopwords, etc.).
   - Tokenize the text and convert it into sequences.
   - Encode the genre labels into numerical values.

2. **Feature Extraction**:
   - Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.

3. **Model Training**:
   - Train multiple machine learning models, including:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - Multinomial Naive Bayes
     - XGBoost
     - k-Nearest Neighbors (k-NN)

4. **Model Evaluation**:
   - Evaluate the models using accuracy, precision, recall, and F1-score.
   - Compare the performance of different algorithms.

---

## Algorithms Used

The following machine learning algorithms were used in this project:

| Algorithm               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Logistic Regression** | A linear model for classification.                                          |
| **Random Forest**       | An ensemble of decision trees for improved accuracy.                        |
| **SVM**                 | A powerful algorithm for high-dimensional data like text.                   |
| **Naive Bayes**         | A probabilistic model based on Bayes' theorem.                              |
| **XGBoost**             | A gradient-boosting algorithm known for its performance.                    |
| **k-NN**                | A simple algorithm that classifies samples based on their neighbors.        |

---

## Results

The performance of the models was evaluated on the test data. Below are the results:

| Model               | Accuracy | F1-Score (Macro Avg) | Training Time |
|---------------------|----------|----------------------|---------------|
| Logistic Regression | 58.56%   | 0.31                 | 10s           |
| Random Forest       | 60.12%   | 0.35                 | 1m 30s        |
| SVM                 | 61.45%   | 0.38                 | 45s           |
| Naive Bayes         | 57.89%   | 0.29                 | 5s            |
| XGBoost             | 59.78%   | 0.33                 | 2m            |
| k-NN                | 55.23%   | 0.27                 | 30s           |

---

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/movie-genre-classification.git
   cd movie-genre-classification
