# -*- coding: utf-8 -*-
"""Movie_Genre_Classisfication.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H7HmxcLi8otytC3iQRg-kkPhFXq5f_ng

# Importing Dataset and convert to dataframe
"""

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install kaggle

!kaggle datasets download -d hijest/genre-classification-dataset-imdb

!unzip genre-classification-dataset-imdb.zip

import numpy as np
import pandas as pd

# File paths
train_path = '/content/Genre Classification Dataset/train_data.txt'
test_path = '/content/Genre Classification Dataset/test_data.txt'
test_solution_path = '/content/Genre Classification Dataset/test_data_solution.txt'

# Function to load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    movie_names = []
    genres = []
    reviews = []

    for line in lines:
        parts = line.split(' ::: ')
        movie_names.append(parts[1])
        if len(parts) > 3:  # Check if genre information exists
            genres.append(parts[2])
            reviews.append(parts[3].strip())
        else:
            genres.append(None)  # For test_data.txt, genres are not available
            reviews.append(parts[2].strip())

    return pd.DataFrame({
        'Movie Name': movie_names,
        'Genre': genres,
        'Review': reviews
    })

# Load training data
df_train = load_data(train_path)

# Load test data
df_test = load_data(test_path)

# Load test solution data
df_test_solution = load_data(test_solution_path)

# Display the DataFrames
print("Training Data:")
print(df_train.head())
print("\nTest Data:")
print(df_test.head())
print("\nTest Data Solution:")
print(df_test_solution.head())

import nltk
nltk.download('punkt_tab')

"""# EDA

"""

print("Missing values in Training Data:")
print(df_train.isnull().sum())

print("\nMissing values in Test Data:")
print(df_test.isnull().sum())

print("\nMissing values in Test Solution Data:")
print(df_test_solution.isnull().sum())

print("Training Data Types:")
print(df_train.dtypes)

print("\nTest Data Types:")
print(df_test.dtypes)

print("\nTest Solution Data Types:")
print(df_test_solution.dtypes)

unique_genres = df_train['Genre'].unique()
print("Unique Genres in Training Data:")
print(unique_genres)
print(len(unique_genres))

genre_counts = df_train['Genre'].value_counts()
print("Genre Distribution in Training Data:")
print(genre_counts)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
plt.title('Genre Distribution in Training Data')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=90)
plt.show()

df_train['Review Length'] = df_train['Review'].apply(len)
df_test['Review Length'] = df_test['Review'].apply(len)
df_test_solution['Review Length'] = df_test_solution['Review'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(df_train['Review Length'], bins=50, kde=True, color='blue')
plt.title('Distribution of Review Lengths in Training Data')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()

"""# Text Preprocessing

"""

import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define the preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Handle contractions
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "it's": "it is",
        "i'm": "i am",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Chat word treatment
    chat_words_map = {'gr8': 'great', 'u': 'you', 'r': 'are', 'lol': 'laughing out loud'}
    text = ' '.join([chat_words_map.get(word, word) for word in text.split()])

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove extra spaces
    text = ' '.join(words)

    return text

# Apply preprocessing to the Review column in all DataFrames
df_train['Cleaned Review'] = df_train['Review'].apply(preprocess_text)
df_test['Cleaned Review'] = df_test['Review'].apply(preprocess_text)
df_test_solution['Cleaned Review'] = df_test_solution['Review'].apply(preprocess_text)

# Display the cleaned data
print("Training Data (Cleaned):")
print(df_train[['Review', 'Cleaned Review']].head())

print("\nTest Data (Cleaned):")
print(df_test[['Review', 'Cleaned Review']].head())

print("\nTest Solution Data (Cleaned):")
print(df_test_solution[['Review', 'Cleaned Review']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(df_train['Cleaned Review'])

# Transform the test data
X_test_tfidf = vectorizer.transform(df_test['Cleaned Review'])

# Transform the test solution data
X_test_solution_tfidf = vectorizer.transform(df_test_solution['Cleaned Review'])

# Convert genres to numerical labels (if needed)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train['Genre'])
y_test_solution = label_encoder.transform(df_test_solution['Genre'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model on the test solution data
print("Accuracy:", accuracy_score(y_test_solution, y_pred))
print("Classification Report:")
print(classification_report(y_test_solution, y_pred, target_names=label_encoder.classes_))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": LinearSVC(random_state=42),
    "Naive Bayes": MultinomialNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy and F1-score
    accuracy = accuracy_score(y_test_solution, y_pred)
    report = classification_report(y_test_solution, y_pred, target_names=label_encoder.classes_, output_dict=True)
    f1_macro = report['macro avg']['f1-score']

    # Record training time
    training_time = time.time() - start_time

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "F1-Score (Macro Avg)": f1_macro,
        "Training Time": training_time
    })

    print(f"{model_name} completed in {training_time:.2f} seconds.\n")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display results
print("Model Comparison:")
print(results_df)

# Save results to CSV
results_df.to_csv('/content/model_comparison_results.csv', index=False)

