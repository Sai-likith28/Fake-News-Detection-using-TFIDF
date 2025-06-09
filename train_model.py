import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Convert to string if not already
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and lemmatize
    words = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(cleaned)

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('images/confusion_matrix.png')
    plt.close()

def plot_metrics(metrics):
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    plt.bar(range(len(metrics)), list(metrics.values()), color=colors)
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(metrics.values()):
        plt.text(i, v/2, f'{v:.4f}', ha='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/performance_metrics.png')
    plt.close()

def plot_dataset_distribution(n_real, n_fake):
    plt.figure(figsize=(8, 6))
    counts = [n_fake, n_real]
    labels = ['Fake News', 'Real News']
    colors = ['#e74c3c', '#2ecc71']
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Distribution')
    plt.axis('equal')
    plt.savefig('images/dataset_distribution.png')
    plt.close()

def print_evaluation_metrics(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Store metrics for plotting
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(conf_matrix)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(conf_matrix)
    plot_metrics(metrics)
    
    return metrics

def main():
    print("Loading data...")
    try:
        # Create images directory if it doesn't exist
        import os
        if not os.path.exists('images'):
            os.makedirs('images')
        if not os.path.exists('models'):
            os.makedirs('models')

        # Load the datasets
        true_df = pd.read_csv('data/True.csv')
        fake_df = pd.read_csv('data/Fake.csv')
        print(f"Loaded {len(true_df)} real news articles and {len(fake_df)} fake news articles")

        # Plot dataset distribution
        plot_dataset_distribution(len(true_df), len(fake_df))

        # Add labels
        true_df['label'] = 1  # 1 for real news
        fake_df['label'] = 0  # 0 for fake news

        # Combine the datasets
        df = pd.concat([true_df, fake_df], ignore_index=True)
        
        # Clean the text
        print("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Split the data
        X = df['cleaned_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize the text
        print("\nVectorizing text...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train the model
        print("\nTraining model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test_vec)
        
        # Print and visualize evaluation metrics
        metrics = print_evaluation_metrics(y_test, y_pred)
        
        # Save the model and vectorizer
        print("\nSaving model and vectorizer...")
        joblib.dump(model, 'models/logistic_model.joblib')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
        print("Model and vectorizer saved successfully!")
        
        print("\nVisualization files have been saved in the 'images' directory:")
        print("1. confusion_matrix.png - Shows the model's prediction errors")
        print("2. performance_metrics.png - Shows accuracy, precision, recall, and F1 score")
        print("3. dataset_distribution.png - Shows the distribution of real vs fake news in the dataset")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 