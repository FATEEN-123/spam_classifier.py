import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (only needed first time)
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Preprocess text by removing punctuation, stopwords, and stemming"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only label and text columns
    df.columns = ['label', 'text']
    
    # Convert labels to binary (0=ham, 1=spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    return df

def extract_features(df):
    """Extract TF-IDF features from text"""
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label'].values
    return X, y, tfidf

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train models and evaluate performance"""
    # Initialize models
    nb_model = MultinomialNB()
    lr_model = LogisticRegression(max_iter=1000)
    
    # Train models
    nb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    nb_pred = nb_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    # Evaluate models
    print("Naive Bayes Results:")
    print(classification_report(y_test, nb_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, nb_pred))
    
    print("\nLogistic Regression Results:")
    print(classification_report(y_test, lr_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))
    
    return nb_model, lr_model

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def main():
    # Load dataset (replace with your file path)
    # Dataset should have two columns: label (ham/spam) and text
    try:
        df = load_and_preprocess_data('spam.csv')  # Common filename for spam datasets
    except:
        print("Error: Please ensure 'spam.csv' exists in your directory or provide the correct path")
        return
    
    # Extract features
    X, y, tfidf = extract_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    nb_model, lr_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot confusion matrices
    nb_pred = nb_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    plot_confusion_matrix(y_test, nb_pred, "Naive Bayes Confusion Matrix")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression Confusion Matrix")

if __name__ == "__main__":
    main()
