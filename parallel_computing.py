import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import tensorflow as tf
import time
import subprocess
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from wordcloud import WordCloud

# Load data
train_df = pd.read_csv("train_data.csv").dropna().reset_index(drop=True)
test_df = pd.read_csv("mixed_data.csv").dropna().reset_index(drop=True)

# EDA
print("--- Summary Statistics (Train) ---")
print(train_df.describe(include='all'))
print("\n--- Null Values ---")
print(train_df.isnull().sum())
print("\n--- Category Distribution ---")
print(train_df['category'].value_counts())

train_df['text_length'] = train_df['body'].apply(lambda x: len(str(x)))
sns.histplot(train_df['text_length'], bins=50)
plt.title("Text Length Distribution")
plt.show()

# Extra EDA Plots
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='category', order=train_df['category'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Category Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='category', y='text_length')
plt.xticks(rotation=45)
plt.title("Text Length Distribution by Category")
plt.tight_layout()
plt.show()

all_text = " ".join(train_df['body'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words")
plt.tight_layout()
plt.show()

# Text Preprocessing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_text(texts, batch_size=1000):
    processed = []
    cleaned = [clean_text(text) for text in texts]
    for i in tqdm(range(0, len(cleaned), batch_size)):
        docs = nlp.pipe(cleaned[i:i + batch_size], disable=["parser", "ner"])
        for doc in docs:
            tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
            processed.append(" ".join(tokens))
    return processed

train_df['clean_body'] = preprocess_text(train_df['body'])
test_df['clean_body'] = preprocess_text(test_df['body'])

# TF-IDF + PCA
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_df['clean_body'])
X_test_tfidf = tfidf.transform(test_df['clean_body'])

pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_tfidf.toarray())
X_test_pca = pca.transform(X_test_tfidf.toarray())

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['category'])
y_test = le.transform(test_df['category'])

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "NaiveBayes": MultinomialNB(),
    "SVM": SVC(probability=True)
}

def evaluate_model(name, model):
    train_start = time.time()

    if name == "NaiveBayes":
        model.fit(X_train_tfidf, y_train)
        train_end = time.time()
        pred_start = time.time()
        preds = model.predict(X_test_tfidf)
    else:
        model.fit(X_train_pca, y_train)
        train_end = time.time()
        pred_start = time.time()
        preds = model.predict(X_test_pca)

    pred_end = time.time()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    train_time = round(train_end - train_start, 3)
    pred_time = round(pred_end - pred_start, 3)

    print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Train Time: {train_time}s, Predict Time: {pred_time}s")
    return name, acc, f1, train_time, pred_time

# Sequential Training
start_seq = time.time()
results_seq = [evaluate_model(name, model) for name, model in models.items()]
end_seq = time.time()

# Parallel Training
start_par = time.time()
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(evaluate_model, name, model) for name, model in models.items()]
    results_par = [f.result() for f in as_completed(futures)]
end_par = time.time()

# Plot Execution Time Comparison
plt.figure(figsize=(8, 6))
seq_time = round(end_seq - start_seq, 2)
par_time = round(end_par - start_par, 2)

plt.bar(['Sequential', 'Parallel'], [seq_time, par_time], color=['blue', 'green'])
plt.title("Execution Time Comparison")
plt.ylabel("Seconds")
plt.text(0, seq_time + 0.5, f"{seq_time}s", ha='center')
plt.text(1, par_time + 0.5, f"{par_time}s", ha='center')
plt.tight_layout()
plt.show()

# Plot Efficiency (Train vs Predict Time)
def plot_efficiency_comparison(results, title):
    names = [r[0] for r in results]
    train_times = [r[3] for r in results]
    pred_times = [r[4] for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_times, width, label='Train Time', color='skyblue')
    plt.bar(x + width/2, pred_times, width, label='Predict Time', color='orange')

    plt.xlabel("Model")
    plt.ylabel("Seconds")
    plt.title(title)
    plt.xticks(x, names)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_efficiency_comparison(results_seq, "Sequential Model Efficiency")
plot_efficiency_comparison(results_par, "Parallel Model Efficiency")

# Performance Table
print("\nSequential Performance:")
for res in results_seq:
    print(f"{res[0]} -> Accuracy: {res[1]:.4f}, F1 Score: {res[2]:.4f}, Train Time: {res[3]}s, Predict Time: {res[4]}s")

print("\nParallel Performance:")
for res in results_par:
    print(f"{res[0]} -> Accuracy: {res[1]:.4f}, F1 Score: {res[2]:.4f}, Train Time: {res[3]}s, Predict Time: {res[4]}s")
