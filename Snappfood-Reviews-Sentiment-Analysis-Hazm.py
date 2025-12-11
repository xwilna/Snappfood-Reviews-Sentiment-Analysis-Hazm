import pandas as pd
import numpy as np
from hazm import Normalizer, word_tokenize, Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df = df.dropna()
    
    normalizer = Normalizer()
    df['normalized_text'] = df['text'].apply(lambda x: normalizer.normalize(x))
    
    return df

def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmer = Stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def prepare_data(df):

    df['processed_text'] = df['normalized_text'].apply(tokenize_and_stem)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=['و', 'در', 'به', 'از', 'که', 'این', 'است', 'را', 'با', 'های']
        )),
        ('clf', SVC(kernel='linear', probability=True))
    ])
    
    parameters = {
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'tfidf__min_df': (1, 5, 10),
        'clf__C': (0.1, 1, 10),
    }
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return grid_search

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return y_pred

def main():
    file_path = 'snappfood.csv'
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = build_and_train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    import joblib
    joblib.dump(model, 'persian_sentiment_model.pkl')
