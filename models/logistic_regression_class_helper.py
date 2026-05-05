# -*- coding: utf-8 -*-
"""
Logistic Regression Functions & class
Author: Umme-Yusrah Sumtally

Separate Models for Sentiment and Sarcasm Classification

Citations: Srirag, Dipankar, Aditya Joshi, Jordan Painter, and Diptesh Kanojia. 2025.
BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English.
"""


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import numpy as np

#Extracting true sentiment and sarcasm labels from dataset
def extract_labels(y_df):
    sarc_labels = y_df['Sarcasm'].astype(int).values
    sent_labels = y_df['Sentiment'].astype(int).values
    return sarc_labels, sent_labels

def calculate_metrics(y_true, y_pred):
    return {
        'F1_Macro':  f1_score(y_true, y_pred, average='macro', zero_division=0), 
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),  
        'Recall':    recall_score(y_true, y_pred),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
    }

#Finding the decision threshold that maximises Macro F1 on the validation set.
def find_best_threshold(model, X_val, y_val, task_name):
    probabilities = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_macro_f1 = 0.0
    
    #0.01 steps gave better results than 0.05
    for thr in np.arange(0.10, 0.90, 0.01):
        predictions = (probabilities >= thr).astype(int)
        if len(np.unique(predictions)) < 2:
            continue
        macro_f1 = f1_score(y_val, predictions, average='macro', zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1= macro_f1
            best_threshold = thr

    return round(best_threshold, 2)

def tune_C(X_train, y_train, task_name):
    best_C = 1.0
    best_macro_f1_score = 0.0
    search_candidates = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    for C in search_candidates:
        model = LogisticRegression(
            C=C,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
        mean_score = scores.mean()

        if mean_score > best_macro_f1_score:
            best_macro_f1_score = mean_score
            best_C = C

    return best_C

#class Separate Models for both sarcasm and sentment tasks
class SeparateLR:
    def __init__(self):
        self.sarcasm_model = None
        self.sentiment_model = None
        self.sarcasm_threshold = 0.5
        self.sentiment_threshold = 0.5

        self.is_trained = False
    def train_SeparateLR(self, X_train, y_train_df, X_validation=None, y_validation_df=None):
        sarcasm_labels, sentiment_labels = extract_labels(y_train_df)
        
        best_c_sarc = tune_C(X_train, sarcasm_labels, 'Sarcasm')
        self.sarcasm_model = LogisticRegression(
            C= best_c_sarc,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        best_c_sent = tune_C(X_train, sentiment_labels, 'Sentiment')
        self.sentiment_model = LogisticRegression(
            C=best_c_sent,
            solver='liblinear',
            class_weight='balanced',
            max_iter=  1000,
            random_state=42
        )

        self.sarcasm_model.fit(X_train, sarcasm_labels)
        self.sentiment_model.fit(X_train, sentiment_labels)
        self.is_trained = True 

        if X_validation is not None and y_validation_df is not None:
            sarc_validation_labels, sent_validation_labels = extract_labels(y_validation_df)
            self.sarcasm_threshold = find_best_threshold(self.sarcasm_model, X_validation, sarc_validation_labels, 'Sarcasm')
            self.sentiment_threshold = find_best_threshold(self.sentiment_model, X_validation, sent_validation_labels, 'Sentiment')
        else:
            print(" No validation data provided — default threshold 0.5 used.")
        return self

    def label_prediction_SeparateLR(self, X_test):
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        sarc_probabilities = self.sarcasm_model.predict_proba(X_test)[:, 1]
        sent_probabilities = self.sentiment_model.predict_proba(X_test)[:, 1]
        return {'Sarcasm':   (sarc_probabilities >= self.sarcasm_threshold).astype(int), 'Sentiment': (sent_probabilities >= self.sentiment_threshold).astype(int)}

    def SeparateLR_evaluation(self, X_test, y_test_df):
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        preds = self.label_prediction_SeparateLR(X_test)
        true_sarc_labels, true_sent_labels = extract_labels(y_test_df)
        return {'Sarcasm':   calculate_metrics(true_sarc_labels, preds['Sarcasm']),'Sentiment': calculate_metrics(true_sent_labels, preds['Sentiment'])}

    def probability_prediction_SeparateLR(self, X_test):
        if self.is_trained == False:
            raise ValueError("Model not trained yet.")
        return {
                'Sarcasm': self.sarcasm_model.predict_proba(X_test),
                'Sentiment': self.sentiment_model.predict_proba(X_test)
               }

    def save_SeparateLR_models(self, filepath_prefix="./models"):
        joblib.dump(self.sarcasm_model,f"{filepath_prefix}/separate_sarcasm.pkl")
        joblib.dump(self.sentiment_model, f"{filepath_prefix}/separate_sentiment.pkl")
        joblib.dump({'sarcasm_threshold': self.sarcasm_threshold, 'sentiment_threshold': self.sentiment_threshold}, f"{filepath_prefix}/separate_thresholds.pkl")
        print(f" Models and thresholds saved")

    def load_SeparateLR_models(self, filepath_prefix="./models"):
        self.sarcasm_model= joblib.load(f"{filepath_prefix}/separate_sarcasm.pkl")
        self.sentiment_model=joblib.load(f"{filepath_prefix}/separate_sentiment.pkl")
        models_thresholds= joblib.load(f"{filepath_prefix}/separate_thresholds.pkl")
        self.sarcasm_threshold= models_thresholds['sarcasm_threshold']
        self.sentiment_threshold=models_thresholds['sentiment_threshold']
        self.is_trained=True
        print(f"Models and thresholds loaded")
        return self
