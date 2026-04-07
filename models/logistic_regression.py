''''
Date: 07/04/2026
Desc: Updated Linear Regression model
-- single model which will train on dataset with tfidf features and output sarcastic/sentiment values as a 2D array 
'''
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import numpy as np

class LogisticRegressionModel:
    def __init__(self, lr_parameters=None):
        default_lr = {
            'C': 1.0,
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }
        if lr_parameters is None:
            lr_parameters = default_lr
            
        self.model = MultiOutputClassifier(LogisticRegression(**lr_parameters)) 
        self.is_trained = False #keep track of whether model has been trained or not

    #function to extract labels and train the model
    def train_logistic_regression(self, X_train, y_train_df):
        #sarcasm labels extracted and converted to integer
        sarcasm_labels = y_train_df['Sarcasm'].astype(int).values
        #sentiment labels extracted and converted to integer
        sentiment_labels = y_train_df['Sentiment'].astype(int).values

        #2D numpy array for true value of text - column 0== sarcasm // column 1 == sentiment
        y_train = np.column_stack([sarcasm_labels,sentiment_labels])
        
        # Train logistic regression model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
        
    #function to make prediction on test data after training model
    #output 0/1 for each task
    def prediction_logistic_regression(self, X_test):
        if self.is_trained == False:
            raise ValueError("Model not trained yet")

        lr_predictions = self.model.predict(X_test)
        return {
            'Sarcasm': lr_predictions[:, 0],
            'Sentiment': lr_predictions[:, 1]
        }

    #function to predict probabilites to illustrate confidence of model
    #output probabilities
    def predict_probabilities_logistic_regression(self, X_test):
        if self.is_trained == False:
            raise ValueError("Model not trained yet.")
        
        logistic_regression_probabilities = self.model.predict_proba(X_test)
        
        return {
            'Sarcasm': logistic_regression_probabilities[0],  
            'Sentiment': logistic_regression_probabilities[1]
        }

    #evaluation function - metrics accuracy, F1, precision and recall
    def evaluate_logistic_regression(self, X_test, y_test_df):
        #sarcasm labels extracted from test data and converted to integer
        sarcasm_labels = y_test_df['Sarcasm'].astype(int).values
        #sentiment labels extracted from test data and converted to integer
        sentiment_labels = y_test_df['Sentiment'].astype(int).values

        predictions = self.prediction_logistic_regression(X_test)

        results = {
            'Sarcasm': {
                'accuracy': accuracy_score(sarcasm_labels, predictions['Sarcasm']),
                'f1': f1_score(sarcasm_labels, predictions['Sarcasm']),
                'precision': precision_score(sarcasm_labels, predictions['Sarcasm']),
                'recall': recall_score(sarcasm_labels, predictions['Sarcasm'])
            },
            'Sentiment': {
                'accuracy': accuracy_score(sentiment_labels, predictions['Sentiment']),
                'f1': f1_score(sentiment_labels, predictions['Sentiment']),
                'precision': precision_score(sentiment_labels, predictions['Sentiment']),
                'recall': recall_score(sentiment_labels, predictions['Sentiment'])
            }
        }
        
        return results

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        return self
