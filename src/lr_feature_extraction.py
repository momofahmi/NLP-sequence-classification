"""LR_Feature_Extraction for TFIDF and GloVe
Date: 07/04/2026
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import joblib
import os
import numpy as np

def tfidf_features(df_train, df_validation, df_test, text_column='text',
                   max_features=15000, save_path="./tfidf"):

    os.makedirs(save_path, exist_ok=True)
    #vectorizer initialisation with parameters
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2), # unigrams and bigrams
        stop_words='english', # all stopwords ignoreed
        min_df=2,     # rare words ignored /
        max_df=0.95,    # common words ignored / appears in almost all instances
        use_idf=True,
        smooth_idf=True
    )

    #fit transform only on training data
    X_train_tfidf = vectorizer.fit_transform(df_train[text_column])
    #transform only on both test and validation data
    X_validation_tfidf = vectorizer.transform(df_validation[text_column])
    X_test_tfidf = vectorizer.transform(df_test[text_column])

    # Save vectorizer
    joblib.dump(vectorizer, f"{save_path}/tfidf_vectorizer.pkl")
    # Save TF-IDF matrices
    save_npz(f"{save_path}/X_train_tfidf.npz", X_train_tfidf)
    save_npz(f"{save_path}/X_validation_tfidf.npz", X_validation_tfidf)
    save_npz(f"{save_path}/X_test_tfidf.npz", X_test_tfidf)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf, vectorizer


def load_tfidf_features(save_path="./tfidf"):
    #load vectorizer
    vectorizer = joblib.load(f"{save_path}/tfidf_vectorizer.pkl")
    
    # Load TF-IDF matrices
    X_train = load_npz(f"{save_path}/X_train_tfidf.npz")
    X_validation = load_npz(f"{save_path}/X_validation_tfidf.npz")
    X_test = load_npz(f"{save_path}/X_test_tfidf.npz")
    
    return X_train, X_validation, X_test, vectorizer

#===================== GloVe Embeddings ====================
# Date: 15/04/2026

def load_glove_embed(file_path):

    glove_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_values = line.strip().split()
            first_word = file_values[0]
            gv_vector = np.array(file_values[1:], dtype=np.float32)
            glove_embeddings[first_word] = gv_vector
    return glove_embeddings


def doc_vector(text, embeddings, embedding_dim=100):
    doc_words = text.lower().split()
    vectors = []
    
    for word in doc_words:
        if word in embeddings:
            vectors.append(embeddings[word])
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    return np.mean(vectors, axis=0)


def glove_features(df_train, df_val, df_test, text_column='clean_text',
                          file_path='glove.6B.100d.txt',
                          embedding_dim=100,
                          save_path='./glove_features'):

    os.makedirs(save_path, exist_ok=True)
    glove_embeddings = load_glove_embed(file_path)

    X_train = np.array([doc_vector(text, glove_embeddings, embedding_dim) 
                        for text in df_train[text_column]])

    X_validation = np.array([doc_vector(text, glove_embeddings, embedding_dim) 
                      for text in df_val[text_column]])

    X_test = np.array([doc_vector(text, glove_embeddings, embedding_dim) 
                       for text in df_test[text_column]])
    
    np.save(f'{save_path}/X_train_glove.npy', X_train)
    np.save(f'{save_path}/X_validation_glove.npy', X_validation)
    np.save(f'{save_path}/X_test_glove.npy', X_test)
    
    return X_train, X_validation, X_test
