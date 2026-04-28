from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import joblib
import os

def tfidf_features(df_train, df_validation, df_test, text_column='text',
                   max_features=15000, save_path="./tfidf"):

    os.makedirs(save_path, exist_ok=True)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2), 
        stop_words='english', 
        min_df=2,    
        max_df=0.95,    
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True  
    )

    X_train_tfidf = vectorizer.fit_transform(df_train[text_column])
    X_validation_tfidf = vectorizer.transform(df_validation[text_column])
    X_test_tfidf = vectorizer.transform(df_test[text_column])
    joblib.dump(vectorizer, f"{save_path}/tfidf_vectorizer.pkl")
    save_npz(f"{save_path}/X_train_tfidf.npz", X_train_tfidf)
    save_npz(f"{save_path}/X_validation_tfidf.npz", X_validation_tfidf)
    save_npz(f"{save_path}/X_test_tfidf.npz", X_test_tfidf)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf, vectorizer


def load_tfidf_features(save_path="./tfidf"):
    vectorizer = joblib.load(f"{save_path}/tfidf_vectorizer.pkl")
    X_train = load_npz(f"{save_path}/X_train_tfidf.npz")
    X_validation = load_npz(f"{save_path}/X_validation_tfidf.npz")
    X_test = load_npz(f"{save_path}/X_test_tfidf.npz")

    return X_train, X_validation, X_test, vectorizer
