from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import joblib
import os

def tfidf_features(df_train, df_validation, df_test, text_column='text',
                   max_features=13500, save_path="./tfidf"):

    os.makedirs(save_path, exist_ok=True)
    vect_feat = TfidfVectorizer(
        max_features=max_features,
        use_idf=True, stop_words='english',    
        max_df=0.95,    
        smooth_idf=True,
        ngram_range=(1, 2), min_df=2,  
        sublinear_tf=True  
    )

    X_train_tfidf = vect_feat.fit_transform(df_train[text_column])
    X_validation_tfidf = vect_feat.transform(df_validation[text_column])
    X_test_tfidf = vect_feat.transform(df_test[text_column])
                     
    joblib.dump(vect_feat, f"{save_path}/tfidf_vectorizer.pkl")
    #saving all 3 vectorzed featurse in same format
    save_npz(f"{save_path}/X_train_tfidf.npz", X_train_tfidf)
    save_npz(f"{save_path}/X_validation_tfidf.npz", X_validation_tfidf)
    save_npz(f"{save_path}/X_test_tfidf.npz", X_test_tfidf)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf, vect_feat


def load_tfidf_features(save_path="./tfidf"):
    vector= joblib.load(f"{save_path}/tfidf_vectorizer.pkl")
    train_features=load_npz(f"{save_path}/X_train_tfidf.npz")
  
    val_feat= load_npz(f"{save_path}/X_validation_tfidf.npz")
    X_test =load_npz(f"{save_path}/X_test_tfidf.npz")

    return train_features, val_feat, X_test, vector
