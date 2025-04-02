import pandas as pd
import pickle

# load scaler
with open('std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# load pca
with open('PCA.pkl', 'rb') as file:
    pca = pickle.load(file)

# load vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def feature_extraction(processed_text):
    # Convert text into series
    processed_text = pd.Series(processed_text)
    
    # Apply TF-IDF Vectorizer
    tfidf_matrix = vectorizer.transform(processed_text)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Apply PCA
    tfidf_scaled = scaler.transform(tfidf_df)
    pca_result = pca.transform(tfidf_scaled)

    return pca_result




