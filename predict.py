import pickle

# load model
with open('kmeans_model.pkl', 'rb') as file:
    model = pickle.load(file)

mapping = {0:'Sports', 1:'Business and Politics', 2:'ART and Entertainment'}

def predict(pca_result):
    predicted_cluster = model.predict(pca_result)
    cluster_name = mapping[predicted_cluster[0]]
    return predicted_cluster[0], cluster_name

    
