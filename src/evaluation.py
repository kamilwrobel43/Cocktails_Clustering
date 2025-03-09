from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def evaluate(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    print(f"Silhouette score: {silhouette}")
    print(f"Davies Bouldin score: {davies_bouldin}")
    print(f"Calinski Harabasz score: {calinski_harabasz}")

    colors_dict = {0: 'blue', 1: 'green', 2: 'black'}
    colors = [colors_dict[label] for label in labels]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.title("Clustering visualization with PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
    plt.show()
    print(f"PCA Preserved variance: {np.sum(pca.explained_variance_ratio_):.3f}")

    return silhouette, davies_bouldin, calinski_harabasz
