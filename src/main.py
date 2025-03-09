import pandas as pd

from evaluation import evaluate
from preprocessing import preprocess_data
from sklearn.cluster import SpectralClustering
df = pd.read_json('../data/cocktail_dataset.json')
X=preprocess_data(df)

spectral=SpectralClustering(n_clusters=3)
spectral.fit(X)
labels=spectral.labels_

evaluate(X,labels)

df=pd.read_json('../data/cocktail_dataset.json')
df['cluster']=labels
df.to_json('clustered_cocktails.json')