"""
clustering using kmeans
input:number of clusters , data
output: clustering labels of data
"""
from sklearn.cluster import KMeans
def kmeans(n_clusters,data):
    cluster = KMeans(n_clusters=n_clusters, random_state=0, algorithm='elkan').fit(data)
    y_pred_kemans = cluster.labels_
    return y_pred_kemans