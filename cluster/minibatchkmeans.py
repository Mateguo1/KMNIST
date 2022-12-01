"""
clustering using MiniBatchKMeans
input:number of clusters,data
output: clustering labels of data
"""
from sklearn.cluster import MiniBatchKMeans
def minibatchkmeans(n_clusters,data):
    cluster = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(data)
    y_pred_kemans = cluster.labels_
    return y_pred_kemans