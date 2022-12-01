"""
clustering using Gaussian mixture model
input:data,number of clusters,Number of iterations
output: clustering labels of data
"""
from sklearn import mixture
def gmm(data, n_clusters,iters):
    clst = mixture.GaussianMixture(n_components=n_clusters,max_iter=iters,covariance_type="full")
    clst.fit(data)
    predicted_labels =clst.predict(data)
    return predicted_labels