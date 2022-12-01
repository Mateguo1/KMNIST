import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from kmeans import *
from minibatchkmeans import *
from gmm import *
from Dense_AutoEncoder import *
from CNN_AutoEncoder_TSNE import *

# load the data
data1 = np.load(r'kmnist-train-imgs.npz')
data2 = np.load(r'kmnist-train-labels.npz')
train_imgs = data1['arr_0']
train_labels = data2['arr_0']

NMI = []
ARI = []
n_clusters=10

#### Clustering with unprocessed data ####
train_imgs1=train_imgs.reshape(60000,28*28)

# kmeans
pred1 = kmeans(n_clusters,train_imgs1)
NMI1 = metrics.normalized_mutual_info_score(pred1, train_labels)
ARI1 = metrics.adjusted_rand_score(pred1, train_labels)
NMI.append(NMI1)
ARI.append(ARI1)

# gmm
pred2 = gmm(train_imgs1,n_clusters,1)
NMI2 = metrics.normalized_mutual_info_score(pred2, train_labels)
ARI2 = metrics.adjusted_rand_score(pred2, train_labels)
NMI.append(NMI2)
ARI.append(ARI2)

# minibatchkmeans
pred3 = minibatchkmeans(n_clusters,train_imgs1)
NMI3 = metrics.normalized_mutual_info_score(pred3, train_labels)
ARI3 = metrics.adjusted_rand_score(pred3, train_labels)
NMI.append(NMI3)
ARI.append(ARI3)

#### Clustering with data after dimensionality reduction using Dense Autoencoder ####
train_imgs2 = dense_AE(train_imgs,2)

# kmeans
pred4 = kmeans(n_clusters,train_imgs2)
NMI4 = metrics.normalized_mutual_info_score(pred4, train_labels)
ARI4 = metrics.adjusted_rand_score(pred4, train_labels)
NMI.append(NMI4)
ARI.append(ARI4)

# gmm
pred5 = gmm(train_imgs2,n_clusters,30)
NMI5 = metrics.normalized_mutual_info_score(pred5, train_labels)
ARI5 = metrics.adjusted_rand_score(pred5, train_labels)
NMI.append(NMI5)
ARI.append(ARI5)

# minibatchkmeans
pred6 = minibatchkmeans(n_clusters,train_imgs2)
NMI6 = metrics.normalized_mutual_info_score(pred6, train_labels)
ARI6 = metrics.adjusted_rand_score(pred6, train_labels)
NMI.append(NMI6)
ARI.append(ARI6)

#### Clustering with data after dimensionality reduction using Convolutional Autoencoder and T-sne ####
train_imgs3 = cnn_AE_Tsne(train_imgs)

# kmeans
pred7 = kmeans(n_clusters,train_imgs3)
NMI7 = metrics.normalized_mutual_info_score(pred7, train_labels)
ARI7 = metrics.adjusted_rand_score(pred7, train_labels)
NMI.append(NMI7)
ARI.append(ARI7)

# gmm
pred8 = gmm(train_imgs3,n_clusters,2)
NMI8 = metrics.normalized_mutual_info_score(pred8, train_labels)
ARI8 = metrics.adjusted_rand_score(pred8, train_labels)
NMI.append(NMI8)
ARI.append(ARI8)

# minibatchkmeans
pred9 = minibatchkmeans(n_clusters,train_imgs3)
NMI9 = metrics.normalized_mutual_info_score(pred9, train_labels)
ARI9 = metrics.adjusted_rand_score(pred9, train_labels)
NMI.append(NMI9)
ARI.append(ARI9)


#### Results Visualization ####

NMI_1=[round(x1,3) for x1 in NMI]
ARI_1=[round(x2,3) for x2 in ARI]

size = 9
x = np.arange(size)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.figure(dpi=300,figsize=(13,6))
p1=plt.bar(x, NMI_1, width=width, label="NMI")
plt.bar_label(p1, label_type='edge',fontsize=8)
p2=plt.bar(x + width, ARI_1,  width=width, label="ARI")
plt.bar_label(p2, label_type='edge',fontsize=8)
x_labels = ['kmeans', 'GMM', 'Mkmeans', 'pre1_kmeans', 'pre1_GMM','pre1_Mkmeans', 'pre2_kmeans', 'pre2_GMM','pre2_Mkmeans']
plt.xticks(x, x_labels)
plt.title('NMI and ARI in different algorithms')
plt.legend()
plt.show()


