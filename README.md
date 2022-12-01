# KMNIST:
A simple attempt for the classification of the KMNIST dataset.

![image](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/203089514-885a0207-19b3-4d76-95d4-77854e17204e.png)

# Programming running introduction: 

## 1. Cluster:

Store in the folder ". /cluster".

### 1.1 Operating environment and requirement:

```
-python ==3.9
-tensorflow ==2.9.1
-scikit-learn==1.0.2
-matplotlib==3.0.2
-numpy==1.15.4
```

### 1.2 What's in this repository:

```
kmeans.py---------------------Contains the implementaion of the k-menas algorithm
minibatchkmeans.py----------Contains the implementaion of the minibatchkmenas algorithm
gmm.py-----------------------Contains the implementaion of the Gaussian Mixture Model algorithm
Dense_AutoEncoder.py--------Contains the implementaion of data dimension using Dense AutoEncoder 
CNN_AutoEncoder_TSNE.py-- Contains the implementaion of data dimension using CNN AutoEncoder and T-sne
kmnist-train-imgs.npz------- Contains the Kuzushiji-MNIST data
kmnist-train-labels.npz------ Contains the labels of the Kuzushiji-MNIST
main.py----------------------Cluster the Kuzushiji-MNIST and visualize the results.
```

### 1.3 Run the program:

(1) Please change the type and the name of the dataset in main.py

(2) Run the main.py

------

## 2.MLP

Store in the folder ". /mlp", and enter the "mlp.ipynb"

Please run according to the order of the run box in the ipynb file, where model 4, 6, 20, 25, please select the corresponding model needed to run, and finally you can get the relevant results.

------

## 3. CNN:

Store in the folder ". /cnn". 

------

## 4. EF-CNN:

Store in the folder ". /ef_cnn". 

Three models for this task: MobileNet V3, MobileViT and MobileViT with GhostNet. (Just use the smallest models in these papers).

### 4.1 Terminal: 

Open a terminal and use "cd . /ef_cnn/" (this is the relative path, if access from the system terminal, it need to be changed to an absolute path) to enter the directory of this part, and then run the following commands for train and predict according to the requirements. 

If need to change more parameters for training or prediction, please refer to the comments and other contents in train.py and predict.py. 

#### 4.1.1 Train: 

```python
# More optional parameters can be found in train.py
!python train.py --model_name mobilenet
```

#### 4.1.2 Predict: 

```python
# More optional parameters can be found in predict.py
!python predict.py --model_name mobilenet
```

### 4.2 Colab:

Here is the link of its <a href="https://colab.research.google.com/drive/1Ap9wky1dPe-Jxo9cNd1bxMiMJo1QuS5R?usp=sharing">colab notebook</a>.

### 4.3 Evaluation:

Calculations such as the confusion matrix and accuracy can be obtained by running "confusion_matrix.ipynb"

If there are any problems, please contact us by this email: gzypro@connect.hku.hk. Thanks.

# Cite: 

```
@online{clanuwat2018deep,
  author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
  title        = {Deep Learning for Classical Japanese Literature},
  date         = {2018-12-03},
  year         = {2018},
  eprintclass  = {cs.CV},
  eprinttype   = {arXiv},
  eprint       = {cs.CV/1812.01718},
}
```
