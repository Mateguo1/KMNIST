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

### 3.1 create virtual environment: 

```python
conda create -n Resnet18 pytorch matplotlib numpy \
scikit-learn torchvision pandas requests python=3.8
conda activate Resnet18
conda install torchmetrics==0.10.3 -c conda-forge
```

### 3.2 download dataset: 

```
git clone https://github.com/rois-codh/kmnist.git
cd kmnist
python download_data.py
##choose to download .npz sformat file
#type in 1 -> 2
```

### 3.3 train model: 

```python
python train.py
```

### 3.4 test model:

```python
##Test accuracy: 95.97%
acc:  tensor(95.9700)
confuse matrix
tensor([[971,   3,   0,   0,  11,   5,   0,   4,   4,   2],
        [  2, 921,  27,   0,   2,   3,  24,   4,   7,  10],
        [ 13,   0, 955,  12,   2,   4,   6,   2,   2,   4],
        [  0,   0,   5, 976,   0,  11,   4,   2,   2,   0],
        [ 28,   0,   1,   3, 932,   7,   3,   4,  16,   6],
        [  1,   1,  23,   3,   2, 946,   2,   5,   4,  13],
        [  4,   1,  11,   3,   0,   7, 971,   2,   0,   1],
        [  6,   0,   0,   2,   4,   1,   2, 971,   7,   7],
        [  0,   4,  12,   2,   1,   2,   0,   1, 978,   0],
        [  3,   1,   5,   2,   1,   3,   6,   1,   2, 976]])
```

------

## 4. EF-CNN:

Store in the folder ". /ef_cnn". 

Three models for this task: MobileNet V3, MobileViT and MobileViT with GhostNet. (Just use the smallest models in these papers).

### 4.1 Packages to use:

please "cd" into ". /ef_cnn".

```python
!pip install -r requirements.txt
```

### 4.1 Terminal: 

Open a terminal and use "cd . /ef_cnn/" (this is the relative path, if access from the system terminal, it need to be changed to an absolute path) to enter the directory of this part, and then run the following commands for train and predict according to the requirements. 

If need to change more parameters for training or prediction, please refer to the comments and other contents in train.py and predict.py. 

#### 4.1.1 Train: 

```python
# More optional parameters can be found in train.py
!python train.py --model_name mobilenet
```

#### 4.1.2 Predict: 

The trained models (20 epochs) are stored in "./weights".

```python
# More optional parameters can be found in predict.py
!python predict.py --model_name mobilenet
```

### 4.2 Colab:

Here is the link of its <a href="https://colab.research.google.com/drive/1Ap9wky1dPe-Jxo9cNd1bxMiMJo1QuS5R?usp=sharing">colab notebook</a>.

### 4.3 Evaluation:

Calculations such as the confusion matrix and accuracy can be obtained by running "confusion_matrix.ipynb".

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
