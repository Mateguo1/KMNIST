# KMNIST:
A simple attempt at the KMNIST classification.

![image](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/203089514-885a0207-19b3-4d76-95d4-77854e17204e.png)

I try these three models on this task: MobileNet V3, MobileViT and MobileViT with GhostNet. (Just use the smallest models in these papers).

## Programming runnin introduction: 

### 1. Preparation: 

#### 1.1 Code: 

Please extract the zip file, the code in this section can be accessed by "cd . /code/ef_cnn/" for the corresponding folder.

#### 1.2 Data: 

Please download lease download the MNIST format's KMNIST from https://github.com/rois-codh/kmnist, create a folder "data" in the folder ". /code/ef_cnn/", and put those downloaded data into this folder.

### 2. Code running:

Open a terminal and use "cd . /code/ef_cnn/" (this is the relative path, if access from the system terminal, it need to be changed to an absolute path) to enter the directory of this part, and then run the following commands for train and predict according to the requirements. 

If need to change more parameters for training and prediction, please refer to the comments and other contents in the specific code of train.py and predict.py. If there are any problems, please contact me by this email: gzypro@connect.hku.hk. Thanks.

#### 2.1 Train:

```python
# More optional parameters can be found in train.py
!python train.py
```

#### 2.2 Predict:

```python
# More optional parameters can be found in predict.py
!python predict.py
```



## Cite: 

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
