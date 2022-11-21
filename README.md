# KMNIST
A simple attempt at the KMNIST classification.

Here is the link for the KMNIST(https://github.com/rois-codh/kmnist). Please download the MNIST format and put it into this path: "./data".

I try these three models on this task: MobileNet V3, MobileViT and MobileViT with GhostNet. (I just use the smallest models in these papers).

## Train:

```python
# More optional parameters can be found in train.py
!python train.py
```

## Test:

```python
# More optional parameters can be found in predict.py
!python predict.py
```

