# KMNIST
A simple attempt at the KMNIST classification.

Here is the link for the dataset <a url="https://github.com/rois-codh/kmnist">KMNIST </a>. Please download the MNIST format and put it into this path: "./data".

I try these three models on this task: MobileNet V3, MobileViT and MobileViT with GhostNet. (I just use the smallest models in these papers). Paper links: <a url="https://arxiv.org/pdf/1905.02244v5.pdf">MobileNet V3</a>, <a url="https://arxiv.org/pdf/1911.11907.pdf">GhostNet</a>, <a url="https://arxiv.org/pdf/2110.02178.pdf" > MobileViT</a>

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

