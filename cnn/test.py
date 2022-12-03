import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from model import resnet18
from evaluation import compute_accuracy, evaluation, confuse_matrix_data
from data_loader import load_data
train_loader, test_loader = load_data()
NUM_CLASSES = 10
DEVICE = 'cuda:1'
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

model = resnet18(NUM_CLASSES)
model.load_state_dict(torch.load('./pretrained/pretraind_model'))
with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    acc, confuse_matrix = evaluation(model, test_loader, device = DEVICE)
    print('acc: ', acc)
    print('confuse matrix')
    print(confuse_matrix)

predictive_label, ground_truth = confuse_matrix_data(model, test_loader)



disp = ConfusionMatrixDisplay(confusion_matrix = confuse_matrix)
disp.plot(
    include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
    ax=None,                        # 同上
    xticks_rotation="horizontal",   # 同上
    values_format="d"               # 显示的数值格式
)
plt.show()
