import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torchvision import transforms
import os
BATCH_SIZE = 128
def load_data():
    #load data
    train_img_file = np.load('./kmnist/kmnist-train-imgs.npz')
    train_img_label = np.load('./kmnist/kmnist-train-labels.npz')
    test_img_file = np.load('./kmnist/kmnist-test-imgs.npz')
    test_img_label = np.load('./kmnist/kmnist-test-labels.npz')

    train_img_index = train_img_file.files
    train_label_index = train_img_label.files
    test_img_index= test_img_file.files
    test_label_index= test_img_label.files

    train_imgs = train_img_file[train_img_index[0]]
    train_labels = train_img_label[train_label_index[0]]
    test_imgs = test_img_file[test_img_index[0]]
    test_labels = test_img_label[test_label_index[0]]

    train_dataset_imgs = torch.Tensor(train_imgs)
    train_dataset_labels = torch.LongTensor(train_labels)
    test_dataset_imgs = torch.Tensor(test_imgs)
    test_dataset_labels = torch.LongTensor(test_labels)

    train_dataset_imgs = torch.unsqueeze(train_dataset_imgs, 1)
    test_dataset_imgs = torch.unsqueeze(test_dataset_imgs, 1)

    train_dataset = TensorDataset(train_dataset_imgs, train_dataset_labels)
    test_dataset = TensorDataset(test_dataset_imgs, test_dataset_labels)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)
    return train_loader, test_loader