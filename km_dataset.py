
from torch import utils
import os
import struct
import numpy as np

def read_kmnist_train(path, is_train='train'):
    '''Read KMNIST data in MNIST format'''
    labels_path = os.path.join(path,f'{is_train}-labels-idx1-ubyte')
    images_path = os.path.join(path,f'{is_train}-images-idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
    # gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 1, 28, 28)
    return images.astype(np.float32), labels, is_train

class kmnistDataset(utils.data.Dataset):
  '''KMNIST_DataSet'''
  def __init__(self, file_path, is_train):
    self.features, self.labels, process_type = read_kmnist_train(file_path, is_train)
    print("read "+str(len(self.features))+f' {process_type} examples')
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

  def __len__(self):
    return len(self.features)
