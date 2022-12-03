import os
import numpy as np
import matplotlib.pyplot as plt
from model import resnet18
import torch.nn.functional as F
import torch
from data_loader import load_data
import time
from evaluation import compute_accuracy, evaluation, confuse_matrix_data

# Hyperparameters
RANDOM_SEED = 7008
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 10
train_loader, test_loader = load_data()

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES)
# model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

#Training
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
#         features = features.to(DEVICE)
#         targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

#Save model
torch.save(model.state_dict(), 'pretraind_model')
