from torchmetrics import ConfusionMatrix
import torch
import os
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
#         features = features.to(device)
#         targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
#         print(predicted_labels)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def evaluation(model, data_loader, device):
    predict_label = torch.tensor([], dtype = torch.int8)
    ground_truth = torch.tensor([], dtype = torch.int8)
    for i, (features, targets) in enumerate(data_loader):
        
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        predict_label = torch.cat((predict_label, predicted_labels), 0)
        ground_truth = torch.cat((ground_truth, targets), 0)
    num_examples = ground_truth.size(0)
    correct_pred = (predict_label == ground_truth).sum()
    acc = correct_pred.float()/num_examples * 100
    confuse_matrix = ConfusionMatrix(num_classes = 10)
    return acc, confuse_matrix(predict_label, ground_truth)
def confuse_matrix_data(model, data_loader):
    predict_label = torch.tensor([], dtype = torch.int8)
    ground_truth = torch.tensor([], dtype = torch.int8)
    for i, (features, targets) in enumerate(data_loader):
        
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        predict_label = torch.cat((predict_label, predicted_labels), 0)
        ground_truth = torch.cat((ground_truth, targets), 0)
    num_examples = ground_truth.size(0)
    correct_pred = (predict_label == ground_truth).sum()
    acc = correct_pred.float()/num_examples * 100
    confuse_matrix = ConfusionMatrix(num_classes = 10)
    return predict_label, ground_truth
