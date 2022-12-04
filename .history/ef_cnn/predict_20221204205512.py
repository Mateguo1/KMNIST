import torch
from model import mobile_vit_xx_small
from model import ghost_vit
from MobileNet import mobilenet
import numpy as np
import argparse
from km_dataset import read_kmnist_train
# import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    model_name: Model used for this predicting (mobilenet, mobileViT, ghostViT);
    '''
    parser.add_argument('--model_name', type=str, default="mobilenet")

    opt = parser.parse_args()
    model_name = opt.model_name
    test_imgs, test_labels, is_train = read_kmnist_train(
        "./data",is_train="t10k"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pre_clas = []
    total_loss = 0

    if model_name == "mobilenet":
        model = mobilenet(num_classes=10, in_channels=1).to(device)
    elif model_name == "mobilevit":
        model = mobile_vit_xx_small(num_classes=10, in_channels=1).to(device)
    elif model_name == "ghostnet":
        model = ghost_vit(num_classes=10, in_channels=1).to(device)

    # load weights
    model_weight_path = f"./weights/best_{model_name}.pth"
    model.load_state_dict(torch.load(model_weight_path,map_location=device))

    model.eval()

    # predicting 
    for i in range(len(test_imgs)):
        test_img, test_label = torch.Tensor(test_imgs[i]), torch.from_numpy(np.array(test_labels[i]))
        loss_function = torch.nn.CrossEntropyLoss()
        test_img = torch.unsqueeze(test_img, dim=0)
        

        with torch.no_grad():
            output = torch.squeeze(model(test_img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            pre_cla = torch.argmax(predict)# 

            
            loss = loss_function(output, test_label.long())
            total_loss += loss

        pre_clas.append(str(pre_cla.numpy()))

    avg_loss = total_loss/int(len(test_imgs))
    loss_ = str(avg_loss.numpy())

    result = ",".join(pre_clas)


    if not os.path.exists("./test/"):
        os.makedirs("./test/")
    
    with open(f"./test/{model_name}_pred_result.txt", "w") as f:
        f.write(result)

    with open(f"./test/{model_name}_pred_avgloss.txt", "w") as f:
        f.write(loss_)
    
    num = 0
    with open(f"./test/{model_name}_pred_result.txt") as f:
        re = f.readlines()
        res = re[0].split(",")
    
    for i in range(len(res)):
        if res[i] == str(test_labels[i]):
            num+=1

    acc_ = num/len(res)
    
    print(f"The prediction based on {model_name} is done, and its accuracy is {acc_} and average loss value is {loss_} on the test set.")

