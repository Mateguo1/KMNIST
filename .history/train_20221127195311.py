import os
import argparse
import sys
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from km_dataset import kmnistDataset
from model import mobile_vit_xx_small 
from model import ghost_vit 
from MobileNet import mobilenet

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses
    accu_num = torch.zeros(1).to(device)   # Number of samples with correct cumulative predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.long().to(device)).sum()

        loss = loss_function(pred, labels.long().to(device))
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    
    accu_loss = torch.zeros(1).to(device)  # Accumulated losses
    accu_num = torch.zeros(1).to(device)   # Number of samples with correct cumulative predictions

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def main(args):


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")


    # follow this ratio to split the training and validation set
    ratio = 0.8
    whole_set = kmnistDataset("./data", is_train="train")
    length = len(whole_set)
    train_size,validate_size=int(ratio*length),int((1-ratio)*length)
    train_dataset,val_dataset=torch.utils.data.random_split(whole_set,[train_size, validate_size])
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model_name = args.model_name
    if model_name == "mobilenet":
        model = mobilenet(num_classes=args.num_classes)
    elif model_name == "mobileViT":
        model = mobile_vit_xx_small(num_classes=args.num_classes)
    elif model_name == "ghostViT":
        model = ghost_vit(num_classes=args.num_classes)

    model = model.to(device)
    
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/best_{args.model_name}.pth")

        torch.save(model.state_dict(), f"./weights/latest_{args.model_name}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    model_name: Model used for this mission (mobilenet, mobileViT, ghostViT);
    num_classes: Number of categories;
    epochs: Number of training epochs;
    batch-size: batch_size
    lr: learning rate
    
    '''
    parser.add_argument('--model_name', type=str, default="mobilenet")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./data")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)