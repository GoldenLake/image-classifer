import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from mobilenetv2.model_v2 import MobileNetV2

from datasets.dataloader import get_loader
from datasets import transforms
from utils.utils import train_one_epoch, evaluate
import torch.optim.lr_scheduler as lr_scheduler
from predict import test
from models import get_models
from utils.loss import FocalLoss

data_transforms = transforms.get_transform()

epochs = 200
best_acc = 0.0
lr = 0.001
lrf = 0.01

train_loader = get_loader('../Medical classification', data_transforms['train'], 32, shuffle=True, num_workers=4,
                          flag='train')
# val_loader = get_loader('../Medical classification', data_transforms['val'], 32, shuffle=False, num_workers=0,
#                         flag='val')
test_loader = get_loader('../Medical classification', data_transforms['test'], 32, shuffle=False, num_workers=4,
                         flag='test')

model = MobileNetV2(num_classes=7)
model_weight_path = "./mobilenet_v2.pth"
pre_weights = torch.load(model_weight_path, map_location='cpu')
# 先加载到CPU 把网络的最后一层参数修改
pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 加载网络参数

# freeze features weights
# for param in net.features.parameters():
#     param.requires_grad = False

checkpoint_dir = './checkpoints'
# 损失函数
# loss_function = nn.CrossEntropyLoss()
loss_function = FocalLoss(reduction='mean')

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)

# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

last_checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
optimizer_path = os.path.join(checkpoint_dir, "optimizer.pth")

max_accuracy = [0]
train_loss_acc_list = []
test_loss_acc_list = []

for epoch in range(epochs):
    # train
    train_loss, train_acc = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch,
                                            loss_function=loss_function)
    with open('train_acc_loss.txt', 'a+') as train_txt:
        train_txt.writelines(f"{epoch}\t{train_loss}\t{train_acc}\n")

    scheduler.step()
    # validate
    # val_loss, val_acc = evaluate(model=model,
    #                              data_loader=val_loader,
    #                              device=device,
    #                              epoch=epoch,
    #                              loss_function=loss_function)

    test_loss, test_acc = evaluate(model=model,
                                   data_loader=test_loader,
                                   device=device,
                                   epoch=epoch,
                                   loss_function=loss_function)

    with open('test_acc_loss.txt', 'a+') as test_txt:
        test_txt.writelines(f"{epoch}\t{test_loss}\t{test_acc}\n")

    test(model=model, data_loader=test_loader, device=device, epoch=epoch, max_accuracy=max_accuracy)

    if test_acc >= best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))

    # Save the model and optimizer state at the end of each epoch
    torch.save(model.state_dict(), last_checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save the current epoch and learning rate to a file
