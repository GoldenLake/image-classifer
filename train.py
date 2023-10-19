import math

import torch

from models import ghostnet
from datasets import transforms
from datasets.dataloader import get_loader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import sys
from utils.utils import train_one_epoch, evaluate
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ghostnet.ghostnet(num_classes=7)
model.to(device)
# print(model)

data_transforms = transforms.get_transform()

# print(data_transforms)
epochs = 10
best_acc = 0.0
lr = 0.01
lrf = 0.1
train_loader = get_loader('yixue', data_transforms['train'], 8, shuffle=True, num_workers=0, flag='train')
# print(train_loader)
val_loader = get_loader('yixue', data_transforms['val'], 8, shuffle=True, num_workers=0, flag='val')

loss_function = nn.CrossEntropyLoss()
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

train_steps = len(train_loader)
for epoch in range(epochs):
    # train
    train_loss, train_acc = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch)

    scheduler.step()

    # validate
    val_loss, val_acc = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=epoch)

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "./checkpoints/best.pth".format(epoch))
    torch.save(model.state_dict(), "./checkpoints/last.pth".format(epoch))
