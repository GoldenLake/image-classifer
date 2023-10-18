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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ghostnet.ghostnet(num_classes=7)
net.to(device)
# print(model)

data_transforms = transforms.get_transform()

print(data_transforms)


train_loader = get_loader('yixue', data_transforms['train'], 8, shuffle=True, num_workers=0, flag='train')
print(train_loader)
val_loader = get_loader('yixue', data_transforms['val'], 8, shuffle=True, num_workers=0, flag='val')

loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = './net.pth'
best_acc = 0.0
train_steps = len(train_loader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_num = 193
    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')
