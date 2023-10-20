import os
import math
import torch
import pickle
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

# Create the "checkpoints" folder if it does not exist
checkpoint_dir = "./checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Check if the "checkpoints/last.pth" file exists
last_checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model = ghostnet.ghostnet(num_classes=7).to(device)
    model.load_state_dict(checkpoint)
else:
    model = ghostnet.ghostnet(num_classes=7).to(device)

data_transforms = transforms.get_transform()

# Load the saved epoch and learning rate
saved_data = None
try:
    with open(os.path.join(checkpoint_dir, "training_info.pkl"), "rb") as file:
        saved_data = pickle.load(file)
except FileNotFoundError:
    saved_data = {"epoch": 0, "lr": 0.01}

epochs = 100
best_acc = 0.0
lr = saved_data["lr"]
lrf = 0.1
train_loader = get_loader('../Medical classification', data_transforms['train'], 32, shuffle=True, num_workers=4, flag='train')
val_loader = get_loader('../Medical classification', data_transforms['val'], 32, shuffle=False, num_workers=4, flag='val')

loss_function = nn.CrossEntropyLoss()
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)

# Load the previously saved optimizer state
optimizer_path = os.path.join(checkpoint_dir, "optimizer.pth")
if os.path.exists(optimizer_path):
    optimizer.load_state_dict(torch.load(optimizer_path))
else:
    print("Optimizer state not found. Using default optimizer.")

# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# Get the starting epoch from the loaded model checkpoint
starting_epoch = saved_data["epoch"]

train_steps = len(train_loader)

for epoch in range(starting_epoch, epochs):
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
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))

    # Save the model and optimizer state at the end of each epoch
    torch.save(model.state_dict(), last_checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save the current epoch and learning rate to a file
    saved_data["epoch"] = epoch
    saved_data["lr"] = optimizer.param_groups[0]["lr"]
    with open(os.path.join(checkpoint_dir, "training_info.pkl"), "wb") as file:
        pickle.dump(saved_data, file)

# Training is now complete
