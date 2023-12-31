from torchvision import transforms, datasets, utils
import os
import torch


def get_loader(image_path, transform, batch_size, shuffle, num_workers, flag='train',drop_last=True):
    dataset = datasets.ImageFolder(root=os.path.join(image_path, flag),
                                   transform=transform)

    # print(dataset.classes)
    # print(dataset.targets)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers,drop_last=drop_last)
    return loader


