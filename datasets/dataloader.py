from torchvision import transforms, datasets, utils
import os
import torch


def get_loader(image_path, transform, batch_size, shuffle, num_workers, flag='train'):
    dataset = datasets.ImageFolder(root=os.path.join(image_path, flag),
                                   transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers)
    return loader


