import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.ghostnet import ghostnet


def main(classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ghostnet(num_classes=7).to(device)
    model_weight_path = "./checkpoints/best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_dir = "../Medical classification/test"  # Path to the directory containing test images
    assert os.path.exists(test_dir), "Directory '{}' does not exist.".format(test_dir)

    total_images = 0
    correct_predictions = 0
    for root, dirs, files in os.walk(test_dir):
        for filename in files:
            img_path = os.path.join(root, filename)
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            true_label = os.path.basename(root)  # Assuming the folder name is the true class label
            predicted_label = classes[predict_cla]

            if true_label == predicted_label:
                correct_predictions += 1
            total_images += 1

    accuracy = (correct_predictions / total_images) * 100
    print("Accuracy: {:.2f}%".format(accuracy))


if __name__ == '__main__':
    classes = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    main(classes)
