import torch
import torchvision.models as models


def get_torchvision_model(model_name, num_classes, pretrained=True):
    """
    根据配置参数返回TorchVision中的不同模型。

    参数：
    - model_name：要返回的模型的名称，例如 "resnet", "mobilenetv2" 等。
    - num_classes：模型的输出类别数量。
    - pretrained：是否使用预训练权重。

    返回值：
    - 返回所选模型的实例。
    """
    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == "vgg":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == "resnet":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=pretrained)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_name == "densenet":
        model = models.densenet161(pretrained=pretrained)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "inception":
        model = models.inception_v3(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
        model.fc = torch.nn.Linear(1024, num_classes)
    elif model_name == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    elif model_name == "resnext":
        model = models.resnext50_32x4d(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "wide_resnet":
        model = models.wide_resnet50_2(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "mnasnet":
        model = models.mnasnet1_0(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenetv3":
        if pretrained:
            model = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("不支持的模型名称")

    return model


# 使用示例：
# 现在好像不推荐这么写了 都是写权重weight
model_name = "mobilenetv3"  # 选择要使用的模型，可以是上述列出的任何一个模型
num_classes = 10  # 设置输出类别数量
pretrained = True  # 是否使用预训练权重

model = get_torchvision_model(model_name, num_classes, pretrained)
