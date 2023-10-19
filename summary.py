import torch
from torchsummary import summary
from thop import profile
from models.ghostnet import ghostnet


model = ghostnet(num_classes=7).to('cuda')  # 替换为你自己的模型

# summary(model, input_size=(3, 224, 224))

input = torch.randn(1, 3, 224, 224).to('cuda')
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {round(flops / (10 ** 9), 5)}G, Params: {round(params/ (10 ** 6), 5)}M")