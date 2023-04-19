import torch
# 新建两个tensor
x = torch.randn(3, 224, 224)
print(x.shape[0])
print(torch.unsqueeze(x, dim=0).repeat(8, 1,1,1).shape)