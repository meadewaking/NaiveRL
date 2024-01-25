import torch

a = torch.rand([2, 100, 32000])
print(a[:,-1].argmax(-1))