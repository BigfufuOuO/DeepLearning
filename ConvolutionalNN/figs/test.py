import torch

a = torch.tensor([1,2,3])
b = []
for i in range (len(a)):
    b.append(a[i].item())
print(b)
