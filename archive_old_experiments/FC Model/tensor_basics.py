import torch
data = ([1,2,3], [4,5,6]) #creates a 2D tensor
x = torch.tensor(data)
print(x)
print(x.shape)
y = x * 2 # multiplying array by 2
print(y)
z = y + x # adding 2 arrays togethor
print(z)
a = torch.tensor([1,2,3]) #creates a 1D tensor
print(a)
print(a.shape)