import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 1-x)])

train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} / {num_epochs}")
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print(f" Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved to mnist_cnn.pth")