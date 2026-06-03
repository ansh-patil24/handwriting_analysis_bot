import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 1-x)])

test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
