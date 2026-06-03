#datasets : includes MNIST
#transforms : function to preprocess images
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#.ToTensor() convert PIL Image (0-255) to PyTorch tensor (0.0-1.0)
#.Compose() chains multiple transforms together
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

#Should print 60k
print(f"Dataset size: {len(train_dataset)}")

#get first image and label
image, label = train_dataset[0]

print(f"Image shape: {image.shape}")
print(f"Label: {label}")

#Opens a window showing the image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()