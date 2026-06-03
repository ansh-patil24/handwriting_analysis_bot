import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from cnn_model import CNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 1-x)])
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

cnn_model = CNN()
cnn_model.load_state_dict(torch.load('mnist_cnn.pth'))
cnn_model.eval()

def visualize_predictions(num_images=10, show_incorrect=False):
    fig, axes = plt.subplots(2,5,figsize=(12,6))
    axes = axes.flatten()

    count = 0
    idx = 0

    with torch.no_grad():
        while count < num_images and idx < len(test_dataset):
            image, label = test_dataset[idx]
            idx += 1
            output = cnn_model(image.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            is_correct = (predicted == label)
            if show_incorrect and is_correct:
                continue
            if not show_incorrect and not is_correct:
                continue
            ax = axes[count]
            ax.imshow(image.squeeze(), cmap = 'grey')
            color = 'green' if is_correct else 'red'
            ax.set_title(f'True: {label}, Pred: {predicted}', color=color)
            ax.axis('off')
            count += 1

    plt.tight_layout()
    plt.show()

print("Showing correct predictions:")
visualize_predictions(num_images=10, show_incorrect=False)
print("\nShowing incorrect predictions:")
visualize_predictions(num_images=10, show_incorrect=True)    