import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cnn_model import CNN

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img)
    img = img / 255.0
    img = Image.fromarray((img * 255).astype('uint8'))
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    return img

def predict_digit(model, image_path):
    img = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item(), img

def visualize_prediction(image_path, predicted):
    original = Image.open(image_path).convert('L')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(original, cmap='gray')
    ax2.set_title('Original Image')
    ax1.axis('off')

    processed = preprocess_image(image_path)
    ax2.imshow(processed.squeeze(), cmap='gray')
    ax2.set_title(f'Processed (predicted: {predicted})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()

    #path = Path('my_digits')
    #for file in path.iterdir():
    image_path = 'my_digits/digit_9.png'
    predicted, img = predict_digit(model, image_path)
    print(f"Predicted digit: {predicted}")
    visualize_prediction(image_path, predicted)