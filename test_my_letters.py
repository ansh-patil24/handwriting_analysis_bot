import torch
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from emnist_utils import label_to_char, char_to_label
from skimage.filters import threshold_otsu
from emnist_cnn_model import EMNIST_CNN

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_array = np.array(img)
    threshold = threshold_otsu(img_array)
    img_array = np.where(img_array > threshold, 255, 0)
    img = img_array/255.0
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    return img

def predict_char(model, image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, 1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), img

def visualize_prediction(image_path, predicted_label, confidence):
    original = Image.open(image_path).convert('L')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    processed = preprocess_image(image_path)
    ax2.imshow(processed.squeeze(), cmap='grey')

    predicted_letter = label_to_char(predicted_label)
    ax2.set_title(f'Predicted: {predicted_letter} ({confidence*100:.1f}% confident)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def test_all_uppercase_letters(model, folder='my_letters'):
    results = []

    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        possible_paths = [f'{folder}/{char}.png', f'{folder}/letter_{char}.png',f'{folder}/{char}.jpg']
        image_path = None
        for path in possible_paths:
            try:
                Image.open(path)
                image_path = path
                break
            except:
                continue
        
        if image_path is None:
            print(f"Skipping {char} - image not found")
            continue

        predicted_label, confidence, _ = predict_char(model, image_path)
        predicted_char = label_to_char(predicted_label)
        is_correct = (predicted_char == char)

        results.append({'true': char, 'predicted': predicted_char, 'confidence': confidence, 'correct': is_correct})
        status = "✓" if is_correct else "✗"
        print(f"{status} {char} → {predicted_char} ({confidence*100:.1f}% confident)")
    
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Accuray: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*50}")

    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print("\nIncorrect predictions:")
        for r in incorrect:
            print(f" {r['true']} → {r['predicted']} ({r['confidence']*100:.1f}% confident)")
    return results

if __name__ == "__main__":
    model = EMNIST_CNN()
    model.load_state_dict(torch.load('emnist_cnn_focal.pth'))
    model.eval()
    
    test_all_uppercase_letters(model, folder='my_letters')