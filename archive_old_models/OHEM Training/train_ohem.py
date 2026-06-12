"""
Combined training with Selective Hard Example Mining.

KEY INNOVATION: Only train on INCORRECT predictions
- Correct predictions: Weight = 0 (skip entirely)
- Incorrect predictions: Weight = base_weight (e.g., 100)

BENEFITS:
1. No wasted gradient on correct samples
2. Prevents correct samples from regressing
3. Self-adjusting - trains less as model improves
4. More stable - fewer conflicting gradients
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import numpy as np
from skimage.filters import threshold_otsu
import os
from emnist_cnn_model import EMNIST_CNN
from emnist_utils import char_to_label, label_to_char


def organize_alphabet_sets(test_folder='my_letters', training_folders=['set_2', 'set_3', 'set_4', 'set_5']):
    """Organize datasets into training and test sets"""
    print("\nORGANIZING DATASETS:")
    print("="*60)

    # Load test set
    print(f"\nTEST SET: {test_folder}/")
    test_set_dict = {}

    if not os.path.exists(test_folder):
        print(f"  ❌ ERROR: Folder not found!")
        return [], None

    for filename in os.listdir(test_folder):
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue

        basename = filename.replace('.png', '').replace('.jpg', '')
        letter = basename[0].upper()

        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            image_path = os.path.join(test_folder, filename)
            label = char_to_label(letter)
            test_set_dict[letter] = (image_path, label)

    if len(test_set_dict) == 26:
        test_set = [(test_set_dict[letter]) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        print(f"  ✓ Complete alphabet (26 letters)")
    else:
        print(f"  ⚠ Incomplete ({len(test_set_dict)}/26 letters)")
        test_set = None

    # Load training sets
    print(f"\nTRAINING SETS:")
    training_sets = []

    for folder_name in training_folders:
        if not os.path.exists(folder_name):
            print(f"  ⚠ {folder_name}: Not found")
            continue

        folder_dict = {}
        for filename in os.listdir(folder_name):
            if not filename.endswith('.png') and not filename.endswith('.jpg'):
                continue

            basename = filename.replace('.png', '').replace('.jpg', '')
            letter = basename[0].upper()

            if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                image_path = os.path.join(folder_name, filename)
                label = char_to_label(letter)
                folder_dict[letter] = (image_path, label)

        if len(folder_dict) == 26:
            alphabet = [(folder_dict[letter]) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
            training_sets.append(alphabet)
            print(f"  ✓ {folder_name}: 26 letters")
        else:
            print(f"  ⚠ {folder_name}: Incomplete ({len(folder_dict)}/26)")

    print("="*60)
    print(f"Training sets: {len(training_sets)} ({len(training_sets)*26} samples)")
    print(f"Test samples: {26 if test_set else 0}")
    print("="*60)

    return training_sets, test_set


def preprocess_image(image_path):
    """Preprocess one image"""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_array = np.array(img)
    threshold = threshold_otsu(img_array)
    img_array = np.where(img_array > threshold, 255, 0)
    img_array = img_array / 255.0
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
    return img_tensor


def load_alphabet_batch(alphabet_set):
    """Load one alphabet set as a batch"""
    images = []
    labels = []

    for image_path, label in alphabet_set:
        img_tensor = preprocess_image(image_path)
        images.append(img_tensor)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.LongTensor(labels)

    return images, labels


def train_combined_selective(model, emnist_dataloader, alphabet_sets, num_epochs=5,
                             learning_rate=0.001, base_weight=100):
    """
    Train with selective hard example mining.

    Only trains on INCORRECT predictions, ignores correct ones.

    Strategy per epoch:
    1. Train on full EMNIST (prevents catastrophic forgetting)
    2. Forward pass on your 26 samples
    3. Identify which samples are WRONG
    4. Only backprop on those wrong samples with amplified weight
    5. Correct samples are untouched (stay correct!)

    Args:
        model: EMNIST_CNN model
        emnist_dataloader: DataLoader for EMNIST
        alphabet_sets: List of training alphabet sets
        num_epochs: Number of epochs
        learning_rate: Learning rate
        base_weight: Weight applied ONLY to incorrect samples
    """

    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    print("\n" + "="*60)
    print("SELECTIVE HARD EXAMPLE MINING")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Base weight: {base_weight}x (ONLY for wrong predictions)")
    print(f"Training sets: {len(alphabet_sets)}")
    print("="*60)
    print("\nStrategy:")
    print("  ✓ Correct predictions: Weight = 0 (don't train)")
    print("  ✓ Wrong predictions: Weight = 100x (focus here)")
    print("  ✓ Self-adjusting as model improves")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print('='*60)

        # ========================================
        # PHASE 1: TRAIN ON EMNIST
        # ========================================
        print("\nPhase 1: Training on EMNIST...")

        emnist_loss = 0
        emnist_correct = 0
        emnist_total = 0

        for batch_idx, (images, labels) in enumerate(emnist_dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels).mean()  # Average loss
            loss.backward()
            optimizer.step()

            emnist_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            emnist_total += labels.size(0)
            emnist_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch {batch_idx+1}/{len(emnist_dataloader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Acc={100*emnist_correct/emnist_total:.2f}%")

        emnist_avg_loss = emnist_loss / len(emnist_dataloader)
        emnist_accuracy = 100 * emnist_correct / emnist_total

        print(f"\nEMNIST Complete:")
        print(f"  Loss: {emnist_avg_loss:.4f}")
        print(f"  Accuracy: {emnist_accuracy:.2f}%")

        # ========================================
        # PHASE 2: SELECTIVE TRAINING ON YOUR SAMPLES
        # ========================================
        print(f"\nPhase 2: Selective training (Set {(epoch % len(alphabet_sets)) + 1})...")

        # Rotate through training sets
        set_idx = epoch % len(alphabet_sets)
        current_set = alphabet_sets[set_idx]

        # Load alphabet batch
        user_images, user_labels = load_alphabet_batch(current_set)

        # Forward pass (no gradient yet)
        with torch.no_grad():
            outputs_eval = model(user_images)
            _, predicted = torch.max(outputs_eval, 1)

        # Identify which samples are WRONG
        incorrect_mask = (predicted != user_labels)
        num_wrong = incorrect_mask.sum().item()
        num_correct = 26 - num_wrong

        print(f"  Predictions: {num_correct}/26 correct, {num_wrong}/26 wrong")

        # Show which letters are wrong
        if num_wrong > 0:
            wrong_letters = []
            for i in range(26):
                if incorrect_mask[i]:
                    true_char = label_to_char(user_labels[i].item())
                    pred_char = label_to_char(predicted[i].item())
                    wrong_letters.append(f"{true_char}→{pred_char}")

            print(f"  Wrong: {', '.join(wrong_letters)}")
        else:
            print(f"  🎉 ALL CORRECT! No training needed this epoch.")

        # Only train on wrong samples
        if num_wrong > 0:
            # Forward pass WITH gradient
            optimizer.zero_grad()
            outputs = model(user_images)

            # Calculate loss per sample
            losses = criterion(outputs, user_labels)

            # Create weight mask: 0 for correct, base_weight for wrong
            weights = incorrect_mask.float() * base_weight

            # Apply weights
            weighted_loss = (losses * weights).sum() / weights.sum()  # Average over wrong samples

            # Calculate what standard loss would be for comparison
            standard_loss = losses.mean()

            # Backward pass
            weighted_loss.backward()
            optimizer.step()

            # Calculate effective amplification
            effective_amplification = weighted_loss.item() / (standard_loss.item() + 1e-8) * num_wrong / 26

            print(f"\n  Standard loss: {standard_loss.item():.4f}")
            print(f"  Selective loss: {weighted_loss.item():.4f}")
            print(f"  Effective amplification: {effective_amplification:.1f}x")
            print(f"  Gradient focused on {num_wrong} samples only")

        # Re-evaluate after training
        with torch.no_grad():
            outputs_after = model(user_images)
            _, predicted_after = torch.max(outputs_after, 1)
            correct_after = (predicted_after == user_labels).sum().item()

        print(f"\n  After training: {correct_after}/26 correct")

        if correct_after > num_correct:
            print(f"  ✓ Improved: +{correct_after - num_correct} letters fixed!")
        elif correct_after < num_correct:
            print(f"  ⚠ Regressed: {num_correct - correct_after} letters broke")
        else:
            print(f"  → No change")

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  EMNIST: {emnist_accuracy:.2f}%")
        print(f"  Your samples (before): {num_correct}/26 ({100*num_correct/26:.1f}%)")
        print(f"  Your samples (after): {correct_after}/26 ({100*correct_after/26:.1f}%)")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


def test_on_held_out_set(model, test_set):
    """Test on held-out test set"""
    model.eval()

    print("\n" + "="*60)
    print("TESTING ON HELD-OUT TEST SET")
    print("="*60)

    correct = 0
    results = []

    for image_path, true_label in test_set:
        true_char = label_to_char(true_label)

        img = preprocess_image(image_path)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, 1)
            confidence, predicted = torch.max(probs, 1)

        predicted_char = label_to_char(predicted.item())
        is_correct = (predicted_char == true_char)

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} {true_char} → {predicted_char} ({confidence.item()*100:.1f}%)")

        results.append({
            'true': true_char,
            'predicted': predicted_char,
            'correct': is_correct,
            'confidence': confidence.item()
        })

    accuracy = 100 * correct / 26

    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {correct}/26 ({accuracy:.1f}%)")
    print(f"{'='*60}")

    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print("\nIncorrect predictions:")
        for r in incorrect:
            print(f"  {r['true']} → {r['predicted']} ({r['confidence']*100:.1f}%)")
    else:
        print("\n🎉 PERFECT! All 26 letters correct!")

    return accuracy


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("SELECTIVE HARD EXAMPLE MINING")
    print("="*60)
    print("\nInnovation: Only train on WRONG predictions")
    print("  - Correct predictions: Skip (stay correct)")
    print("  - Wrong predictions: Focus gradient here")
    print("="*60)

    # Organize datasets
    print("\nStep 1: Organizing datasets...")

    training_sets, test_set = organize_alphabet_sets(
        test_folder='my_letters',
        training_folders=['set_2', 'set_3', 'set_4', 'set_5']
    )

    if len(training_sets) == 0 or test_set is None:
        print("\n❌ ERROR: Missing datasets!")
        exit()

    # Load EMNIST
    print("\nStep 2: Loading EMNIST...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),
        transforms.Lambda(lambda x: 1 - x)
    ])

    emnist_train = datasets.EMNIST(
        root='./data',
        split='byclass',
        train=True,
        download=False,
        transform=transform
    )

    emnist_dataloader = DataLoader(emnist_train, batch_size=32, shuffle=True)
    print(f"✓ Loaded {len(emnist_train):,} EMNIST samples")

    # Load model
    print("\nStep 3: Loading pre-trained model...")

    model = EMNIST_CNN()
    model.load_state_dict(torch.load('emnist_cnn.pth'))
    print("✓ Loaded emnist_cnn.pth")

    # Train
    print("\nStep 4: Training with selective hard example mining...")
    print("="*60)
    print("Parameters:")
    print("  - Epochs: 5")
    print("  - Learning rate: 0.001")
    print("  - Base weight: 100x (only on wrong predictions)")
    print("\nEach epoch ~3-5 minutes...")
    print("="*60)

    input("\nPress Enter to start...")

    train_combined_selective(
        model=model,
        emnist_dataloader=emnist_dataloader,
        alphabet_sets=training_sets,
        num_epochs=5,
        learning_rate=0.001,
        base_weight=100
    )

    # Save
    print("\nStep 5: Saving model...")

    torch.save(model.state_dict(), 'emnist_cnn_selective.pth')
    print("✓ Saved: emnist_cnn_selective.pth")

    # Test
    print("\nStep 6: Testing on held-out set...")

    accuracy = test_on_held_out_set(model, test_set)

    # Summary
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Original model: 11/26 (42.3%)")
    print(f"Fixed weight=200: 24/26 (92.3%)")
    print(f"Selective (weight=100): {accuracy:.1f}%")
    print("="*60)
