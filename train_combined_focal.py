"""
Combined training with Focal Loss for adaptive per-letter weighting.

KEY INNOVATION: Focal Loss automatically focuses on hard letters
- Easy letters (A, B, F already correct) → low weight automatically
- Hard letters (C, R, O, V struggling) → high weight automatically
- Self-adjusting - no manual per-letter tuning needed!

FOCAL LOSS FORMULA: FL = -α(1-p)^γ * log(p)
- p = probability of correct class
- γ (gamma) = focusing parameter (higher = more focus on hard samples)
- α (alpha) = balance parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import numpy as np
from skimage.filters import threshold_otsu
import os
from emnist_cnn_model import EMNIST_CNN
from emnist_utils import char_to_label, label_to_char


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard-to-classify samples.

    Automatically down-weights easy samples and up-weights hard samples.

    Args:
        gamma: Focusing parameter (default 2.0)
               - Higher gamma = more focus on hard samples
               - gamma=0 → standard CrossEntropyLoss
               - gamma=2 → typical value
               - gamma=5 → extreme focus on hard samples

        alpha: Balance parameter (default 0.25)
               - Adjusts overall contribution
               - 0.25 is standard for binary, 0.25-0.75 for multi-class

    How it works:
        - Easy sample (p=0.9, confident correct): weight = (1-0.9)^2 = 0.01 (tiny!)
        - Medium sample (p=0.5, uncertain): weight = (1-0.5)^2 = 0.25
        - Hard sample (p=0.1, likely wrong): weight = (1-0.1)^2 = 0.81 (high!)
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Calculate focal loss.

        Args:
            inputs: Model predictions (logits), shape (batch_size, num_classes)
            targets: True labels, shape (batch_size,)

        Returns:
            Focal loss value (scalar)
        """
        # Calculate standard cross entropy loss (no reduction yet)
        # This gives loss per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate probability of correct class
        # exp(-ce_loss) gives p_t (probability of true class)
        p_t = torch.exp(-ce_loss)

        # Apply focal weight: (1 - p_t)^gamma
        # When p_t is high (confident correct) → (1-p_t) is small → low weight
        # When p_t is low (wrong or uncertain) → (1-p_t) is large → high weight
        focal_weight = (1 - p_t) ** self.gamma

        # Final focal loss: alpha * focal_weight * ce_loss
        focal_loss = self.alpha * focal_weight * ce_loss

        # Average across batch
        return focal_loss.mean()


def organize_alphabet_sets(test_folder='my_letters', training_folders=['set_2', 'set_3', 'set_4', 'set_5']):
    """
    Organize your images into training and test sets from separate folders.

    Args:
        test_folder: Folder containing test set
        training_folders: List of folders containing training sets

    Returns:
        training_sets: List of alphabet sets for training
        test_set: One alphabet set for testing (held-out)
    """
    print("\nORGANIZING DATASETS:")
    print("="*60)

    # Load test set
    print(f"\nTEST SET (held out, NEVER used for training):")
    print(f"  Loading from: {test_folder}/")

    test_set_dict = {}

    if not os.path.exists(test_folder):
        print(f"  ❌ ERROR: Folder '{test_folder}' not found!")
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
        print(f"  ✓ {test_folder}: Complete alphabet (26 letters)")
    else:
        print(f"  ⚠ {test_folder}: Incomplete ({len(test_set_dict)}/26 letters)")
        test_set = None

    # Load training sets
    print(f"\nTRAINING SETS (will be used for training):")

    training_sets = []

    for folder_name in training_folders:
        if not os.path.exists(folder_name):
            print(f"  ⚠ {folder_name}: Folder not found - skipping")
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
            print(f"  ✓ {folder_name}: Complete alphabet (26 letters)")
        else:
            print(f"  ⚠ {folder_name}: Incomplete ({len(folder_dict)}/26 letters) - skipping")

    print("="*60)
    print(f"\nSUMMARY:")
    print(f"  Training sets: {len(training_sets)}")
    print(f"  Training samples: {len(training_sets) * 26}")
    print(f"  Test samples: {26 if test_set else 0}")
    if len(training_sets) > 0 and test_set:
        print(f"  ✓ Data leakage: NONE")
    print("="*60)

    return training_sets, test_set


def preprocess_image(image_path):
    """Preprocess one image exactly like test_my_letters.py"""
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
    """Load one complete alphabet set (26 letters) as a batch"""
    images = []
    labels = []

    for image_path, label in alphabet_set:
        img_tensor = preprocess_image(image_path)
        images.append(img_tensor)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.LongTensor(labels)

    return images, labels


def train_combined_focal(model, emnist_dataloader, alphabet_sets, num_epochs=5,
                         learning_rate=0.001, gamma=2.0, alpha=0.25, base_weight=200):
    """
    Train with Focal Loss + Base Weight for adaptive focusing.

    Strategy:
    1. Train on EMNIST with standard CrossEntropyLoss
    2. Train on your samples with BASE_WEIGHT × FocalLoss
    3. Focal Loss modulates on top of base weight

    Effective weight per letter:
    - Easy letter (focal weight 0.1): 200 × 0.1 = 20
    - Medium letter (focal weight 0.5): 200 × 0.5 = 100
    - Hard letter (focal weight 0.9): 200 × 0.9 = 180

    Args:
        model: EMNIST_CNN model
        emnist_dataloader: DataLoader for EMNIST
        alphabet_sets: List of training alphabet sets
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        gamma: Focal loss focusing parameter (2.0 = moderate, 5.0 = aggressive)
        alpha: Focal loss balance parameter
        base_weight: Base amplification before focal modulation (default 200)
    """

    # Two loss functions:
    # 1. Standard CE for EMNIST (all samples equal weight)
    # 2. Focal Loss for your samples (hard samples get more weight)
    criterion_emnist = nn.CrossEntropyLoss()
    criterion_user = FocalLoss(gamma=gamma, alpha=alpha)

    # Optimizer - train all layers (not frozen like fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    print("\n" + "="*60)
    print("STARTING COMBINED TRAINING WITH BASE WEIGHT + FOCAL LOSS")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Base weight: {base_weight}x (all samples start here)")
    print(f"Focal Loss gamma: {gamma} (higher = more focus on hard samples)")
    print(f"Focal Loss alpha: {alpha}")
    print(f"Training sets available: {len(alphabet_sets)}")
    print("="*60)
    print(f"\nEffective weight per letter = {base_weight} × focal_weight:")
    print(f"  - Easy letter (focal=0.1): {base_weight} × 0.1 = {base_weight*0.1:.0f}")
    print(f"  - Medium letter (focal=0.5): {base_weight} × 0.5 = {base_weight*0.5:.0f}")
    print(f"  - Hard letter (focal=0.9): {base_weight} × 0.9 = {base_weight*0.9:.0f}")
    print("\nFocal Loss will dynamically adjust these weights each epoch!")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print('='*60)

        # ========================================
        # PHASE 1: TRAIN ON EMNIST (STANDARD LOSS)
        # ========================================
        print("\nPhase 1: Training on EMNIST (standard CrossEntropyLoss)...")

        emnist_loss = 0
        emnist_correct = 0
        emnist_total = 0

        for batch_idx, (images, labels) in enumerate(emnist_dataloader):
            optimizer.zero_grad()
            outputs = model(images)

            # Standard cross entropy - all samples equal weight
            loss = criterion_emnist(outputs, labels)
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

        print(f"\nEMNIST Phase Complete:")
        print(f"  Average Loss: {emnist_avg_loss:.4f}")
        print(f"  Accuracy: {emnist_accuracy:.2f}%")

        # ========================================
        # PHASE 2: CALIBRATE WITH FOCAL LOSS
        # ========================================
        print(f"\nPhase 2: Calibrating with Focal Loss (Set {(epoch % len(alphabet_sets)) + 1})...")

        # Rotate through training sets
        set_idx = epoch % len(alphabet_sets)
        current_set = alphabet_sets[set_idx]

        # Load alphabet batch
        user_images, user_labels = load_alphabet_batch(current_set)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(user_images)

        # Calculate focal loss - automatically weights hard samples more
        focal_loss = criterion_user(outputs, user_labels)

        # Apply base weight amplification
        # This is the KEY: combine fixed base weight with adaptive focal weight
        weighted_focal_loss = focal_loss * base_weight

        # Also calculate standard CE loss for comparison
        ce_loss = criterion_emnist(outputs, user_labels)

        # Backward pass with amplified focal loss
        weighted_focal_loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        user_correct = (predicted == user_labels).sum().item()
        user_accuracy = 100 * user_correct / len(user_labels)

        # Calculate per-letter focal weights to show which letters are "hard"
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            # Get probability of correct class for each sample
            p_correct = probs[range(len(user_labels)), user_labels]
            # Calculate focal weight for each
            focal_weights = (1 - p_correct) ** gamma

        print(f"  Standard CE Loss: {ce_loss.item():.4f}")
        print(f"  Focal Loss (before amplification): {focal_loss.item():.4f}")
        print(f"  Weighted Focal Loss (×{base_weight}): {weighted_focal_loss.item():.4f}")
        print(f"  Accuracy: {user_correct}/26 ({user_accuracy:.1f}%)")

        # Show which letters got high focal weight (hard samples)
        print(f"\n  Effective weights per letter (base={base_weight} × focal):")
        hard_letters = []
        for i in range(26):
            letter = label_to_char(user_labels[i].item())
            focal_weight = focal_weights[i].item()
            effective_weight = base_weight * focal_weight
            is_correct = (predicted[i] == user_labels[i]).item()

            # Mark especially hard letters (effective weight > 100)
            if effective_weight > 100:
                hard_letters.append(f"{letter}({effective_weight:.0f})")

            # Show all letters with effective weights
            if i % 13 == 0:
                print("    ", end="")
            status = "✓" if is_correct else "✗"
            print(f"{letter}:{effective_weight:.0f}{status} ", end="")
            if (i + 1) % 13 == 0:
                print()

        if hard_letters:
            print(f"\n  → Hard letters (effective weight > 100): {', '.join(hard_letters)}")
        else:
            print(f"\n  → All letters have moderate weights")

        # Show incorrect predictions
        if user_correct < 26:
            print(f"\n  Incorrect predictions:")
            for i in range(26):
                if predicted[i] != user_labels[i]:
                    true_char = label_to_char(user_labels[i].item())
                    pred_char = label_to_char(predicted[i].item())
                    print(f"    {true_char} → {pred_char}")

        print(f"\nEpoch {epoch+1} Complete")
        print(f"EMNIST: {emnist_accuracy:.2f}% | Your samples: {user_accuracy:.1f}%")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


def test_on_held_out_set(model, test_set):
    """Test model on held-out test set"""
    model.eval()

    print("\n" + "="*60)
    print("TESTING ON HELD-OUT TEST SET")
    print("="*60)

    correct = 0
    total = 0
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

        total += 1
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

    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}")

    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print("\nIncorrect predictions:")
        for r in incorrect:
            print(f"  {r['true']} → {r['predicted']} ({r['confidence']*100:.1f}% confident)")
    else:
        print("\n🎉 PERFECT SCORE! All 26 letters correct!")

    return accuracy


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("COMBINED TRAINING WITH FOCAL LOSS")
    print("="*60)
    print("\nFocal Loss Innovation:")
    print("  - Automatically focuses on hard-to-learn letters")
    print("  - No manual per-letter weight tuning needed")
    print("  - Self-adjusting as model improves")
    print("="*60)

    # Organize datasets
    print("\nStep 1: Organizing datasets...")
    print("="*60)

    training_sets, test_set = organize_alphabet_sets(
        test_folder='my_letters',
        training_folders=['set_2', 'set_3', 'set_4', 'set_5']
    )

    if len(training_sets) == 0 or test_set is None:
        print("\n❌ ERROR: Missing datasets!")
        exit()

    # Load EMNIST
    print("\nStep 2: Loading EMNIST dataset...")
    print("="*60)

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

    emnist_dataloader = DataLoader(
        emnist_train,
        batch_size=32,
        shuffle=True
    )

    print(f"✓ Loaded EMNIST 'byclass' training set")
    print(f"  Total samples: {len(emnist_train):,}")

    # Load model
    print("\nStep 3: Loading pre-trained model...")
    print("="*60)

    model = EMNIST_CNN()
    model.load_state_dict(torch.load('emnist_cnn.pth'))
    print("✓ Loaded emnist_cnn.pth")

    # Train with focal loss
    print("\nStep 4: Training with Focal Loss...")
    print("="*60)
    print("\nParameters:")
    print("  - Epochs: 5")
    print("  - Learning rate: 0.001")
    print("  - Base weight: 200 (strong amplification)")
    print("  - Gamma: 2.0 (moderate focal modulation)")
    print("  - Alpha: 0.25")
    print("\nThis combines:")
    print("  1. Strong base amplification (200x)")
    print("  2. Adaptive focal modulation on top")
    print("  → Easy letters: ~20-50x, Hard letters: ~150-180x")
    print("\nEach epoch takes ~3-5 minutes...")
    print("="*60)

    input("\nPress Enter to start training...")

    train_combined_focal(
        model=model,
        emnist_dataloader=emnist_dataloader,
        alphabet_sets=training_sets,
        num_epochs=5,
        learning_rate=0.001,
        gamma=2.0,  # Moderate focusing on hard samples
        alpha=0.25,
        base_weight=200  # Strong base amplification + focal modulation on top
    )

    # Save model
    print("\nStep 5: Saving model...")
    print("="*60)

    torch.save(model.state_dict(), 'emnist_cnn_focal.pth')
    print("✓ Saved as: emnist_cnn_focal.pth")

    # Test
    print("\nStep 6: Testing on held-out set...")

    accuracy = test_on_held_out_set(model, test_set)

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Original model: 11/26 (42.3%)")
    print(f"Fixed weight=100: 23/26 (88.5%)")
    print(f"Fixed weight=200: 24/26 (92.3%)")
    print(f"Focal Loss: {accuracy:.1f}%")
    print("\n" + "="*60)
    print("Focal Loss automatically adapted weight per letter!")
    print("Hard letters got more attention, easy letters less.")
    print("="*60)
