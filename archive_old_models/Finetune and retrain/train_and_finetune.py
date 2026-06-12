"""
Combined training: EMNIST + Your handwriting with weighted end-of-epoch calibration.

STRATEGY:
1. Each epoch: Train on full EMNIST dataset (normal weight)
2. End of epoch: Train on ONE set of your alphabet (26 letters, high weight)
3. Rotate through your 4 alphabet sets across epochs
4. High weight amplifies gradient from your samples without overfitting

BENEFITS:
- No catastrophic forgetting (always trains on EMNIST)
- No overfitting (only 26 samples per epoch, not repeated)
- Variety (different alphabet set each epoch)
- Strong adaptation (high weight makes your samples impactful)
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
    """
    Organize your images into training and test sets from separate folders.

    IMPORTANT: Prevents data leakage by separating train/test
    - Training: set_2, set_3, set_4, set_5 folders = 4 sets × 26 letters = 104 samples
    - Testing: my_letters folder = 26 samples NEVER seen during training

    Args:
        test_folder: Folder containing test set (default: 'my_letters')
        training_folders: List of folders containing training sets

    Returns:
        training_sets: List of 4 alphabet sets for training
        test_set: One alphabet set for testing (held-out)
    """
    print("\nORGANIZING DATASETS:")
    print("="*60)

    # ========================================
    # LOAD TEST SET (my_letters)
    # ========================================
    print(f"\nTEST SET (held out, NEVER used for training):")
    print(f"  Loading from: {test_folder}/")

    test_set_dict = {}

    # Check if test folder exists
    if not os.path.exists(test_folder):
        print(f"  ❌ ERROR: Folder '{test_folder}' not found!")
        return [], None


    # Load all letters from test folder
    for filename in os.listdir(test_folder):
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue

        # Parse filename: "A.png", "B.png", etc.
        basename = filename.replace('.png', '').replace('.jpg', '')
        letter = basename[0].upper()

        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            image_path = os.path.join(test_folder, filename)
            label = char_to_label(letter)
            test_set_dict[letter] = (image_path, label)

    # Convert to ordered list A-Z
    if len(test_set_dict) == 26:
        test_set = [(test_set_dict[letter]) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        print(f"  ✓ {test_folder}: Complete alphabet (26 letters)")
        print(f"  → This set will ONLY be used for final testing")
    else:
        print(f"  ⚠ {test_folder}: Incomplete ({len(test_set_dict)}/26 letters)")
        print(f"  Missing letters: {set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(test_set_dict.keys())}")
        test_set = None

    # ========================================
    # LOAD TRAINING SETS (set_2, set_3, set_4, set_5)
    # ========================================
    print(f"\nTRAINING SETS (will be used for training):")

    training_sets = []

    for folder_name in training_folders:
        # Check if folder exists
        if not os.path.exists(folder_name):
            print(f"  ⚠ {folder_name}: Folder not found - skipping")
            continue

        # Load letters from this folder
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

        # Check if complete
        if len(folder_dict) == 26:
            alphabet = [(folder_dict[letter]) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
            training_sets.append(alphabet)
            print(f"  ✓ {folder_name}: Complete alphabet (26 letters)")
        else:
            print(f"  ⚠ {folder_name}: Incomplete ({len(folder_dict)}/26 letters) - skipping")
            if len(folder_dict) > 0:
                missing = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(folder_dict.keys())
                print(f"    Missing: {missing}")

    # ========================================
    # SUMMARY
    # ========================================
    print("="*60)
    print(f"\nSUMMARY:")
    print(f"  Training sets found: {len(training_sets)}")
    print(f"  Training samples: {len(training_sets) * 26}")
    print(f"  Test samples: {26 if test_set else 0}")

    if len(training_sets) > 0 and test_set:
        print(f"  ✓ Data leakage: NONE - test set never seen during training!")
    else:
        print(f"  ❌ ERROR: Missing training sets or test set")

    print("="*60)

    return training_sets, test_set


def preprocess_image(image_path):
    """
    Preprocess one image exactly like test_my_letters.py

    Returns:
        Tensor of shape (1, 28, 28) ready for model input
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 28x28 (model input size)
    img = img.resize((28, 28))

    # Add blur to smooth edges (matches EMNIST gradients)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Convert to numpy for thresholding
    img_array = np.array(img)

    # Otsu's automatic thresholding
    # Finds optimal threshold for this specific image
    threshold = threshold_otsu(img_array)

    # Binarize: convert to pure black/white
    img_array = np.where(img_array > threshold, 255, 0)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Convert to tensor with shape (1, 28, 28)
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0)

    return img_tensor


def load_alphabet_batch(alphabet_set):
    """
    Load one complete alphabet set (26 letters) as a batch.

    Args:
        alphabet_set: List of 26 (image_path, label) tuples

    Returns:
        images: Tensor of shape (26, 1, 28, 28)
        labels: Tensor of shape (26,)
    """
    images = []
    labels = []

    for image_path, label in alphabet_set:
        # Preprocess image
        img_tensor = preprocess_image(image_path)
        images.append(img_tensor)
        labels.append(label)

    # Stack into batch
    # 26 individual (1, 28, 28) -> (26, 1, 28, 28)
    images = torch.stack(images)
    labels = torch.LongTensor(labels)

    return images, labels


def train_combined(model, emnist_dataloader, alphabet_sets, num_epochs=5,
                   learning_rate=0.001, user_weight=100):
    """
    Train on EMNIST + your handwriting with weighted calibration.

    Training strategy per epoch:
    1. Train on full EMNIST dataset (normal loss weight)
    2. Train on ONE alphabet set from user (amplified loss weight)
    3. Rotate through alphabet sets across epochs

    Args:
        model: EMNIST_CNN model (loaded with pretrained weights)
        emnist_dataloader: DataLoader for EMNIST training set
        alphabet_sets: List of 4 alphabet sets (each with 26 letters)
        num_epochs: Number of training epochs (default 5)
        learning_rate: Learning rate (default 0.001)
        user_weight: Weight multiplier for your samples (default 100)

    Why this works:
    - Model learns general patterns from 600k EMNIST samples
    - Then "calibrates" to your style with weighted 26 samples
    - High weight = gradient from your samples is amplified 100x
    - Rotating sets prevents overfitting to one specific set
    """

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # All layers trainable (not frozen like fine-tuning)
    # Why? We're training on huge EMNIST dataset, so safe to update all layers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set to training mode
    model.train()

    print("\n" + "="*60)
    print("STARTING COMBINED TRAINING")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"User sample weight: {user_weight}x")
    print(f"Alphabet sets available: {len(alphabet_sets)}")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print('='*60)

        # ========================================
        # PHASE 1: TRAIN ON EMNIST (NORMAL WEIGHT)
        # ========================================
        print("\nPhase 1: Training on EMNIST dataset...")

        emnist_loss = 0
        emnist_correct = 0
        emnist_total = 0

        # Train on full EMNIST dataset
        # This takes ~2-5 minutes per epoch depending on hardware
        for batch_idx, (images, labels) in enumerate(emnist_dataloader):
            # Standard training step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Normal weight (1.0)
            loss.backward()
            optimizer.step()

            # Track statistics
            emnist_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            emnist_total += labels.size(0)
            emnist_correct += (predicted == labels).sum().item()

            # Progress indicator every 500 batches
            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch {batch_idx+1}/{len(emnist_dataloader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Acc={100*emnist_correct/emnist_total:.2f}%")

        # EMNIST phase summary
        emnist_avg_loss = emnist_loss / len(emnist_dataloader)
        emnist_accuracy = 100 * emnist_correct / emnist_total

        print(f"\nEMNIST Phase Complete:")
        print(f"  Average Loss: {emnist_avg_loss:.4f}")
        print(f"  Accuracy: {emnist_accuracy:.2f}%")

        # ========================================
        # PHASE 2: CALIBRATE ON YOUR ALPHABET (HIGH WEIGHT)
        # ========================================
        print(f"\nPhase 2: Calibrating on your handwriting (Set {(epoch % len(alphabet_sets)) + 1})...")

        # Rotate through your alphabet sets
        # Epoch 0 -> Set 0, Epoch 1 -> Set 1, etc.
        # After seeing all sets, cycle back to Set 0
        set_idx = epoch % len(alphabet_sets)
        current_set = alphabet_sets[set_idx]

        # Load your 26-letter alphabet as one batch
        user_images, user_labels = load_alphabet_batch(current_set)

        # Forward pass on your samples
        optimizer.zero_grad()
        outputs = model(user_images)
        loss = criterion(outputs, user_labels)

        # AMPLIFY THE LOSS by the weight factor
        # Why multiply? Gradient = dL/dW
        # If we multiply L by 100, gradient becomes 100x larger
        # Model updates weights 100x more based on these samples
        weighted_loss = loss * user_weight

        # Backward pass with amplified gradient
        weighted_loss.backward()
        optimizer.step()

        # Calculate accuracy on your samples
        _, predicted = torch.max(outputs, 1)
        user_correct = (predicted == user_labels).sum().item()
        user_accuracy = 100 * user_correct / len(user_labels)

        print(f"  Original loss: {loss.item():.4f}")
        print(f"  Weighted loss: {weighted_loss.item():.4f} (x{user_weight})")
        print(f"  Your samples accuracy: {user_correct}/26 ({user_accuracy:.1f}%)")

        # Show which letters were wrong (if any)
        if user_correct < 26:
            print("  Incorrect predictions:")
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
    """
    Test model on held-out test set (NEVER seen during training).

    Args:
        model: Trained model
        test_set: List of 26 (image_path, label) tuples

    Returns:
        accuracy: Percentage correct
    """
    model.eval()

    print("\n" + "="*60)
    print("TESTING ON HELD-OUT TEST SET")
    print("="*60)
    print("(This set was NEVER seen during training - true generalization test)")
    print("="*60)

    correct = 0
    total = 0
    results = []

    # Test on each letter in test set
    for image_path, true_label in test_set:
        # Get letter for printing
        true_char = label_to_char(true_label)

        # Preprocess image
        img = preprocess_image(image_path)

        # Predict
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

    # Show incorrect predictions
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
    print("COMBINED TRAINING: EMNIST + YOUR HANDWRITING")
    print("="*60)
    print("\nThis script trains on:")
    print("1. Full EMNIST dataset (600k samples, normal weight)")
    print("2. Your alphabet sets (26 samples/epoch, 100x weight)")
    print("\nRotates through your 4 alphabet sets across epochs.")
    print("="*60)

    # ========================================
    # STEP 1: ORGANIZE YOUR ALPHABET SETS
    # ========================================
    print("\nStep 1: Organizing your alphabet sets...")
    print("="*60)

    training_sets, test_set = organize_alphabet_sets(
        test_folder='my_letters',
        training_folders=['set_2', 'set_3', 'set_4', 'set_5']
    )

    if len(training_sets) == 0:
        print("\n❌ ERROR: No complete training sets found!")
        print("Make sure you have folders: set_2, set_3, set_4, set_5")
        print("Each should contain: A.png, B.png, ..., Z.png")
        exit()

    if test_set is None:
        print("\n❌ ERROR: Test set incomplete!")
        print("Make sure my_letters folder has: A.png, B.png, ..., Z.png")
        exit()

    print(f"\n✓ Ready to train on {len(training_sets)} sets, test on 1 held-out set")

    # ========================================
    # STEP 2: LOAD EMNIST DATASET
    # ========================================
    print("\nStep 2: Loading EMNIST dataset...")
    print("="*60)

    # Same transform as original training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),  # Fix orientation
        transforms.Lambda(lambda x: 1 - x)                # Invert colors
    ])

    emnist_train = datasets.EMNIST(
        root='./data',
        split='byclass',
        train=True,
        download=False,
        transform=transform
    )

    # Create dataloader
    # batch_size=32 matches original training
    emnist_dataloader = DataLoader(
        emnist_train,
        batch_size=32,
        shuffle=True
    )

    print(f"✓ Loaded EMNIST 'byclass' training set")
    print(f"  Total samples: {len(emnist_train):,}")
    print(f"  Batches per epoch: {len(emnist_dataloader):,}")

    # ========================================
    # STEP 3: LOAD PRE-TRAINED MODEL
    # ========================================
    print("\nStep 3: Loading pre-trained model...")
    print("="*60)

    model = EMNIST_CNN()
    model.load_state_dict(torch.load('emnist_cnn.pth'))
    print("✓ Loaded emnist_cnn.pth (86% EMNIST accuracy)")

    # ========================================
    # STEP 4: COMBINED TRAINING
    # ========================================
    print("\nStep 4: Starting combined training...")
    print("="*60)
    print("\nTraining parameters:")
    print("  - Epochs: 5")
    print("  - Learning rate: 0.001")
    print("  - User sample weight: 100x")
    print("\nEach epoch takes ~3-5 minutes (depending on hardware)")
    print("="*60)

    input("\nPress Enter to start training...")

    train_combined(
        model=model,
        emnist_dataloader=emnist_dataloader,
        alphabet_sets=training_sets,  # Changed from alphabet_sets
        num_epochs=5,
        learning_rate=0.001,
        user_weight= 200
    )

    # ========================================
    # STEP 5: SAVE MODEL
    # ========================================
    print("\nStep 5: Saving trained model...")
    print("="*60)

    torch.save(model.state_dict(), 'emnist_cnn_finetuned.pth')
    print("✓ Saved as: emnist_cnn_finetuned.pth")

    # ========================================
    # STEP 6: TEST ON HELD-OUT TEST SET
    # ========================================
    print("\nStep 6: Testing on held-out test set (my_letters/)...")

    accuracy = test_on_held_out_set(model, test_set)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Original EMNIST model: 86% on EMNIST test set")
    print(f"First attempt on your letters: 11/26 (42.3%)")
    print(f"After fine-tuning: 23/26 (88.5%) - but had data leakage!")
    print(f"After combined training: {accuracy:.1f}% - TRUE generalization!")
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("This accuracy is on a HELD-OUT test set (my_letters/)")
    print("that was NEVER seen during training.")
    print("This is your model's TRUE ability to generalize!")
    print("\n" + "="*60)
    print("TO USE THIS MODEL:")
    print("="*60)
    print("In test_my_letters.py, change line ~121:")
    print("  FROM: model.load_state_dict(torch.load('emnist_cnn.pth'))")
    print("  TO:   model.load_state_dict(torch.load('emnist_cnn_combined.pth'))")
    print("="*60)
