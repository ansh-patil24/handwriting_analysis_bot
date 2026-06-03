"""
Fine-tuning script for adapting EMNIST model to your handwriting style.

KEY CONCEPT: Fine-tuning vs Training from Scratch
- Training from scratch: Start with random weights, learn everything
- Fine-tuning: Start with trained model, only adjust to new data
- Advantage: Need much less data (50 samples vs 60,000)

APPROACH:
1. Load pre-trained EMNIST model (already knows features)
2. FREEZE early layers (keep feature detection - edges, curves)
3. TRAIN last layers (adapt decision-making to your style)
4. Use LOW learning rate (gentle adjustments, don't break existing knowledge)
5. Use FEW epochs (avoid "catastrophic forgetting" of EMNIST)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import numpy as np
from skimage.filters import threshold_otsu
import os
from emnist_cnn_model import EMNIST_CNN
from emnist_utils import char_to_label, label_to_char


class MyHandwritingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading your handwritten letters.

    PyTorch Dataset requirements:
    - __init__: Initialize and find all data
    - __len__: Return total number of samples
    - __getitem__: Load and return one sample

    Why custom? PyTorch's built-in EMNIST dataset doesn't know about
    your folder structure or filename conventions.
    """

    def __init__(self, folder='my_letters'):
        """
        Find all letter images in the folder and parse filenames.

        Args:
            folder: Path to folder containing your images
        """
        self.samples = []  # List of (image_path, label, letter) tuples

        # Scan folder for all image files
        for filename in os.listdir(folder):
            # Only process image files
            if not filename.endswith('.png') and not filename.endswith('.jpg'):
                continue

            # Parse filename to extract the letter
            # Examples: "C.png" -> "C", "C2.png" -> "C", "C3.png" -> "C"
            basename = filename.replace('.png', '').replace('.jpg', '')

            # Get first character (the letter)
            # "C" -> "C", "C2" -> "C", "C3" -> "C"
            letter = basename[0].upper()

            # Only process uppercase letters A-Z
            if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                image_path = os.path.join(folder, filename)

                # Convert letter to numeric label
                # A->10, B->11, ..., Z->35 (EMNIST 'byclass' encoding)
                label = char_to_label(letter)

                # Store: (path to image, numeric label, letter for printing)
                self.samples.append((image_path, label, letter))

        print(f"Found {len(self.samples)} images")

        # Show how many samples per letter (debugging info)
        from collections import Counter
        letter_counts = Counter([s[2] for s in self.samples])
        print(f"\nLetter distribution:")
        for letter in sorted(letter_counts.keys()):
            print(f"  {letter}: {letter_counts[letter]} samples")

    def __len__(self):
        """
        Return total number of samples.

        PyTorch uses this to know how many batches to create.
        Called automatically by DataLoader.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and preprocess one image.

        Args:
            idx: Index of sample to load (0 to len-1)

        Returns:
            (image_tensor, label): Preprocessed image and its numeric label

        CRITICAL: Preprocessing MUST match test_my_letters.py exactly!
        Any mismatch = model sees different data = predictions fail.
        """
        # Get the sample info
        image_path, label, letter = self.samples[idx]

        # Step 1: Load image and convert to grayscale
        # 'L' mode = 8-bit grayscale (0=black, 255=white)
        img = Image.open(image_path).convert('L')

        # Step 2: Resize to 28x28 (EMNIST/MNIST standard size)
        # Model expects 28x28 input
        img = img.resize((28, 28))

        # Step 3: Add blur to create smooth edges like EMNIST
        # Radius=0.5 is subtle - removes sharp pixelation
        # EMNIST has gradients from scanning, we simulate this
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Step 4: Convert to numpy array for thresholding
        img_array = np.array(img)

        # Step 5: Otsu's thresholding - automatic per-image threshold
        # Finds optimal threshold to separate foreground/background
        # Better than fixed threshold=128 because it adapts to each image
        threshold = threshold_otsu(img_array)

        # Step 6: Binarize - convert to pure black (0) or white (255)
        # Pixels above threshold -> 255 (white/background)
        # Pixels below threshold -> 0 (black/letter)
        img_array = np.where(img_array > threshold, 255, 0)

        # Step 7: Normalize to [0, 1] range
        # Neural networks work best with normalized inputs
        # 0-255 range -> 0.0-1.0 range
        img_array = img_array / 255.0

        # Step 8: Convert to PyTorch tensor
        # Add channel dimension: (28, 28) -> (1, 28, 28)
        # 1 = single channel (grayscale)
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0)

        # Return image and label
        # Model will receive: image=(1, 28, 28), label=integer
        return img_tensor, label


def finetune_model(model, dataloader, num_epochs=3, learning_rate=0.0001):
    """
    Fine-tune the model on your handwriting.

    Strategy: Transfer Learning
    - Keep early layers (feature detection)
    - Train last layers (decision-making)

    Args:
        model: Pre-trained EMNIST_CNN model
        dataloader: DataLoader with your images
        num_epochs: How many times to see all data (default 3)
        learning_rate: Step size for updates (default 0.0001, 10x smaller than normal)
    """

    # ========================================
    # STEP 1: FREEZE CONVOLUTIONAL LAYERS
    # ========================================
    # These layers detect low-level features (edges, curves, corners)
    # They work for ANY handwriting, so we keep them unchanged

    print("\n" + "="*60)
    print("FREEZING LAYERS")
    print("="*60)

    # Freeze conv1 (first convolutional layer)
    # requires_grad=False means: don't calculate gradients, don't update weights
    for param in model.conv1.parameters():
        param.requires_grad = False
    print("✓ Frozen conv1 (32 filters, detects basic edges/shapes)")

    # Freeze conv2 (second convolutional layer)
    # This layer combines conv1 features into more complex patterns
    for param in model.conv2.parameters():
        param.requires_grad = False
    print("✓ Frozen conv2 (64 filters, detects letter parts)")

    # Note: pool and relu have no parameters, so nothing to freeze

    # ========================================
    # STEP 2: UNFREEZE FULLY CONNECTED LAYERS
    # ========================================
    # These layers make the final decision: "these features = letter C"
    # We train these to recognize YOUR specific handwriting patterns

    # Unfreeze fc1 (first fully connected layer)
    # This layer learns which feature combinations matter
    for param in model.fc1.parameters():
        param.requires_grad = True
    print("✓ Training fc1 (combines features)")

    # Unfreeze fc2 (output layer)
    # This layer maps features to 62 classes (0-9, A-Z, a-z)
    for param in model.fc2.parameters():
        param.requires_grad = True
    print("✓ Training fc2 (final classification)")

    print("\nResult: Keep 85% of model, train 15%")
    print("="*60)

    # ========================================
    # STEP 3: SETUP TRAINING
    # ========================================

    # Loss function: CrossEntropyLoss
    # Measures "how wrong" predictions are
    # Combines softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam with LOW learning rate
    # learning_rate=0.0001 (vs 0.001 for training from scratch)
    # filter(lambda p: p.requires_grad, ...) = only optimize unfrozen layers
    # Why low LR? Big steps might break what the model already learned
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Set model to training mode
    # Enables dropout (if any) and batch norm training behavior
    model.train()

    # ========================================
    # STEP 4: TRAINING LOOP
    # ========================================

    print("\nSTARTING FINE-TUNING")
    print("="*60)

    for epoch in range(num_epochs):
        # Track metrics for this epoch
        total_loss = 0      # Accumulated loss
        correct = 0         # Number of correct predictions
        total = 0           # Total samples seen

        # Iterate through batches
        # dataloader automatically shuffles and batches your data
        for images, labels in dataloader:
            # images shape: (batch_size, 1, 28, 28)
            # labels shape: (batch_size,)

            # ---- Standard Training Step ----

            # 1. Zero out gradients from previous batch
            # Why? Gradients accumulate, so we clear them each iteration
            optimizer.zero_grad()

            # 2. Forward pass: run images through model
            # outputs shape: (batch_size, 62) - scores for each class
            outputs = model(images)

            # 3. Calculate loss: how wrong are the predictions?
            # Compares model outputs to true labels
            loss = criterion(outputs, labels)

            # 4. Backward pass: calculate gradients
            # Computes how to adjust weights to reduce loss
            # Only calculates gradients for requires_grad=True parameters
            loss.backward()

            # 5. Update weights: take a step in direction of gradient
            # Only updates fc1 and fc2 (frozen layers ignored)
            optimizer.step()

            # ---- Track Statistics ----

            # Accumulate loss for reporting
            total_loss += loss.item()

            # Get predictions: highest scoring class
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # ---- Epoch Summary ----

        # Average loss across all batches
        avg_loss = total_loss / len(dataloader)

        # Training accuracy: % of samples predicted correctly
        accuracy = 100 * correct / total

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Training Accuracy: {accuracy:.2f}%")

        # What to expect:
        # - Loss should decrease each epoch (model getting better)
        # - Accuracy should increase (possibly reach 100% on training data)
        # - If loss increases: learning rate might be too high
        # - If accuracy stays low: might need more epochs or samples

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)


def test_finetuned_model(model, folder='my_letters'):
    """
    Test the fine-tuned model on all 26 uppercase letters.

    Tests on ONE sample per letter (to match original test methodology).

    Args:
        model: Fine-tuned EMNIST_CNN model
        folder: Folder containing test images

    Returns:
        accuracy: Percentage correct (0-100)
    """
    # Import preprocessing function from test script
    # This ensures we use IDENTICAL preprocessing
    from test_my_letters import preprocess_image

    # Set model to evaluation mode
    # Disables dropout, uses batch norm eval statistics
    model.eval()

    results = []
    tested_letters = set()  # Track which letters we've tested

    print("\nTESTING ON 26 UPPERCASE LETTERS")
    print("="*60)

    # Scan folder for images
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue

        # Parse filename
        basename = filename.replace('.png', '').replace('.jpg', '')
        letter = basename[0].upper()

        # Only test each letter ONCE (use first file found)
        # This matches your original testing: A.png used, A2.png and A3.png ignored
        if letter in tested_letters:
            continue

        if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            tested_letters.add(letter)
            image_path = os.path.join(folder, filename)

            # Preprocess using same function as test_my_letters.py
            img = preprocess_image(image_path)

            # No gradient calculation needed for testing
            # Saves memory and computation
            with torch.no_grad():
                # Forward pass
                output = model(img)

                # Convert raw scores to probabilities
                probs = torch.softmax(output, 1)

                # Get highest probability prediction
                confidence, predicted = torch.max(probs, 1)

            # Convert numeric label to character
            predicted_char = label_to_char(predicted.item())

            # Check if correct
            is_correct = (predicted_char == letter)

            # Store results
            results.append({
                'true': letter,
                'predicted': predicted_char,
                'confidence': confidence.item(),
                'correct': is_correct
            })

            # Print result
            status = "✓" if is_correct else "✗"
            print(f"{status} {letter} → {predicted_char} ({confidence.item()*100:.1f}% confident)")

    # Calculate overall accuracy
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}")

    # Show which letters are still wrong
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print("\nStill incorrect:")
        for r in incorrect:
            print(f"  {r['true']} → {r['predicted']} ({r['confidence']*100:.1f}%)")
    else:
        print("\n🎉 PERFECT! All letters correct!")

    return accuracy


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("FINE-TUNING EMNIST MODEL ON YOUR HANDWRITING")
    print("="*60)
    print("\nThis script will:")
    print("1. Load your 78 handwriting samples (26 letters × 3 samples)")
    print("2. Load pre-trained EMNIST model")
    print("3. Freeze early layers (keep feature detection)")
    print("4. Train last layers (adapt to your style)")
    print("5. Save fine-tuned model")
    print("6. Test accuracy on your letters")
    print()

    # ========================================
    # STEP 1: LOAD YOUR HANDWRITING DATASET
    # ========================================
    print("STEP 1: Loading your handwriting samples...")
    print("="*60)

    # Create dataset from your images
    dataset = MyHandwritingDataset(folder='finetune')

    # Sanity check: make sure we have enough data
    if len(dataset) < 26:
        print(f"\n⚠️  WARNING: Only found {len(dataset)} samples.")
        print("For best results, you should have at least 26 (one per letter)")
        print("Ideally 2-3 samples per letter.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()

    # Create DataLoader
    # - Batches your data (processes 8 images at once)
    # - Shuffles data each epoch (prevents learning order patterns)
    # batch_size=8: Small batches because we have small dataset
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"✓ Created DataLoader with batch_size=8")
    print(f"✓ Total batches per epoch: {len(dataloader)}")

    # ========================================
    # STEP 2: LOAD PRE-TRAINED EMNIST MODEL
    # ========================================
    print("\nSTEP 2: Loading pre-trained EMNIST model...")
    print("="*60)

    # Create model architecture
    model = EMNIST_CNN()

    # Load weights from training on 62-class EMNIST
    # This model already achieved 86% on EMNIST test set
    model.load_state_dict(torch.load('emnist_cnn.pth'))
    print("✓ Loaded emnist_cnn.pth (86% EMNIST accuracy)")

    # ========================================
    # STEP 3: FINE-TUNE ON YOUR HANDWRITING
    # ========================================
    print("\nSTEP 3: Fine-tuning on your handwriting...")
    print("="*60)
    print("Parameters:")
    print("  - Epochs: 3 (few to avoid catastrophic forgetting)")
    print("  - Learning rate: 0.0001 (10x smaller than normal training)")
    print("  - Trainable layers: fc1, fc2 only")
    print("  - Frozen layers: conv1, conv2")
    print("\nThis will take 10-30 seconds...")

    # Run fine-tuning
    # - 3 epochs: enough to adapt, not too much to forget EMNIST
    # - lr=0.0001: gentle adjustments to existing knowledge
    finetune_model(model, dataloader, num_epochs=3, learning_rate=0.0001)

    # ========================================
    # STEP 4: SAVE FINE-TUNED MODEL
    # ========================================
    print("\nSTEP 4: Saving fine-tuned model...")
    print("="*60)

    # Save the fine-tuned weights to a new file
    # Keep original emnist_cnn.pth unchanged (in case you want to re-train)
    torch.save(model.state_dict(), 'emnist_cnn_finetuned.pth')
    print("✓ Saved as: emnist_cnn_finetuned.pth")

    # ========================================
    # STEP 5: TEST ACCURACY
    # ========================================
    print("\nSTEP 5: Testing fine-tuned model...")
    print("="*60)

    accuracy = test_finetuned_model(model, folder='my_letters')

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Before fine-tuning: 11/26 (42.3%)")
    print(f"After fine-tuning:  {accuracy:.1f}%")
    improvement = accuracy - 42.3
    print(f"Improvement: {improvement:+.1f}%")
    print("\n" + "="*60)
    print("TO USE THE FINE-TUNED MODEL:")
    print("="*60)
    print("In test_my_letters.py, change line ~121:")
    print("  FROM: model.load_state_dict(torch.load('emnist_cnn.pth'))")
    print("  TO:   model.load_state_dict(torch.load('emnist_cnn_finetuned.pth'))")
    print("\nThen run: python test_my_letters.py")
    print("="*60)
