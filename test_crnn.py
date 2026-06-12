"""
Test script for trained CRNN model.

This script:
1. Loads the trained model (crnn_best.pth)
2. Tests on validation samples
3. Displays predictions vs ground truth
4. Calculates accuracy metrics

Usage:
    python test_crnn.py                    # Test on validation set
    python test_crnn.py --image line.png   # Test on single image
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from pathlib import Path

from crnn_model import CRNN
from crnn_utils import NUM_CLASSES, decode_prediction, encode_text
from iam_words_dataset import IAMWordsDataset, collate_fn
from torch.utils.data import DataLoader


def load_model(checkpoint_path='checkpoints_words/crnn_words_best.pth', device='cpu'):
    """
    Load trained CRNN model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: 'cuda' or 'cpu'

    Returns:
        Loaded CRNN model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")

    # Create model architecture
    model = CRNN(
        num_classes=NUM_CLASSES,
        hidden_size=256,
        num_lstm_layers=2
    )

    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # checkpoint is a dict with 'model_state_dict', 'epoch', etc.
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode (disables dropout, etc.)
    model.eval()
    model = model.to(device)

    # Print checkpoint info
    print(f"[OK] Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model


def preprocess_image(image_path, img_height=32):
    """
    Load and preprocess a line image for CRNN inference.

    Same preprocessing as training:
    1. Load as grayscale
    2. Resize to height 32, keep aspect ratio
    3. Normalize to [0, 1]
    4. Convert to tensor (1, 32, width)

    Args:
        image_path: Path to line image
        img_height: Target height (default 32)

    Returns:
        Preprocessed tensor (1, 1, 32, width)
    """
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize (keep aspect ratio)
    original_h, original_w = img.shape
    scale = img_height / original_h
    new_width = int(original_w * scale)
    new_width = max(8, min(new_width, 1000))  # Clamp to reasonable range

    img = cv2.resize(img, (new_width, img_height), interpolation=cv2.INTER_LINEAR)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Convert to tensor: (H, W) → (1, 1, H, W)
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)

    return img_tensor


def recognize_line(model, image_path, device='cpu'):
    """
    Recognize text in a single line image.

    Args:
        model: Trained CRNN model
        image_path: Path to line image
        device: 'cuda' or 'cpu'

    Returns:
        Predicted text string
    """
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        # output shape: (time_steps, 1, num_classes)

    # Decode prediction
    # Get most likely character at each time step
    _, pred_indices = output.squeeze(1).max(dim=1)  # (time_steps,)
    pred_indices = pred_indices.cpu().numpy()

    # CTC decode (remove blanks and collapse repeats)
    predicted_text = decode_prediction(pred_indices)

    return predicted_text


def calculate_accuracy(pred, target):
    """
    Calculate character-level accuracy.

    Simple metric: how many characters match?
    (Not the same as CER - this doesn't count insertions/deletions)

    Args:
        pred: Predicted string
        target: Ground truth string

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    # Pad shorter string with spaces for comparison
    max_len = max(len(pred), len(target))
    pred_padded = pred.ljust(max_len)
    target_padded = target.ljust(max_len)

    # Count matches
    matches = sum(p == t for p, t in zip(pred_padded, target_padded))
    accuracy = matches / max_len if max_len > 0 else 0.0

    return accuracy


def levenshtein_distance(s1, s2):
    """
    Calculate edit distance between two strings.

    Used for Character Error Rate (CER) calculation.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(pred, target):
    """
    Calculate Character Error Rate.

    CER = edit_distance / len(target)

    Lower is better. 0% = perfect recognition.
    """
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    distance = levenshtein_distance(pred, target)
    cer = distance / len(target)

    return cer


def test_on_validation_set(model, words_file='data/IAM_words/words_new.txt',
                           images_dir='data/IAM_words/iam_words/words',
                           num_samples=50, device='cpu'):
    """
    Test model on validation samples.

    Args:
        model: Trained CRNN model
        words_file: Path to words_new.txt
        images_dir: Path to words directory
        num_samples: How many samples to test (default 50)
        device: 'cuda' or 'cpu'
    """
    print("\n" + "="*70)
    print("Testing on validation set")
    print("="*70 + "\n")

    # Load validation dataset
    # Use same 90/10 split as training
    full_dataset = IAMWordsDataset(words_file, images_dir, img_height=32)

    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Same seed as training
    )

    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing on first {num_samples} samples\n")

    total_cer = 0.0
    total_acc = 0.0
    perfect_matches = 0

    for i in range(min(num_samples, len(val_dataset))):
        # Get sample
        img_tensor, encoded_text, text_length = val_dataset[i]

        # Get ground truth text
        # Need to get the original sample to find the text
        sample_idx = val_dataset.indices[i]
        image_path, target_text = full_dataset.samples[sample_idx]

        # Move to device and add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 1, 32, width)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            # output: (time_steps, 1, num_classes)

        # Decode prediction
        _, pred_indices = output.squeeze(1).max(dim=1)
        pred_indices = pred_indices.cpu().numpy()
        predicted_text = decode_prediction(pred_indices)

        # Calculate metrics
        cer = calculate_cer(predicted_text, target_text)
        acc = calculate_accuracy(predicted_text, target_text)

        total_cer += cer
        total_acc += acc

        if predicted_text == target_text:
            perfect_matches += 1

        # Print result
        print(f"Sample {i+1}/{num_samples}:")
        print(f"  Target: '{target_text}'")
        print(f"  Predicted: '{predicted_text}'")
        print(f"  CER: {cer*100:.1f}%  |  Accuracy: {acc*100:.1f}%")

        # Visual comparison for mismatches
        if predicted_text != target_text:
            print(f"  Errors: ", end="")
            for p, t in zip(predicted_text.ljust(len(target_text)), target_text):
                if p != t:
                    print(f"[{t}->{p}]", end=" ")
            print()
        print()

    # Summary statistics
    avg_cer = (total_cer / num_samples) * 100
    avg_acc = (total_acc / num_samples) * 100
    perfect_rate = (perfect_matches / num_samples) * 100

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Perfect matches: {perfect_matches}/{num_samples} ({perfect_rate:.1f}%)")
    print()

    # Interpretation guide
    if avg_cer < 5:
        print("[EXCELLENT] Model is production-ready.")
    elif avg_cer < 15:
        print("[GOOD] Model works well for most text.")
    elif avg_cer < 30:
        print("[FAIR] Model needs more training or better data.")
    else:
        print("[POOR] Model needs significant improvement.")


def test_single_image(model, image_path, device='cpu'):
    """
    Test model on a single line image.

    Args:
        model: Trained CRNN model
        image_path: Path to line image
        device: 'cuda' or 'cpu'
    """
    print("\n" + "="*70)
    print(f"Testing on: {image_path}")
    print("="*70 + "\n")

    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        return

    # Recognize text
    predicted_text = recognize_line(model, image_path, device)

    print(f"Predicted text: '{predicted_text}'")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test CRNN model')
    parser.add_argument('--image', type=str, help='Path to single line image to test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_words/crnn_words_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--words-file', type=str, default='data/IAM_words/words_new.txt',
                        help='Path to words_new.txt')
    parser.add_argument('--images-dir', type=str, default='data/IAM_words/iam_words/words',
                        help='Path to words directory')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of validation samples to test')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}\n")

    # Load model
    model = load_model(args.checkpoint, device)

    # Test mode: single image or validation set
    if args.image:
        test_single_image(model, args.image, device)
    else:
        test_on_validation_set(model, args.words_file, args.images_dir,
                              num_samples=args.samples, device=device)


if __name__ == "__main__":
    main()
