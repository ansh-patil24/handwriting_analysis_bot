"""
Training script for CRNN model on IAM word-level dataset.

Uses the official IAM words dataset with proper ground truth labels.
Should achieve much better results than the line-based approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from crnn_model import CRNN
from crnn_utils import NUM_CLASSES, decode_prediction
from iam_words_dataset import get_data_loaders


def calculate_cer(predictions, targets):
    """Calculate Character Error Rate."""
    total_errors = 0
    total_chars = 0

    for pred, target in zip(predictions, targets):
        errors = levenshtein_distance(pred, target)
        total_errors += errors
        total_chars += len(target)

    if total_chars == 0:
        return 0.0

    cer = total_errors / total_chars
    return cer


def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings."""
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


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()

    total_loss = 0
    num_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (images, texts, text_lengths, input_lengths) in enumerate(train_loader):
        if images is None:
            continue

        images = images.to(device)
        texts = texts.to(device)
        text_lengths = text_lengths.to(device)
        input_lengths = input_lengths.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate CTC loss
        loss = criterion(outputs, texts, input_lengths, text_lengths)

        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, dataset):
    """
    Evaluate the model on validation set.

    Args:
        model: CRNN model
        val_loader: Validation DataLoader
        criterion: CTCLoss
        device: cuda or cpu
        dataset: The dataset object (to access original texts)
    """
    model.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, texts, text_lengths, input_lengths in val_loader:
            if images is None:
                continue

            images = images.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)
            input_lengths = input_lengths.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, texts, input_lengths, text_lengths)
            total_loss += loss.item()

            # Decode predictions for CER calculation
            batch_size = outputs.shape[1]

            # Decode each sample in batch
            for i in range(batch_size):
                output = outputs[:, i, :]
                _, pred_indices = output.max(dim=1)
                pred_indices = pred_indices.cpu().numpy()
                pred_text = decode_prediction(pred_indices)
                all_predictions.append(pred_text)

            # Decode targets from concatenated texts
            text_start = 0
            for length in text_lengths:
                target_indices = texts[text_start:text_start + length].cpu().numpy()
                target_text = decode_prediction(target_indices)
                all_targets.append(target_text)
                text_start += length

    avg_loss = total_loss / len(val_loader)

    # Calculate CER
    cer = calculate_cer(all_predictions, all_targets)

    return avg_loss, cer


def train_crnn(
    words_file='data/IAM_words/words_new.txt',
    images_dir='data/IAM_words/iam_words/words',
    num_epochs=30,
    batch_size=64,
    learning_rate=0.0003,
    device=None,
    save_dir='checkpoints_words'
):
    """
    Main training function for word-level IAM dataset.

    Args:
        words_file: Path to words_new.txt
        images_dir: Path to words directory
        num_epochs: Number of training epochs (default 30)
        batch_size: Batch size (default 64 - words are smaller than lines)
        learning_rate: Learning rate for Adam optimizer (default 0.0003)
        device: 'cuda' or 'cpu' (auto-detected if None)
        save_dir: Directory to save model checkpoints
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("\nLoading IAM words dataset...")
    train_loader, val_loader = get_data_loaders(
        words_file,
        images_dir,
        batch_size=batch_size,
        train_split=0.9,
        num_workers=0
    )

    # Get the underlying dataset for validation CER calculation
    val_dataset = val_loader.dataset.dataset  # Access through Subset wrapper

    # Create model
    print("\nCreating model...")
    model = CRNN(
        num_classes=NUM_CLASSES,
        hidden_size=256,
        num_lstm_layers=2
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # Training loop
    print("\nStarting training...")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*70)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_cer = validate(model, val_loader, criterion, device, val_dataset)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val CER:    {val_cer:.2%}")
        print(f"  Learning Rate: {current_lr:.6f}")

        if new_lr != current_lr:
            print(f"  -> LR reduced to {new_lr:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'crnn_words_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_cer': val_cer,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'crnn_words_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_cer': val_cer,
            }, best_path)
            print(f"  ★ New best model saved: {best_path}")

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(save_dir, 'crnn_words_best.pth')}")


if __name__ == "__main__":
    # Train on IAM words dataset with proper labels
    train_crnn(
        words_file='data/IAM_words/words_new.txt',
        images_dir='data/IAM_words/iam_words/words',
        num_epochs=50,
        batch_size=64,  # Can use larger batch size since words are smaller
        learning_rate=0.0003,
        save_dir='checkpoints_words'
    )
