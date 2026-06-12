"""
PyTorch Dataset for IAM word-level images.

This Dataset handles the official IAM words dataset:
- Pre-segmented word images
- Official ground truth labels from words.txt
- Much better quality than automatic line segmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from crnn_utils import encode_text


class IAMWordsDataset(Dataset):
    """
    PyTorch Dataset for IAM word images with official labels.

    Usage:
        dataset = IAMWordsDataset('data/IAM_words/words_new.txt', 'data/IAM_words/iam_words/words')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    """

    def __init__(self, words_file, images_dir, img_height=32, img_width=None, filter_errors=True):
        """
        Initialize the dataset.

        Args:
            words_file: Path to words_new.txt (contains labels)
            images_dir: Path to words directory (contains a01/, a02/, etc.)
            img_height: Target height to resize images to (default 32 pixels)
            img_width: Target width (default None = variable width, keeps aspect ratio)
            filter_errors: If True, skip samples marked as 'er' (segmentation errors)
        """
        self.images_dir = images_dir
        self.img_height = img_height
        self.img_width = img_width
        self.samples = []

        # Parse words_new.txt
        # Format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
        #         word_id status ... tag transcription

        with open(words_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Split line into parts (limit to 9 splits to preserve spaces in transcription)
                parts = line.split(None, 8)  # Split on whitespace, max 9 parts
                if len(parts) < 9:
                    continue

                word_id = parts[0]  # e.g., a01-000u-00-00
                status = parts[1]   # ok or er
                # parts[2] = graylevel
                # parts[3-6] = bounding box (x, y, w, h)
                # parts[7] = grammatical tag
                # parts[8] = transcription (may contain spaces)
                transcription = parts[8]

                # Filter out error samples if requested
                if filter_errors and status == 'er':
                    continue

                # Skip empty transcriptions
                if not transcription:
                    continue

                # Build image path from word_id
                # a01-000u-00-00 -> a01/a01-000u/a01-000u-00-00.png
                form_id = word_id.split('-')[0]  # a01
                page_id = '-'.join(word_id.split('-')[:2])  # a01-000u
                image_path = os.path.join(images_dir, form_id, page_id, f"{word_id}.png")

                self.samples.append((image_path, transcription))

        print(f"Loaded {len(self.samples)} word samples from {words_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return one word sample.

        Returns:
            Tuple of (image, encoded_text, text_length):
            - image: Tensor (1, 32, width)
            - encoded_text: List of integers
            - text_length: Integer
        """
        image_path, text = self.samples[idx]

        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not load {image_path}, skipping")
            return None, None, None

        # Resize image (fixed height, variable width to preserve aspect ratio)
        original_h, original_w = img.shape

        if self.img_width is None:
            # Variable width: scale proportionally
            scale = self.img_height / original_h
            new_width = int(original_w * scale)

            # Clamp width (words are shorter than lines)
            # Min 8px, max 400px (most words fit in 200px)
            new_width = max(8, min(new_width, 400))
        else:
            new_width = self.img_width

        img = cv2.resize(img, (new_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (add channel dimension)
        img_tensor = torch.FloatTensor(img).unsqueeze(0)

        # Encode text
        encoded_text = encode_text(text)
        text_length = len(encoded_text)

        # Validate CTC requirements: input_length > text_length
        input_length = new_width // 4  # After CNN pooling

        if input_length <= text_length:
            # Pad image wider to satisfy CTC
            required_width = (text_length + 1) * 4
            padded_img = np.zeros((self.img_height, required_width), dtype=np.float32)
            padded_img[:, :new_width] = img
            img = padded_img
            new_width = required_width
            img_tensor = torch.FloatTensor(img).unsqueeze(0)

        return img_tensor, encoded_text, text_length


def collate_fn(batch):
    """
    Custom collate function for batching variable-width word images.

    Args:
        batch: List of (image, encoded_text, text_length) tuples

    Returns:
        Tuple of (images, texts, text_lengths, input_lengths)
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample[0] is not None]

    if len(batch) == 0:
        return None, None, None, None

    images, texts, text_lengths = zip(*batch)

    # Find max width in batch
    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    batch_size = len(images)
    img_height = images[0].shape[1]

    # Create padded tensor
    padded_images = torch.zeros(batch_size, 1, img_height, max_width)

    for i, img in enumerate(images):
        width = img.shape[2]
        padded_images[i, :, :, :width] = img

    # Calculate input lengths (time steps for CTC)
    input_lengths = torch.LongTensor([w // 4 for w in widths])

    # Concatenate all encoded texts
    concatenated_texts = []
    for text in texts:
        concatenated_texts.extend(text)

    texts_tensor = torch.LongTensor(concatenated_texts)
    text_lengths_tensor = torch.LongTensor(text_lengths)

    return padded_images, texts_tensor, text_lengths_tensor, input_lengths


def get_data_loaders(words_file, images_dir, batch_size=32, train_split=0.9, num_workers=0):
    """
    Create train and validation data loaders for IAM words dataset.

    Args:
        words_file: Path to words_new.txt
        images_dir: Path to words directory
        batch_size: Batch size (default 32)
        train_split: Train/val split (default 0.9)
        num_workers: Number of data loading workers (default 0)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = IAMWordsDataset(words_file, images_dir)

    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing IAM Words Dataset\n")

    words_file = 'data/IAM_words/words_new.txt'
    images_dir = 'data/IAM_words/iam_words/words'

    # Create dataset
    dataset = IAMWordsDataset(words_file, images_dir)
    print(f"Dataset size: {len(dataset)}\n")

    # Test loading one sample
    print("Loading sample 0...")
    img, encoded_text, text_length = dataset[0]
    if img is not None:
        print(f"  Image shape: {img.shape}")
        print(f"  Encoded text: {encoded_text}")
        print(f"  Text length: {text_length}\n")

    # Test data loader
    print("Testing DataLoader with batch_size=4...")
    train_loader, val_loader = get_data_loaders(
        words_file,
        images_dir,
        batch_size=4,
        train_split=0.9,
        num_workers=0
    )

    # Get one batch
    batch = next(iter(train_loader))
    if batch[0] is not None:
        images, texts, text_lengths, input_lengths = batch

        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Texts: {texts.shape}")
        print(f"  Text lengths: {text_lengths.tolist()}")
        print(f"  Input lengths: {input_lengths.tolist()}")

        print("\n✓ Dataset and DataLoader working correctly!")
