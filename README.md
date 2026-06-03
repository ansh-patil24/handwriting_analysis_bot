# Handwriting Recognition OCR System

Deep learning-based OCR system for recognizing handwritten text from full page images.

## Current Status

**Phase 1: Character Recognition (Complete)**
- ✅ Trained CNN on EMNIST 'byclass' (62 classes: digits, uppercase, lowercase)
- ✅ Fine-tuned on personal handwriting using selective OHEM
- ✅ Achieved 24/26 (92.3%) accuracy on uppercase letters
- ✅ Model handles O→D and V→Y errors (fixable with spell checking)

**Phase 2: Full OCR Pipeline (In Progress)**
- ⏳ CNN-RNN-CTC model for line recognition
- ⏳ Line segmentation from full pages
- ⏳ Training on IAM Handwriting Dataset
- ⏳ Fine-tuning on personal handwriting

## Architecture

### Current: Isolated Character Recognition
```
Single character image (28×28) → CNN → Character prediction
```

**Model:** `emnist_cnn_focal.pth`
- CNN with 2 conv layers + 2 FC layers
- Trained with focal loss for hard example mining
- 62-class output (0-9, A-Z, a-z)

### Target: Full Page OCR
```
Full page image
    ↓
Line Segmentation (computer vision)
    ↓
Line images
    ↓
CNN-RNN-CTC (sequence recognition)
    ↓
Text output per line
    ↓
Post-processing (spell check)
    ↓
Final text
```

## Files

**Models:**
- `emnist_cnn_focal.pth` - Best character recognition model (92.3% accuracy)
- `emnist_cnn_model.py` - CNN architecture definition
- `emnist_utils.py` - Character-to-label encoding utilities

**Data:**
- `my_letters/` - Test set (26 uppercase letters)
- `Set_2/`, `Set_3/`, `Set_4/`, `Set_5/` - Training sets (26 letters each)

**Training Scripts:** (archived - to be replaced with CRNN pipeline)

## Training Details

**Dataset:**
- Base: EMNIST 'byclass' split (600k+ samples)
- Personalization: 4 sets of handwritten alphabet (104 samples)
- Held-out test: 26 uppercase letters in `my_letters/`

**Training Approach:**
1. Trained on full EMNIST dataset (86% accuracy)
2. Combined training: EMNIST + personal samples each epoch
3. Selective OHEM: Only trains on incorrect predictions
4. Base weight: 100x amplification on personal samples

**Results:**
- Original EMNIST model: 11/26 (42%) on personal handwriting
- After combined training: 24/26 (92%)
- Remaining errors: O→D, V→Y (both fixable with spell checking)

## Next Steps

1. **Download IAM Dataset** (~4GB)
   - 115k text line images with transcriptions
   - Real-world cursive + print handwriting

2. **Build CNN-RNN-CTC Model**
   - CNN for feature extraction
   - Bidirectional LSTM for sequence modeling
   - CTC loss for alignment-free training

3. **Implement Line Segmentation**
   - Horizontal projection for line detection
   - Handle skew and varying spacing

4. **End-to-End Pipeline**
   - Page → Lines → Text
   - Post-processing with spell checking

## Requirements

```
torch
torchvision
pillow
numpy
scikit-image
matplotlib
```

## Timeline

- Week 1: Download IAM, implement CRNN
- Week 2: Train CRNN on IAM dataset
- Week 3: Fine-tune on personal handwriting
- Week 4: Line segmentation
- Week 5: Integration and testing

## Learning Journey

This project started as learning isolated character recognition (MNIST → EMNIST → personal handwriting) and is evolving into a full OCR system capable of reading entire pages of handwritten text.

Key milestones:
1. ✅ Understood neural networks (FC → CNN)
2. ✅ Trained on MNIST digits (99% accuracy)
3. ✅ Expanded to EMNIST letters (86% accuracy)
4. ✅ Personalized to own handwriting (92% accuracy)
5. ⏳ Building sequence recognition (CRNN)
6. ⏳ Full page OCR pipeline

---

**Last Updated:** June 3, 2026
