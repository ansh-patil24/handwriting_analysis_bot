"""
CRNN (Convolutional Recurrent Neural Network) for handwriting recognition.

Architecture Overview:
    Line image (variable width × 32 height)
        ↓
    CNN layers (extract visual features at each horizontal position)
        ↓
    Sequence of feature vectors (one per position along the line)
        ↓
    Bidirectional LSTM (read sequence with context from both directions)
        ↓
    Linear layer (map features to character probabilities)
        ↓
    CTC loss (align predictions to ground truth text)

Key insight: The CNN "scans" horizontally across the line, the LSTM reads
the sequence of features, and CTC figures out which features = which characters.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN model for line-level text recognition.

    Input: (batch_size, 1, 32, width) - grayscale line images, 32px tall, variable width
    Output: (time_steps, batch_size, num_classes) - character probabilities at each position

    The model has three main parts:
    1. CNN: Extracts features from the image (similar to your EMNIST CNN)
    2. RNN: Reads the sequence of features with context (NEW!)
    3. FC: Maps to character probabilities
    """

    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=2):
        """
        Initialize CRNN architecture.

        Args:
            num_classes: Number of character classes (including CTC blank)
                         For our character set: ~80 classes
            hidden_size: LSTM hidden state size (default 256)
                         Larger = more memory/capacity, but slower
            num_lstm_layers: Number of stacked LSTM layers (default 2)
                             More layers = deeper network, can learn more complex patterns
        """
        super(CRNN, self).__init__()

        # ============================================================
        # Part 1: CNN - Extract visual features
        # ============================================================
        #
        # Goal: Convert image into sequence of feature vectors
        # Strategy: Apply convolutions that preserve width but reduce height
        #
        # Why preserve width? Width represents horizontal position (left-to-right)
        # Each position along the width will become one "time step" for the LSTM
        #
        # Input:  (batch, 1, 32, width)
        # Output: (batch, 512, 1, width/4)  - height collapsed to 1, width reduced by 4

        self.cnn = nn.Sequential(
            # Layer 1: 1 → 64 channels
            # Input: (batch, 1, 32, width)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # Why padding=1? Keeps width same: width_out = width_in with padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # MaxPool reduces by 2: (batch, 64, 16, width/2)

            # Layer 2: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After pooling: (batch, 128, 8, width/4)

            # Layer 3: 128 → 256 channels
            # BatchNorm added here - helps with training stability
            # Why BatchNorm? Normalizes activations, prevents vanishing/exploding gradients
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # No pooling yet: (batch, 256, 8, width/4)

            # Layer 4: 256 → 256 channels
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Pool only height, NOT width
            # Why (2,1)? Reduce height by 2, keep width same
            # After pooling: (batch, 256, 4, width/4)

            # Layer 5: 256 → 512 channels (final CNN layer)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Again, pool only height
            # Final output: (batch, 512, 2, width/4)
        )

        # After CNN, we have:
        # - 512 feature channels (rich representation)
        # - Height = 2 pixels (collapsed from 32)
        # - Width = width/4 (will become time_steps for LSTM)

        # ============================================================
        # Part 2: RNN - Read sequence with context
        # ============================================================
        #
        # Goal: Read the feature sequence and understand character patterns
        #
        # Why LSTM? Unlike CNN which looks at local patches, LSTM sees the whole line
        # It learns patterns like:
        # - "Q" is almost always followed by "u"
        # - "ca" is likely "cat" or "can", not "caz"
        # - Context helps resolve ambiguous characters (I vs l, O vs 0)
        #
        # Why Bidirectional? Reading both directions gives more context
        # Forward pass: knows what came BEFORE each character
        # Backward pass: knows what comes AFTER each character
        # Combined: knows full context

        self.rnn = nn.LSTM(
            input_size=512 * 2,  # CNN output has 512 channels × 2 height = 1024 features
            hidden_size=hidden_size,  # Default 256 - LSTM hidden state size
            num_layers=num_lstm_layers,  # Default 2 - stack 2 LSTM layers
            bidirectional=True,  # Read sequence both ways (forward + backward)
            batch_first=False  # LSTM expects (seq_len, batch, features)
        )

        # Bidirectional doubles the output size:
        # Forward LSTM: hidden_size (256)
        # Backward LSTM: hidden_size (256)
        # Combined: hidden_size * 2 (512)

        # ============================================================
        # Part 3: Fully Connected - Map to character probabilities
        # ============================================================
        #
        # Goal: Convert LSTM output to character predictions
        # For each time step, predict probability of each character

        self.fc = nn.Linear(
            hidden_size * 2,  # 256 * 2 = 512 (bidirectional LSTM output)
            num_classes  # ~80 classes (all characters + blank)
        )

        # No softmax here! Why?
        # CTCLoss expects raw logits (unnormalized scores), not probabilities
        # It applies log_softmax internally

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, 1, 32, width)
               Batch of grayscale line images, 32 pixels tall, variable width

        Returns:
            Output tensor (time_steps, batch_size, num_classes)
            Character probabilities at each time step
            Shape is transposed for CTCLoss which expects (T, N, C)
        """
        # Step 1: CNN feature extraction
        # Input: (batch, 1, 32, width)
        conv_output = self.cnn(x)
        # Output: (batch, 512, 2, width/4)

        # Step 2: Reshape for RNN
        # LSTM expects (sequence_length, batch, features)
        # We need to convert CNN's 4D output to RNN's 3D input
        #
        # Strategy:
        # - Treat width/4 as sequence_length (time steps)
        # - Flatten channels and height into features (512*2 = 1024)

        batch_size, channels, height, width = conv_output.size()
        # Example: batch=16, channels=512, height=2, width=100

        # Permute to move width to front: (batch, width, channels, height)
        conv_output = conv_output.permute(0, 3, 1, 2)
        # Now: (16, 100, 512, 2)

        # Reshape to flatten channels and height into one feature dimension
        conv_output = conv_output.reshape(batch_size, width, channels * height)
        # Now: (16, 100, 1024) = (batch, time_steps, features)

        # Transpose to LSTM's expected format: (time_steps, batch, features)
        rnn_input = conv_output.permute(1, 0, 2)
        # Now: (100, 16, 1024) = (seq_len, batch, features)

        # Step 3: LSTM sequence processing
        # Input: (seq_len, batch, features) = (100, 16, 1024)
        rnn_output, _ = self.rnn(rnn_input)
        # Output: (seq_len, batch, hidden*2) = (100, 16, 512)
        # We ignore the hidden states (the _ part) - only need output

        # Step 4: Fully connected layer
        # Apply FC to each time step independently
        # rnn_output shape: (seq_len, batch, 512)
        # We need to apply FC to the last dimension

        # Reshape to apply FC: (seq_len * batch, 512)
        seq_len, batch, hidden = rnn_output.size()
        rnn_output = rnn_output.view(seq_len * batch, hidden)

        # Apply FC: (seq_len * batch, 512) → (seq_len * batch, num_classes)
        output = self.fc(rnn_output)

        # Reshape back: (seq_len, batch, num_classes)
        output = output.view(seq_len, batch, -1)

        # Final output shape: (time_steps, batch, num_classes)
        # This is what CTCLoss expects!
        return output


if __name__ == "__main__":
    # Test the model to verify shapes work correctly
    from crnn_utils import NUM_CLASSES

    print(f"Testing CRNN architecture")
    print(f"Number of classes: {NUM_CLASSES}\n")

    # Create model
    model = CRNN(num_classes=NUM_CLASSES, hidden_size=256, num_lstm_layers=2)
    print(f"Model created successfully\n")

    # Print model summary
    print("Model architecture:")
    print(model)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Test forward pass with dummy data
    print("Testing forward pass...")

    # Create dummy input: batch of 4 images, 32px tall, 100px wide
    batch_size = 4
    height = 32
    width = 100
    dummy_input = torch.randn(batch_size, 1, height, width)
    print(f"Input shape: {dummy_input.shape} (batch, channels, height, width)")

    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Don't compute gradients (faster)
        output = model(dummy_input)

    print(f"Output shape: {output.shape} (time_steps, batch, num_classes)")
    print(f"  Time steps: {output.shape[0]} (width/4 = {width}/4 = {width//4})")
    print(f"  Batch size: {output.shape[1]}")
    print(f"  Num classes: {output.shape[2]}")
    print("\n✓ Forward pass successful!")

    # Test with different widths (variable length lines)
    print("\nTesting variable width inputs:")
    for test_width in [50, 150, 300]:
        test_input = torch.randn(2, 1, 32, test_width)
        output = model(test_input)
        print(f"  Width {test_width:3d} → time steps {output.shape[0]:3d}")

    print("\n✓ All tests passed!")
