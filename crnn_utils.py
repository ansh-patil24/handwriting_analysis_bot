"""
Character encoding utilities for CRNN training.

The CRNN model needs to map between:
- Characters (what humans read): 'A', 'b', '3', '.'
- Integers (what neural networks process): 0, 1, 2, 3...

This file provides:
- Character set definition (all possible characters)
- Encoding: text string → list of integers
- Decoding: list of integers → text string
"""

# Define the complete character set
# This is every character that can appear in the IAM dataset
# Order matters - each position becomes that character's index

# Build character set from components
SPACE = ' '
UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LOWERCASE = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
BASIC_PUNCT = '.,!?\'"$%&()-/:;@'
EXTRA_SYMBOLS = '*+<=>[]_{|}~'
# Special Unicode characters found in IAM dataset
SPECIAL_CHARS = (
    '©'  # © Copyright
    '«'  # « Left guillemet
    '»'  # » Right guillemet
    '®'  # ® Registered trademark
    '¢'  # ¢ Cent
    '°'  # ° Degree
    '—'  # — Em-dash
    '‘'  # ' Left single quote
    '’'  # ' Right single quote
    '“'  # " Left double quote
    '”'  # " Right double quote
    '™'  # ™ Trademark
    'é'  # é e-acute
)

# Combine all characters
CHARACTERS = (
    SPACE +
    UPPERCASE +
    LOWERCASE +
    DIGITS +
    BASIC_PUNCT +
    EXTRA_SYMBOLS +
    SPECIAL_CHARS
)

# Total character count (including blank token for CTC at index 0)
# CTC needs a "blank" symbol to represent gaps between characters
# PyTorch CTCLoss automatically treats index 0 as blank
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank at index 0

# Create lookup dictionaries for fast encoding/decoding
# char_to_idx: 'A' → 1, 'B' → 2, etc.
# idx_to_char: 1 → 'A', 2 → 'B', etc.

# Why enumerate(CHARACTERS, start=1)?
# start=1 because index 0 is reserved for CTC blank
# Example: enumerate('ABC', start=1) gives [(1,'A'), (2,'B'), (3,'C')]
char_to_idx = {char: idx for idx, char in enumerate(CHARACTERS, start=1)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTERS, start=1)}

# Add blank token explicitly (even though CTC handles it)
# This makes debugging easier - you can see blank symbols in output
idx_to_char[0] = '<blank>'


def encode_text(text):
    """
    Convert text string to list of integer indices for training.

    This is used during training to convert ground truth labels into numbers
    that CTC loss can work with.

    Example:
        encode_text("Hi") → [34, 44]  (indices for 'H' and 'i')
        encode_text("A1") → [1, 53]   (indices for 'A' and '1')

    Args:
        text: String to encode (e.g., "Hello world")

    Returns:
        List of integers (e.g., [34, 40, 43, 43, 50, 0, 58, 50, 53, 43, 39])

    Handles unknown characters:
        If a character isn't in CHARACTERS, it's skipped
        This prevents crashes on unexpected symbols
    """
    # List comprehension: for each char in text, look up its index
    # Only include chars that exist in char_to_idx (skip unknowns)
    encoded = [char_to_idx[char] for char in text if char in char_to_idx]
    return encoded


def decode_prediction(indices):
    """
    Convert list of integer predictions back to readable text.

    This is used after inference to convert model output into human-readable text.

    CTC output contains:
    - Repeated characters: [1, 1, 1, 2] → "AB" (collapse repeats)
    - Blank tokens (0): [1, 0, 2] → "AB" (remove blanks)

    This function handles the CTC decoding:
    1. Remove blanks (index 0)
    2. Collapse repeated characters

    Example CTC sequence:
        [1, 1, 0, 2, 2, 2, 0, 1]  # Raw CTC output
            ↓
        [1, 2, 1]                  # After removing blanks and collapsing
            ↓
        "ABA"                      # Final decoded text

    Args:
        indices: List of integers from model output

    Returns:
        Decoded text string

    Why collapse repeats?
        CTC allows the model to output the same character multiple times
        Example: 'L' might span 5 time steps: [L, L, L, L, L]
        We want just one 'L' in the final output
    """
    # Step 1: Remove consecutive duplicates
    # Example: [1, 1, 2, 2, 2, 3] → [1, 2, 3]
    #
    # Why? CTC outputs can have repeats when a character spans multiple positions
    # zip(indices, indices[1:]) pairs each element with the next one
    # Example: [1,1,2,2,3] → [(1,1), (1,2), (2,2), (2,3)]
    # Keep i only when i != i_next (i.e., when character changes)

    collapsed = []
    prev = None
    for idx in indices:
        if idx != prev:  # Character changed (or first character)
            collapsed.append(idx)
            prev = idx

    # Step 2: Remove blank tokens (index 0)
    # CTC uses blanks to separate repeated characters
    # Example: [1, 0, 1] → "AA" (not "A"), [1, 1] → "A" (collapsed)
    no_blanks = [idx for idx in collapsed if idx != 0]

    # Step 3: Convert indices back to characters
    # Look up each index in idx_to_char dictionary
    # Skip any indices not in dictionary (shouldn't happen, but defensive)
    chars = [idx_to_char[idx] for idx in no_blanks if idx in idx_to_char]

    # Step 4: Join into final string
    decoded_text = ''.join(chars)

    return decoded_text


def print_character_set():
    """
    Utility function to display the character set and indices.
    Useful for debugging - shows what index maps to what character.
    """
    print(f"Total classes (including blank): {NUM_CLASSES}")
    print(f"Total characters (excluding blank): {len(CHARACTERS)}")
    print("\nCharacter mapping:")
    print("0: <blank> (CTC token)")
    for char, idx in sorted(char_to_idx.items(), key=lambda x: x[1]):
        # Print with repr() to make spaces visible as ' ' not invisible
        print(f"{idx}: {repr(char)}")


if __name__ == "__main__":
    # Test the encoding/decoding functions
    print("Testing character encoding/decoding\n")

    # Test 1: Simple text
    test_text = "Hello World"
    encoded = encode_text(test_text)
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")

    # Decode it back
    decoded = decode_prediction(encoded)
    print(f"Decoded: '{decoded}'")
    print(f"Match: {test_text == decoded}\n")

    # Test 2: CTC sequence with blanks and repeats
    # Simulate CTC output: "HI" with blanks and repeats
    # H(34) appears 3 times, blank(0), I(44) appears 2 times
    ctc_output = [34, 34, 34, 0, 44, 44]
    decoded_ctc = decode_prediction(ctc_output)
    print(f"CTC output: {ctc_output}")
    print(f"Decoded: '{decoded_ctc}' (should be 'Hi')\n")

    # Test 3: Mixed case, digits, punctuation
    complex_text = "Test 123, done!"
    encoded_complex = encode_text(complex_text)
    decoded_complex = decode_prediction(encoded_complex)
    print(f"Complex: '{complex_text}'")
    print(f"Encoded: {encoded_complex}")
    print(f"Decoded: '{decoded_complex}'")
    print(f"Match: {complex_text == decoded_complex}\n")

    # Show full character set
    print("\n" + "="*50)
    print_character_set()
