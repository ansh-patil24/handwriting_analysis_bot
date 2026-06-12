def label_to_char(label):
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(65 + label - 10)
    else:
        return chr(97 + label - 36)

def char_to_label(char):
    if char.isdigit():
        return int(char)
    elif char.isupper():
        return ord(char) - 65 + 10
    elif char.islower():
        return ord(char) - 97 + 36
    else:
        raise ValueError(f"Character '{char}' not in EMNIST byclass")