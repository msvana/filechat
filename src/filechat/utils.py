def truncate_text(text: str, max_length: int = 100) -> str:
    words = []
    length_without_spaces = 0
    all_words = text.split(" ")

    for word in all_words:
        if length_without_spaces + len(words) - 1 + len(word) > max_length:
            break
        words.append(word)
        length_without_spaces += len(word)

    truncated = " ".join(words)
    if len(truncated) < len(text):
        truncated += "..."
    return truncated
