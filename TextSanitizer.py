def sanitizer_string(text: str | list[str], stop_words: list[str], only_alpha: bool, split_regex: str) -> str:
    word_list = text if isinstance(text, list) else text.split(split_regex)
    sanitized_text = ""
    if only_alpha:
        for word in word_list:
            if word not in stop_words and word.isalpha():
                sanitized_text += word + " "
    else:
        for word in word_list:
            if word not in stop_words:
                sanitized_text += word + " "
    return sanitized_text


def sanitizer_string_list(text: str | list[str], stop_words: list[str], only_alpha: bool, split_regex: str) -> list[str]:
    word_list = text if isinstance(text, list) else text.split(split_regex)
    sanitized_text = ""
    if only_alpha:
        for word in word_list:
            if word not in stop_words and word.isalpha():
                sanitized_text += word + " "
    else:
        for word in word_list:
            if word not in stop_words:
                sanitized_text += word + " "
    return sanitized_text.split(split_regex)