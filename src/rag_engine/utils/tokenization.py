from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def get_encoding(encoding_name: str):
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str | None = None, encoding_name: str = "cl100k_base") -> int:
    if model:
        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            encoder = get_encoding(encoding_name)
    else:
        encoder = get_encoding(encoding_name)
    return len(encoder.encode(text))
