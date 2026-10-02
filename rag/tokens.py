from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / no cached BPE file
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)  # rough fallback: ~4 chars per token
    return len(enc.encode(text, disallowed_special=()))
