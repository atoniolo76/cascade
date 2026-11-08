from __future__ import annotations

import hashlib
import logging
import random
import re
import threading
from pathlib import Path

import httpx
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

log = logging.getLogger(__name__)
MODEL_NAME = re.compile(r"[\w.-]+(?:/[\w.-]+)?", re.ASCII)


class Tokenizers:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "tokenizers"
        self._cache: dict[str, tuple[Tokenizer, tuple[int, ...]] | None] = {}
        self._lock = threading.Lock()

    def get(self, model: str | None) -> tuple[Tokenizer, tuple[int, ...]] | None:
        if not model or not MODEL_NAME.fullmatch(model):
            return None
        with self._lock:
            if model not in self._cache:
                self._cache[model] = self._load(model)
            return self._cache[model]

    def _load(self, model: str) -> tuple[Tokenizer, tuple[int, ...]] | None:
        path = self.root / f"{hashlib.sha256(model.encode()).hexdigest()}.json"
        try:
            if path.is_file():
                tokenizer = Tokenizer.from_file(str(path))
            else:
                response = httpx.get(
                    f"https://huggingface.co/{model}/resolve/main/tokenizer.json",
                    follow_redirects=True,
                    timeout=5,
                )
                response.raise_for_status()
                tokenizer = Tokenizer.from_str(response.text)
                self.root.mkdir(parents=True, exist_ok=True)
                path.write_text(response.text)
            special = {
                token_id
                for token_id, token in tokenizer.get_added_tokens_decoder().items()
                if token.special
            }
            vocab = tuple(sorted(set(tokenizer.get_vocab().values()) - special))
            return (tokenizer, vocab) if vocab else None
        except Exception:
            log.warning("tokenizer unavailable for %s; using placeholder tokens", model)
            return None


def sample_tokens(
    loaded: tuple[Tokenizer, tuple[int, ...]] | None, count: int, seed: int | None, prompt: str
) -> dict:
    rng = random.Random(seed)
    if loaded is None:
        ids = [rng.randrange(50_000) for _ in range(count)]
        pieces = [f"{' ' if i else ''}tok{token_id}" for i, token_id in enumerate(ids)]
        return {
            "text": "".join(pieces),
            "pieces": pieces,
            "ids": ids,
            "source": "placeholder",
            "prompt_tokens": None,
        }
    tokenizer, vocab = loaded
    ids = [rng.choice(vocab) for _ in range(count)]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    decoder = DecodeStream(skip_special_tokens=False)
    pieces = [piece for token_id in ids if (piece := decoder.step(tokenizer, token_id))]
    prefix = "".join(pieces)
    if not text.startswith(prefix):
        pieces = [text] if text else []
    elif len(prefix) < len(text):
        pieces.append(text[len(prefix) :])
    return {
        "text": text,
        "pieces": pieces,
        "ids": ids,
        "source": "vocab_uniform",
        "prompt_tokens": len(tokenizer.encode(prompt).ids),
    }
