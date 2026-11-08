from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
from math import ceil
from pathlib import Path

import httpx
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

log = logging.getLogger(__name__)
MODEL_NAME = re.compile(r"[\w.-]+(?:/[\w.-]+)?", re.ASCII)


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        token = next(parser.stream)
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("render"), [], [], body).set_lineno(token.lineno)

    def render(self, caller):
        return caller()


class Tokenizers:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "tokenizers"
        self._cache: dict[str, tuple[Tokenizer, tuple[int, ...]] | None] = {}
        self._chat_configs: dict[str, dict] = {}
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

    def _chat_config(self, model: str) -> dict:
        if not model or not MODEL_NAME.fullmatch(model):
            return {}
        with self._lock:
            if model not in self._chat_configs:
                path = self.root / f"{hashlib.sha256(model.encode()).hexdigest()}-config.json"
                try:
                    if path.is_file():
                        value = json.loads(path.read_text())
                    else:
                        response = httpx.get(
                            f"https://huggingface.co/{model}/resolve/main/tokenizer_config.json",
                            follow_redirects=True,
                            timeout=5,
                        )
                        response.raise_for_status()
                        value = response.json()
                        self.root.mkdir(parents=True, exist_ok=True)
                        path.write_text(response.text)
                    self._chat_configs[model] = value if isinstance(value, dict) else {}
                except Exception:
                    self._chat_configs[model] = {}
            return self._chat_configs[model]

    def prepare_prompt(self, model, prompt, messages=None):
        loaded = self.get(model)
        templated = False
        if messages is not None:
            prompt = "\n".join(m["content"] for m in messages)
            if loaded:
                config = self._chat_config(model)
                template = config.get("chat_template")
                if isinstance(template, list):
                    template = next(
                        (
                            t.get("template")
                            for t in template
                            if isinstance(t, dict) and t.get("name") == "default"
                        ),
                        None,
                    )
                elif isinstance(template, dict):
                    template = template.get("default")
                if isinstance(template, str):

                    def raise_exception(message):
                        raise ValueError(message)

                    try:
                        env = ImmutableSandboxedEnvironment(
                            trim_blocks=True,
                            lstrip_blocks=True,
                            extensions=[Generation, "jinja2.ext.loopcontrols"],
                        )
                        env.globals["raise_exception"] = raise_exception
                        special = {
                            k: v.get("content", "") if isinstance(v, dict) else v
                            for k, v in config.items()
                            if k.endswith("_token")
                        }
                        prompt = env.from_string(template).render(
                            messages=messages,
                            add_generation_prompt=True,
                            **special,
                        )
                        templated = True
                    except Exception:
                        log.warning("chat template unavailable for %s; using joined text", model)
        if loaded:
            count = len(loaded[0].encode(prompt, add_special_tokens=not templated).ids)
            source = (
                "tokenizer:chat_template"
                if templated
                else "tokenizer:joined"
                if messages is not None
                else "tokenizer"
            )
        else:
            count = max(1, ceil(len(prompt.encode("utf-8")) / 4))
            source = "estimated:utf8-bytes/4"
        return prompt, count, source, loaded


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
