import hashlib
import json

import httpx
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cascade.openai_tokens import Tokenizers, sample_tokens


def vocabulary():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["[UNK]", "[EOS]"])
    return tokenizer


def test_download_cache_uniform_exact_count_seed_and_special_tokens(tmp_path, monkeypatch):
    tokenizer = vocabulary()
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, text=tokenizer.to_str(), request=httpx.Request("GET", url))

    monkeypatch.setattr("cascade.openai_tokens.httpx.get", fetch)
    cache = Tokenizers(tmp_path)
    loaded = cache.get("test/model")
    assert loaded[1] == (1, 2)
    assert cache.get("test/model") is loaded and len(calls) == 1
    first = sample_tokens(loaded, 100, 123, "hello world")
    assert len(first["ids"]) == 100 and set(first["ids"]) == {1, 2}
    assert first == sample_tokens(loaded, 100, 123, "hello world")
    assert first["ids"] != sample_tokens(loaded, 100, 124, "hello world")["ids"]
    assert first["source"] == "vocab_uniform" and first["prompt_tokens"] == 2
    assert "".join(first["pieces"]) == first["text"] == tokenizer.decode(first["ids"])
    assert sample_tokens(loaded, 0, 0, "")["text"] == ""
    monkeypatch.setattr("cascade.openai_tokens.httpx.get", lambda *a, **k: 1 / 0)
    assert Tokenizers(tmp_path).get("test/model")[1] == (1, 2)


def test_offline_and_unsupported_model_fall_back_without_retries(tmp_path, monkeypatch):
    calls = []

    def offline(*args, **kwargs):
        calls.append(args)
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("cascade.openai_tokens.httpx.get", offline)
    cache = Tokenizers(tmp_path)
    assert cache.get("test/model") is None and cache.get("test/model") is None
    assert len(calls) == 1
    assert cache.get("../../etc/passwd") is None and len(calls) == 1
    first = sample_tokens(None, 7, 123, "hello")
    assert len(first["text"].split()) == 7 and first["source"] == "placeholder"
    assert first == sample_tokens(None, 7, 123, "hello")


def test_no_nonspecial_vocab_falls_back(tmp_path):
    root = tmp_path / "tokenizers"
    root.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]"])
    name = hashlib.sha256(b"test/model").hexdigest()
    (root / f"{name}.json").write_text(json.dumps(json.loads(tokenizer.to_str())))
    assert Tokenizers(tmp_path).get("test/model") is None
