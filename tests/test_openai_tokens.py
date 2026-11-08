import hashlib
import json

import httpx
import pytest
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


@pytest.fixture
def bpe_downloads(monkeypatch):
    resources = {
        "tokenizer.json": (404, None),
        "tokenizer_config.json": (
            200,
            {
                "tokenizer_class": "GPT2Tokenizer",
                "add_bos_token": True,
                "add_prefix_space": False,
                "bos_token": {"content": "</s>", "normalized": True},
                "unk_token": "<unk>",
            },
        ),
        "vocab.json": (
            200,
            {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "h": 4,
                "i": 5,
                "hi": 6,
                "Ġ": 7,
                "Ġhi": 8,
                "#": 9,
                "##": 10,
            },
        ),
        "merges.txt": (200, "#version: 0.2\nh i\nĠ hi\n# #\n"),
        "special_tokens_map.json": (
            200,
            {
                "eos_token": "</s>",
                "pad_token": {"content": "<pad>", "normalized": True},
                "additional_special_tokens": ["<s>"],
            },
        ),
    }
    calls = []

    def fetch(url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        status, value = resources[name]
        return httpx.Response(
            status,
            text=value if isinstance(value, str) else json.dumps(value),
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("cascade.openai_tokens.httpx.get", fetch)
    return resources, calls


@pytest.mark.parametrize("add_bos", [False, True])
def test_bpe_download_bos_decode_and_disk_cache(tmp_path, monkeypatch, bpe_downloads, add_bos):
    resources, calls = bpe_downloads
    resources["tokenizer_config.json"][1]["add_bos_token"] = add_bos
    cache = Tokenizers(tmp_path)
    loaded = cache.get("test/model")
    prefix = [2] if add_bos else []
    assert loaded[0].encode("hi hi").ids == prefix + [6, 8]
    assert loaded[0].encode("##").ids == prefix + [10]
    assert loaded[0].encode("").ids == prefix
    assert loaded[0].encode("hi", add_special_tokens=False).ids == [6]
    assert loaded[1] == tuple(range(4, 11))
    sampled = sample_tokens(loaded, 32, 123, "hi hi")
    assert sampled["source"] == "vocab_uniform"
    assert sampled["text"] == "".join(sampled["pieces"]) == loaded[0].decode(sampled["ids"])
    assert set(sampled["text"]) <= set("hi #")
    assert len(sampled["ids"]) == 32
    assert sample_tokens(loaded, 0, 123, "hi")["text"] == ""
    assert len(calls) == 5
    assert cache.get("test/model") is loaded and len(calls) == 5
    monkeypatch.setattr("cascade.openai_tokens.httpx.get", lambda *a, **k: 1 / 0)
    restarted = Tokenizers(tmp_path)
    assert restarted.get("test/model")[0].encode("hi hi").ids == prefix + [6, 8]
    assert restarted.prepare_prompt("test/model", "hi", [{"content": "hi"}])[:3] == (
        "hi",
        len(prefix) + 1,
        "tokenizer:joined",
    )


@pytest.mark.parametrize("resource", ["vocab.json", "merges.txt", "tokenizer_config.json"])
def test_bpe_missing_required_download_falls_back(tmp_path, bpe_downloads, resource):
    resources, calls = bpe_downloads
    resources[resource] = (404, None)
    cache = Tokenizers(tmp_path)
    assert cache.prepare_prompt("test/model", "hi")[:3] == (
        "hi",
        1,
        "estimated:utf8-bytes/4",
    )
    count = len(calls)
    assert cache.get("test/model") is None and len(calls) == count
    assert sample_tokens(cache.get("test/model"), 2, 123, "hi")["source"] == "placeholder"


def test_bpe_optional_special_map_can_be_absent(tmp_path, bpe_downloads):
    resources, _ = bpe_downloads
    resources["special_tokens_map.json"] = (404, None)
    assert Tokenizers(tmp_path).get("test/model")[0].encode("hi").ids == [2, 6]


@pytest.mark.parametrize("invalid", ["missing_bos", "vocab", "config", "merges", "class"])
def test_bpe_invalid_resources_fall_back(tmp_path, bpe_downloads, invalid):
    resources, _ = bpe_downloads
    if invalid == "missing_bos":
        resources["tokenizer_config.json"][1].pop("bos_token")
    elif invalid == "class":
        resources["tokenizer_config.json"][1]["tokenizer_class"] = "RobertaTokenizer"
    else:
        name = {"vocab": "vocab.json", "config": "tokenizer_config.json", "merges": "merges.txt"}
        resources[name[invalid]] = (200, "invalid")
    assert Tokenizers(tmp_path).get("test/model") is None


def test_chat_config_download_disk_cache_and_render(tmp_path, monkeypatch):
    model = "test/model"
    loaded = (vocabulary(), (1, 2))
    calls = []
    config = {"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}hello"}

    def fetch(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, json=config, request=httpx.Request("GET", url))

    monkeypatch.setattr("cascade.openai_tokens.httpx.get", fetch)
    cache = Tokenizers(tmp_path)
    cache._cache[model] = loaded
    messages = [{"role": "user", "content": "hello world"}]
    first = cache.prepare_prompt(model, "ignored", messages)
    assert first[:3] == ("hello world hello", 3, "tokenizer:chat_template")
    assert cache.prepare_prompt(model, "ignored", messages) == first
    assert len(calls) == 1 and calls[0].endswith("tokenizer_config.json")
    monkeypatch.setattr("cascade.openai_tokens.httpx.get", lambda *a, **k: 1 / 0)
    restarted = Tokenizers(tmp_path)
    restarted._cache[model] = loaded
    assert restarted.prepare_prompt(model, "ignored", messages) == first


@pytest.mark.parametrize("config", [{}, {"chat_template": "{{ raise_exception('unsupported') }}"}])
def test_missing_or_failed_chat_template_uses_joined_text(tmp_path, monkeypatch, config):
    monkeypatch.setattr(Tokenizers, "get", lambda self, model: (vocabulary(), (1, 2)))
    monkeypatch.setattr(Tokenizers, "_chat_config", lambda self, model: config)
    result = Tokenizers(tmp_path).prepare_prompt(
        "test/model",
        "ignored",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    )
    assert result[:3] == ("hello\nworld", 2, "tokenizer:joined")
