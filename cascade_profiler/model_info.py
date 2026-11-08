"""Small JSON-safe summary of the model actually loaded by vLLM."""


def model_summary(llm) -> dict:
    engine = llm.llm_engine
    model = getattr(engine, "model_config", None)
    hf = getattr(model, "hf_text_config", None) or getattr(model, "hf_config", None)
    outer = getattr(model, "hf_config", hf)
    architectures = getattr(outer, "architectures", None)
    heads = getattr(hf, "num_attention_heads", None)
    hidden = getattr(hf, "hidden_size", None)
    dtype = getattr(model, "dtype", None)
    parallel = getattr(getattr(engine, "vllm_config", None), "parallel_config", None)
    return {"architecture": architectures[0] if architectures else getattr(hf, "model_type", None),
            "num_layers": getattr(hf, "num_hidden_layers", None), "hidden_size": hidden,
            "num_heads": heads, "num_kv_heads": getattr(hf, "num_key_value_heads", heads),
            "head_dim": getattr(hf, "head_dim", None) or (hidden // heads if hidden and heads else None),
            "dtype": str(dtype).removeprefix("torch.") if dtype is not None else None,
            "tensor_parallel_size": getattr(parallel, "tensor_parallel_size", 1),
            "quantization": getattr(model, "quantization", None)}
