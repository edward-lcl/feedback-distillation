"""Unified device detection for CUDA / Apple MPS / CPU."""

import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_dtype() -> torch.dtype:
    """bfloat16 on CUDA/CPU; float16 on MPS (better hardware support)."""
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model_auto(model_name: str, **kwargs):
    """Load a causal LM with the right dtype and device for this machine."""
    from transformers import AutoModelForCausalLM
    dtype = best_dtype()
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **kwargs,
    )
