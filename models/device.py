"""Unified device detection for CUDA / Apple MPS / CPU."""

import os
import torch


# Generation context cap on MPS. The old value (256) silently right-truncated
# real prompts — cutting off the "Score:" cue and instructions, so models just
# continued the solution text and every label was garbage. 2048 covers GSM8K/
# ProcessBench prompts comfortably; 0.5–7B fp16 KV caches at 2048 are tiny.
MPS_SAFE_MAX_LENGTH = 2048

# Dev-mode models fit alongside each other in 16GB Apple Silicon.
DEV_MODELS = {
    "teacher": "Qwen/Qwen2.5-1.5B-Instruct",   # fits in 16GB with student
    "student": "Qwen/Qwen2.5-0.5B-Instruct",
}
# Teacher is Qwen2.5-Math-72B-Instruct for paper reproducibility—standard HF model,
# math-specialized, same oracle family used by Math-Shepherd / VersaPRM.
PROD_MODELS = {
    "teacher": "google/gemma-4-26B-A4B-it",
    "student": "Qwen/Qwen2.5-1.5B-Instruct",
}


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_mps() -> bool:
    return torch.backends.mps.is_available() and not torch.cuda.is_available()


def is_dev_mode(flag: bool = False) -> bool:
    """Dev mode active if DEV_MODE env var is truthy or the flag is passed."""
    if flag:
        return True
    return os.environ.get("DEV_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def best_dtype() -> torch.dtype:
    """bfloat16 on CUDA/CPU; float16 on MPS (better hardware support)."""
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model_for_device(model_name: str, dev_mode: bool = False):
    """Load a causal LM with the right precision/placement for this machine.

    MPS: float16 + explicit .to(mps), no bitsandbytes (CUDA-only).
    CUDA dev mode: 4-bit quantization via bitsandbytes.
    CUDA full / CPU: float16 with device_map="auto".
    """
    device = best_device()
    if str(device) == "mps":
        # No bitsandbytes on MPS — use float16, no quantization.
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        return model.to(device)
    elif str(device) == "cuda" and dev_mode:
        # On CUDA in dev mode, use 4-bit.
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, trust_remote_code=True
        )
    else:
        # CUDA full precision or CPU.
        from transformers import AutoModelForCausalLM
        print(f"Native HF loader: Offloading {model_name} to CPU...")
        return AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            max_memory={0: "22GB", 1: "22GB", "cpu": "200GB"},
            trust_remote_code=True
        )


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
