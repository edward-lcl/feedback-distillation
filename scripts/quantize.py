import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

import bitsandbytes.nn.modules as bnb_modules
original_new = bnb_modules.Params4bit.__new__
def custom_new(cls, *args, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return original_new(cls, *args, **kwargs)
bnb_modules.Params4bit.__new__ = staticmethod(custom_new)

import transformers.modeling_utils
transformers.modeling_utils.remove_tied_weights_from_state_dict = lambda state_dict, *args, **kwargs: state_dict

def main():
    model_id = "google/gemma-4-26B-A4B-it"
    save_dir = "/home/skapoor/feedback-distillation/models/gemma4-bnb-4bit"
    token = os.environ.get("HF_TOKEN")

    os.makedirs(save_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)

    print("Initializing BitsAndBytesConfig...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True
    )

    print("Loading model and quantizing on the fly...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "20GB", 1: "20GB", "cpu": "200GB"},
        token=token,
        trust_remote_code=True
    )

    print("Saving quantized model to disk...")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print("Quantization complete! Saved to:", save_dir)

if __name__ == "__main__":
    main()
