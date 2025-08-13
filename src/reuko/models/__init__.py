from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name: str, quantize: bool = False):
    """Modeli ve tokenizer'ı yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    return model, tokenizer
