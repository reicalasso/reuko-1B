try:
    from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
    QWEN2_AVAILABLE = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QWEN2_AVAILABLE = False

def load_qwen_model(model_name, quantize=False):
    """
    Qwen2.5 modeli ve tokenizer'ı yükler. trust_remote_code=True, device_map='cpu' ve padding_side ayarlanır.
    """
    if QWEN2_AVAILABLE:
        model = Qwen2ForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='cpu')
        tokenizer = Qwen2Tokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return model, tokenizer
