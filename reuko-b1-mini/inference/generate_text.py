from transformers import pipeline

def generate_text(model, tokenizer, prompt="Merhaba, bugün hava", max_length=50, device=-1):
    # device: -1 for CPU, 0 for first GPU, 1 for second, etc.
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    return output[0]['generated_text']
