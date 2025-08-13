from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--output_dir", type=str, default="data/")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Dataset
dataset = load_dataset(args.dataset)

def format_prompt(example):
    # Alpaca formatı
    if example.get("input"):
        return f"Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\nResponse:\n{example['output']}"
    return f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['output']}"

tokenized_dataset = dataset.map(lambda x: tokenizer(format_prompt(x), truncation=True, padding="max_length"), batched=False)

tokenized_dataset.save_to_disk(args.output_dir)
print(f"✅ Veri hazır: {args.output_dir}")
