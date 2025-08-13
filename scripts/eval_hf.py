import yaml
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config yükle
with open("configs/eval.yaml", "r") as f:
    config = yaml.safe_load(f)

# Model ve tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config["model_path"])
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Dataset
dataset = load_dataset(config["dataset_name"], split="test")

# Metricler
metrics = {}
for m in config["metrics"]:
    metrics[m] = load_metric(m)

def evaluate_batch(batch):
    prompts = [item["instruction"] for item in batch]
    labels = [item["output"] for item in batch]
    
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config["max_seq_length"])
    
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions, labels

# Evaluation loop
results = {m: [] for m in metrics}
for i in range(0, len(dataset), config["batch_size"]):
    batch = dataset[i:i+config["batch_size"]]
    # The dataset from hf is a dict of lists, not a list of dicts.
    # So we need to construct the batch differently.
    batch_prompts = batch["instruction"]
    batch_labels = batch["output"]
    
    inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config["max_seq_length"])
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for pred, label in zip(predictions, batch_labels):
        for m in metrics:
            if m == "accuracy":
                results[m].append(int(pred.strip() == label.strip()))
            elif m == "bleu":
                results[m].append(metrics[m].compute(predictions=[pred.split()], references=[[label.split()]] )["bleu"])

# Metric ortalamaları
for m in results:
    print(f"{m}: {sum(results[m])/len(results[m]):.4f}")
