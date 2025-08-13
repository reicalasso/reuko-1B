import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from src.reuko.models import load_qwen_model

# Config yükle
with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
dataset_name = config["dataset_name"]

# Model & tokenizer
model, tokenizer = load_qwen_model(model_name, quantize=config.get("quantize", False))

# Dataset
dataset = load_dataset(dataset_name)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training args
args = TrainingArguments(
    output_dir="outputs/sft",
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    learning_rate=config["lr"],
    num_train_epochs=config["epochs"],
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
)

trainer.train()

# Kaydet
model.save_pretrained("outputs/sft")
tokenizer.save_pretrained("outputs/sft")
print("✅ SFT eğitimi tamamlandı ve kaydedildi: outputs/sft")
print("✅ SFT eğitimi tamamlandı ve kaydedildi: outputs/sft")
