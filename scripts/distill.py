import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.nn import functional as F

# Config yükleme
with open("configs/distill.yaml", "r") as f:
    config = yaml.safe_load(f)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset
dataset = load_dataset(config["dataset_name"], split="train")

def preprocess(examples):
    # Alpaca formatına uygun prompt oluşturma
    prompt = f"Instruction:\n{examples['instruction']}\n\nResponse:\n{examples['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=config["max_seq_length"])

tokenized_dataset = dataset.map(preprocess, batched=False).remove_columns(["instruction", "output", "input", "text"])


# Modeller
teacher = AutoModelForCausalLM.from_pretrained(config["teacher_model"])
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(config["student_model"])

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        labels = inputs.get("labels")
        
        # Distillation loss
        alpha_ce = config.get("alpha_ce", 0.5)
        alpha_kd = config.get("alpha_mlm", 0.5) # config'de alpha_mlm olarak geçiyor
        
        ce_loss = student_outputs.loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (student_logits.size(-1)) # Scale by vocab size

        loss = alpha_ce * ce_loss + alpha_kd * kd_loss
        return (loss, student_outputs) if return_outputs else loss

# Training Arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    logging_steps=config["logging_steps"],
    save_strategy="epoch",
    fp16=True,
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    label_names=["labels"], # Önemli: Trainer'a etiketleri bildirmek için
)

# Trainer
trainer = DistillationTrainer(
    model=student,
    teacher_model=teacher,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]), 
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['input_ids'] for f in data])}
)

# Eğitim
trainer.train()

# Kaydet
trainer.save_model(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print(f"✅ Distilled model kaydedildi: {config['output_dir']}")
