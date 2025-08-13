import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Config yükleme
with open("configs/rl_grpo.yaml", "r") as f:
    config = yaml.safe_load(f)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

# Model (value head ile)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"]
)
device = model.device

# PPO Config
ppo_config = PPOConfig(
    model_name=config["model_name"],
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    forward_batch_size=config["batch_size"],
    log_with=None,  # WandB kullanabilirsin
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# Dataset yükleme
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Basit reward fonksiyonu (örnek, kendi reward modelinle değiştir)
reward_model = AutoModelForSequenceClassification.from_pretrained(config["reward_model"])
reward_model.to(device) # Modeli GPU'ya taşı

def compute_rewards(batch):
    # batch["instruction"] bir liste olacak
    texts = batch["instruction"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
    # rewards'ı liste olarak döndür
    return [reward[0] for reward in outputs.logits]

# Training loop
for epoch in range(config["num_train_epochs"]):
    for batch in dataset.shuffle().select(range(1000)):  # örnek: ilk 1000 veri
        query_tensors = tokenizer(batch["instruction"], return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
        
        # Response üretme
        response_tensors = ppo_trainer.generate(query_tensors, max_length=config["max_seq_length"])
        
        # Reward hesaplama
        rewards = compute_rewards(batch)
        
        # PPO adımı
        # query, response ve reward'ların tensor olması gerekiyor
        rewards_tensor = torch.tensor(rewards).to(device)
        ppo_trainer.step([q for q in query_tensors], [r for r in response_tensors], rewards_tensor)

# Model kaydet
model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print(f"✅ RLHF modeli kaydedildi: {config['output_dir']}")
