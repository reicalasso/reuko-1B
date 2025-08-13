import torch
from torch.utils.data import DataLoader
from transformers import AdamW

class SFTTrainer:
    def __init__(self, model, tokenizer, dataset, lr=5e-5, batch_size=2):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = input_ids.clone()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
