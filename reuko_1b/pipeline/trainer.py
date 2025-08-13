"""
Training pipeline for Reuko-1B
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..utils.config import ReukoConfig
from ..utils.data_utils import DataProcessor
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCalculator

logger = get_logger(__name__)

class ReukoTrainer:
    """Training pipeline for Reuko models"""
    
    def __init__(self, config: ReukoConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing ReukoTrainer with device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name)
        self.model.to(self.device)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            self.tokenizer,
            config.model.max_input_length,
            config.model.max_output_length
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Create output directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.config.paths.output_dir, self.config.paths.model_dir, self.config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def _get_training_args(self, task: str) -> TrainingArguments:
        """Get training arguments for specific task"""
        output_dir = Path(self.config.paths.output_dir) / task
        
        return TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            num_train_epochs=self.config.training.num_epochs,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_num_workers=2,
            fp16=self.config.training.fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            logging_dir=str(Path(self.config.paths.logs_dir) / task),
        )
    
    def train_qa(self) -> Dict[str, Any]:
        """Train QA model"""
        logger.info("=== Starting QA Training ===")
        start_time = datetime.now()
        
        # Load data
        train_dataset, val_dataset = self.data_processor.load_qa_dataset(
            self.config.data.qa_train_size,
            self.config.data.qa_val_size
        )
        
        # Setup training
        training_args = self._get_training_args("qa")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"QA training completed in {training_duration}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save model
        model_dir = Path(self.config.paths.model_dir) / "qa"
        trainer.save_model(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        
        # Save training results
        results = {
            "task": "qa",
            "model_name": self.config.model.name,
            "training_loss": train_result.training_loss,
            "duration": str(training_duration),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "config": {
                "batch_size": self.config.training.batch_size,
                "epochs": self.config.training.num_epochs,
                "learning_rate": self.config.training.learning_rate,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = Path(self.config.paths.output_dir) / "qa_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training results saved to {results_path}")
        return results
    
    def train_summarization(self) -> Dict[str, Any]:
        """Train summarization model"""
        logger.info("=== Starting Summarization Training ===")
        start_time = datetime.now()
        
        # Load data
        train_dataset, val_dataset = self.data_processor.load_summarization_dataset(
            self.config.data.summary_train_size,
            self.config.data.summary_val_size
        )
        
        # Setup training
        training_args = self._get_training_args("summarization")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Summarization training completed in {training_duration}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save model
        model_dir = Path(self.config.paths.model_dir) / "summarization"
        trainer.save_model(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        
        # Save training results
        results = {
            "task": "summarization",
            "model_name": self.config.model.name,
            "training_loss": train_result.training_loss,
            "duration": str(training_duration),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "config": {
                "batch_size": self.config.training.batch_size,
                "epochs": self.config.training.num_epochs,
                "learning_rate": self.config.training.learning_rate,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = Path(self.config.paths.output_dir) / "summarization_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training results saved to {results_path}")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model.name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),
            "device": str(self.device),
            "fp16_enabled": self.config.training.fp16 and torch.cuda.is_available()
        }
