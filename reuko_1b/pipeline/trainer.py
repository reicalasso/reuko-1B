"""
Training pipeline for Reuko-1B, enhanced for DeepSpeed and PEFT (LoRA).
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
from pathlib import Path
import json

from ..utils.config import ReukoConfig
from ..utils.data_utils import DataProcessor
from ..utils.logger import get_logger
from ..models.custom_t5 import Reuko1BModel

logger = get_logger(__name__)

class ReukoTrainer:
    """Training pipeline optimized for 1B+ models with DeepSpeed and PEFT."""
    
    def __init__(self, config: ReukoConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_directories()
        
        logger.info("Initializing ReukoTrainer for large-scale training...")
        
        # 1. Load Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        
        if self.config.model.use_custom_1b:
            logger.info("Loading custom Reuko-1B model architecture.")
            model_config = Reuko1BModel.config_class(**self.config.model_architecture)
            self.model = Reuko1BModel(model_config)
        else:
            logger.info(f"Loading base model: {self.config.model.name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.name)
            
        # 2. Apply PEFT (LoRA) if enabled
        if self.config.peft.enabled:
            logger.info("Applying PEFT with LoRA configuration.")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.config.peft.lora_r,
                lora_alpha=self.config.peft.lora_alpha,
                lora_dropout=self.config.peft.lora_dropout,
                target_modules=self.config.peft.target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        self.data_processor = DataProcessor(self.tokenizer, self.config.model.max_input_length, self.config.model.max_output_length)

    def _create_directories(self):
        Path(self.config.paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.logs_dir).mkdir(parents=True, exist_ok=True)

    def _get_training_args(self, task: str) -> TrainingArguments:
        """Build TrainingArguments with DeepSpeed and modern features."""
        output_dir = Path(self.config.paths.output_dir) / task
        
        return TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size * 2,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            num_train_epochs=self.config.training.num_epochs,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy="steps",
            bf16=self.config.training.bf16 and torch.cuda.is_bf16_supported(),
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            deepspeed=self.config.deepspeed.config_path if self.config.deepspeed.enabled else None,
            report_to=["wandb"] if self.config.monitoring.use_wandb else [],
            run_name=f"reuko-1b-{task}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )

    def train(self, task: str):
        """Unified training function for any task."""
        logger.info(f"=== Starting Training for Task: {task.upper()} ===")
        
        if task == 'qa':
            train_dataset, val_dataset = self.data_processor.load_qa_dataset(
                self.config.data.qa_train_size, self.config.data.qa_val_size
            )
        elif task == 'summarization':
            train_dataset, val_dataset = self.data_processor.load_summarization_dataset(
                self.config.data.summary_train_size, self.config.data.summary_val_size
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        training_args = self._get_training_args(task)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        train_result = trainer.train()
        
        # Save final model
        final_model_path = Path(self.config.paths.model_dir) / task
        trainer.save_model(str(final_model_path))
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
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
