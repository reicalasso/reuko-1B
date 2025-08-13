"""
Data processing utilities for Reuko-1B
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataProcessor:
    """Data processing utilities"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_input_length: int = 512, max_output_length: int = 128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def preprocess_qa_batch(self, examples: Dict) -> Dict:
        """Process QA data for T5 format"""
        inputs = []
        targets = []
        
        for context, question, answers in zip(
            examples['context'], 
            examples['question'], 
            examples['answers']
        ):
            input_text = f"question: {question} context: {context}"
            inputs.append(input_text)
            
            target_text = answers['text'][0] if answers['text'] else ""
            targets.append(target_text)
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=self.max_output_length, 
                truncation=True, 
                padding=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_summarization_batch(self, examples: Dict) -> Dict:
        """Process summarization data for T5 format"""
        inputs = [f"summarize: {article}" for article in examples['article']]
        targets = examples['highlights']
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=self.max_output_length, 
                truncation=True, 
                padding=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_qa_dataset(self, train_size: Optional[int] = None, val_size: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """Load and prepare QA dataset"""
        logger.info("Loading SQuAD dataset...")
        dataset = load_dataset("squad")
        
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
        if train_size:
            train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
        if val_size:
            val_dataset = val_dataset.select(range(min(val_size, len(val_dataset))))
            
        logger.info(f"QA dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Preprocess
        train_processed = train_dataset.map(self.preprocess_qa_batch, batched=True, remove_columns=train_dataset.column_names)
        val_processed = val_dataset.map(self.preprocess_qa_batch, batched=True, remove_columns=val_dataset.column_names)
        
        return train_processed, val_processed
    
    def load_summarization_dataset(self, train_size: Optional[int] = None, val_size: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """Load and prepare summarization dataset"""
        logger.info("Loading CNN/DailyMail dataset...")
        dataset = load_dataset("cnn_dailymail", '3.0.0')
        
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
        if train_size:
            train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
        if val_size:
            val_dataset = val_dataset.select(range(min(val_size, len(val_dataset))))
            
        logger.info(f"Summarization dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Preprocess
        train_processed = train_dataset.map(self.preprocess_summarization_batch, batched=True, remove_columns=train_dataset.column_names)
        val_processed = val_dataset.map(self.preprocess_summarization_batch, batched=True, remove_columns=val_dataset.column_names)
        
        return train_processed, val_processed
    
    def save_dataset_info(self, info: Dict[str, Any], filepath: str):
        """Save dataset information"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Dataset info saved to {filepath}")
