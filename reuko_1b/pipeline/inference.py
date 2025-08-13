"""
Inference pipeline for Reuko-1B
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime

from ..models.t5_model import T5QAModel, T5SummarizationModel
from ..utils.config import ReukoConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ReukoInference:
    """Inference pipeline for Reuko models"""
    
    def __init__(self, config: Optional[ReukoConfig] = None):
        self.config = config or ReukoConfig.default()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.qa_model: Optional[T5QAModel] = None
        self.summarization_model: Optional[T5SummarizationModel] = None
        
        logger.info(f"ReukoInference initialized with device: {self.device}")
    
    def load_qa_model(self, model_path: Optional[str] = None) -> T5QAModel:
        """Load QA model"""
        if model_path is None:
            model_path = Path(self.config.paths.model_dir) / "qa"
            
        logger.info(f"Loading QA model from {model_path}")
        
        if Path(model_path).exists():
            # Load fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            
            self.qa_model = T5QAModel.__new__(T5QAModel)
            self.qa_model.model_name = str(model_path)
            self.qa_model.tokenizer = tokenizer
            self.qa_model.model = model
            self.qa_model.device = self.device
            self.qa_model.model.to(self.device)
            self.qa_model.max_input_length = self.config.model.max_input_length
            self.qa_model.max_output_length = self.config.model.max_output_length
        else:
            # Load base model
            self.qa_model = T5QAModel(
                self.config.model.name,
                max_input_length=self.config.model.max_input_length,
                max_output_length=self.config.model.max_output_length
            )
            
        return self.qa_model
    
    def load_summarization_model(self, model_path: Optional[str] = None) -> T5SummarizationModel:
        """Load summarization model"""
        if model_path is None:
            model_path = Path(self.config.paths.model_dir) / "summarization"
            
        logger.info(f"Loading summarization model from {model_path}")
        
        if Path(model_path).exists():
            # Load fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            
            self.summarization_model = T5SummarizationModel.__new__(T5SummarizationModel)
            self.summarization_model.model_name = str(model_path)
            self.summarization_model.tokenizer = tokenizer
            self.summarization_model.model = model
            self.summarization_model.device = self.device
            self.summarization_model.model.to(self.device)
            self.summarization_model.max_input_length = self.config.model.max_input_length
            self.summarization_model.max_output_length = self.config.model.max_output_length
        else:
            # Load base model
            self.summarization_model = T5SummarizationModel(
                self.config.model.name,
                max_input_length=self.config.model.max_input_length,
                max_output_length=self.config.model.max_output_length
            )
            
        return self.summarization_model
    
    def answer_question(self, question: str, context: str, **generation_kwargs) -> Dict[str, Any]:
        """Answer a question"""
        if self.qa_model is None:
            self.load_qa_model()
            
        return self.qa_model.answer_question(question, context)
    
    def summarize_text(self, text: str, max_length: Optional[int] = None, **generation_kwargs) -> Dict[str, Any]:
        """Summarize text"""
        if self.summarization_model is None:
            self.load_summarization_model()
            
        return self.summarization_model.summarize_text(text, max_length)
    
    def batch_qa(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Batch question answering"""
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have the same length")
            
        results = []
        for question, context in zip(questions, contexts):
            result = self.answer_question(question, context)
            results.append(result)
            
        return results
    
    def batch_summarization(self, texts: List[str], max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Batch summarization"""
        results = []
        for text in texts:
            result = self.summarize_text(text, max_length)
            results.append(result)
            
        return results
    
    def save_results(self, results: Union[Dict, List], output_path: str):
        """Save inference results"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, dict):
            results['timestamp'] = datetime.now().isoformat()
        elif isinstance(results, list) and len(results) > 0:
            for result in results:
                if isinstance(result, dict):
                    result['timestamp'] = datetime.now().isoformat()
                    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {output_path}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "qa_model_loaded": self.qa_model is not None,
            "summarization_model_loaded": self.summarization_model is not None,
            "device": str(self.device),
            "qa_model_info": self.qa_model.get_model_info() if self.qa_model else None,
            "summarization_model_info": self.summarization_model.get_model_info() if self.summarization_model else None
        }
