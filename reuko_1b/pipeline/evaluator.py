"""
Evaluation pipeline for Reuko-1B
"""

import torch
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..utils.config import ReukoConfig
from ..utils.data_utils import DataProcessor
from ..utils.metrics import MetricsCalculator
from ..utils.logger import get_logger
from ..pipeline.inference import ReukoInference

logger = get_logger(__name__)

class ReukoEvaluator:
    """Evaluation pipeline for Reuko models"""
    
    def __init__(self, config: ReukoConfig):
        self.config = config
        self.inference = ReukoInference(config)
        self.metrics_calculator = MetricsCalculator()
        self.data_processor = DataProcessor(
            None,  # Will be set when loading models
            config.model.max_input_length,
            config.model.max_output_length
        )
        
    def evaluate_qa(self, model_path: Optional[str] = None, test_size: int = 100) -> Dict[str, Any]:
        """Evaluate QA model on SQuAD dataset"""
        logger.info("=== QA Evaluation ===")
        
        # Load QA model
        qa_model = self.inference.load_qa_model(model_path)
        
        # Load test dataset
        dataset = load_dataset("squad")
        test_data = dataset["validation"].select(range(min(test_size, len(dataset["validation"]))))
        
        predictions = []
        references = []
        
        logger.info(f"Evaluating on {len(test_data)} examples...")
        
        for i, example in enumerate(test_data):
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(test_data)}")
                
            question = example["question"]
            context = example["context"]
            reference_answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            
            # Get prediction
            result = self.inference.answer_question(question, context)
            prediction = result["answer"]
            
            predictions.append(prediction)
            references.append(reference_answer)
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_qa(predictions, references)
        
        evaluation_results = {
            "task": "qa",
            "model_path": str(model_path) if model_path else "base_model",
            "test_size": len(test_data),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "question": test_data[i]["question"],
                    "context": test_data[i]["context"][:200] + "..." if len(test_data[i]["context"]) > 200 else test_data[i]["context"],
                    "reference": references[i],
                    "prediction": predictions[i]
                }
                for i in range(min(5, len(predictions)))  # Show first 5 examples
            ]
        }
        
        logger.info(f"QA Evaluation Results:")
        logger.info(f"- Exact Match: {metrics['exact_match']:.3f}")
        logger.info(f"- F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"- BLEU Score: {metrics['bleu_score']:.3f}")
        
        return evaluation_results
    
    def evaluate_summarization(self, model_path: Optional[str] = None, test_size: int = 100) -> Dict[str, Any]:
        """Evaluate summarization model on CNN/DailyMail dataset"""
        logger.info("=== Summarization Evaluation ===")
        
        # Load summarization model
        sum_model = self.inference.load_summarization_model(model_path)
        
        # Load test dataset
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        test_data = dataset["validation"].select(range(min(test_size, len(dataset["validation"]))))
        
        predictions = []
        references = []
        original_texts = []
        
        logger.info(f"Evaluating on {len(test_data)} examples...")
        
        for i, example in enumerate(test_data):
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(test_data)}")
                
            article = example["article"]
            reference_summary = example["highlights"]
            
            # Get prediction
            result = self.inference.summarize_text(article)
            prediction = result["summary"]
            
            predictions.append(prediction)
            references.append(reference_summary)
            original_texts.append(article)
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_summarization(predictions, references, original_texts)
        
        evaluation_results = {
            "task": "summarization",
            "model_path": str(model_path) if model_path else "base_model",
            "test_size": len(test_data),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "article": original_texts[i][:300] + "..." if len(original_texts[i]) > 300 else original_texts[i],
                    "reference": references[i],
                    "prediction": predictions[i],
                    "compression_ratio": len(predictions[i].split()) / len(original_texts[i].split())
                }
                for i in range(min(5, len(predictions)))  # Show first 5 examples
            ]
        }
        
        logger.info(f"Summarization Evaluation Results:")
        logger.info(f"- BLEU Score: {metrics['bleu_score']:.3f}")
        logger.info(f"- ROUGE-L F1: {metrics['rouge_l']['f1']:.3f}")
        if 'compression_ratio' in metrics:
            logger.info(f"- Compression Ratio: {metrics['compression_ratio']['mean']:.3f}")
        
        return evaluation_results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Evaluation results saved to {output_path}")
        
    def compare_models(self, model_paths: List[str], task: str = "qa", test_size: int = 100) -> Dict[str, Any]:
        """Compare multiple models"""
        logger.info(f"=== Model Comparison ({task}) ===")
        
        comparison_results = {
            "task": task,
            "models": [],
            "test_size": test_size,
            "timestamp": datetime.now().isoformat()
        }
        
        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")
            
            if task == "qa":
                result = self.evaluate_qa(model_path, test_size)
            else:
                result = self.evaluate_summarization(model_path, test_size)
                
            comparison_results["models"].append(result)
        
        # Find best model
        if task == "qa":
            best_model = max(comparison_results["models"], key=lambda x: x["metrics"]["f1_score"])
        else:
            best_model = max(comparison_results["models"], key=lambda x: x["metrics"]["rouge_l"]["f1"])
            
        comparison_results["best_model"] = best_model["model_path"]
        
        logger.info(f"Best model: {comparison_results['best_model']}")
        
        return comparison_results
