"""
Evaluation metrics for Reuko-1B
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from collections import Counter
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """Calculate exact match score"""
        if len(predictions) != len(references):
            logger.warning("Predictions and references have different lengths")
            
        exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                           if pred.strip().lower() == ref.strip().lower())
        return exact_matches / len(references) if references else 0.0
    
    @staticmethod
    def f1_score(predictions: List[str], references: List[str]) -> float:
        """Calculate F1 score for token-level overlap"""
        if len(predictions) != len(references):
            logger.warning("Predictions and references have different lengths")
            
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = MetricsCalculator._normalize_answer(pred).split()
            ref_tokens = MetricsCalculator._normalize_answer(ref).split()
            
            common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common_tokens.values())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
                
            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for comparison"""
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
        """Simple BLEU score implementation"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens = [ref.lower().split()]  # BLEU expects list of reference lists
                
                if len(pred_tokens) == 0:
                    scores.append(0.0)
                else:
                    score = sentence_bleu(ref_tokens, pred_tokens)
                    scores.append(score)
                    
            return np.mean(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("NLTK not available, skipping BLEU score")
            return 0.0
    
    @staticmethod 
    def rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE-L implementation"""
        scores = {"precision": [], "recall": [], "f1": []}
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                scores["precision"].append(1.0)
                scores["recall"].append(1.0)
                scores["f1"].append(1.0)
                continue
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                scores["precision"].append(0.0)
                scores["recall"].append(0.0)
                scores["f1"].append(0.0)
                continue
                
            # Find LCS
            lcs_length = MetricsCalculator._lcs_length(pred_tokens, ref_tokens)
            
            precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                
            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)
        
        return {
            "precision": np.mean(scores["precision"]),
            "recall": np.mean(scores["recall"]),
            "f1": np.mean(scores["f1"])
        }
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    @staticmethod
    def compression_ratio(original_texts: List[str], summaries: List[str]) -> Dict[str, float]:
        """Calculate compression ratios for summarization"""
        ratios = []
        
        for original, summary in zip(original_texts, summaries):
            original_words = len(original.split())
            summary_words = len(summary.split())
            
            if original_words > 0:
                ratio = summary_words / original_words
                ratios.append(ratio)
            
        return {
            "mean": np.mean(ratios) if ratios else 0.0,
            "std": np.std(ratios) if ratios else 0.0,
            "min": np.min(ratios) if ratios else 0.0,
            "max": np.max(ratios) if ratios else 0.0
        }
    
    def evaluate_qa(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Complete QA evaluation"""
        return {
            "exact_match": self.exact_match(predictions, references),
            "f1_score": self.f1_score(predictions, references),
            "bleu_score": self.bleu_score(predictions, references)
        }
    
    def evaluate_summarization(self, predictions: List[str], references: List[str], 
                             original_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Complete summarization evaluation"""
        results = {
            "bleu_score": self.bleu_score(predictions, references),
            "rouge_l": self.rouge_l(predictions, references)
        }
        
        if original_texts:
            results["compression_ratio"] = self.compression_ratio(original_texts, predictions)
            
        return results
