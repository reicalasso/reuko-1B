"""
Unit tests for Reuko-1B models
"""

import unittest
import torch
from transformers import AutoTokenizer
from reuko_1b.models.t5_model import T5QAModel, T5SummarizationModel
from reuko_1b.models.base_model import BaseReukoModel

class TestT5Models(unittest.TestCase):
    """Test T5 model implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_name = "t5-small"
        self.qa_model = T5QAModel(self.model_name)
        self.sum_model = T5SummarizationModel(self.model_name)
        
    def test_qa_model_initialization(self):
        """Test QA model initialization"""
        self.assertEqual(self.qa_model.model_name, self.model_name)
        self.assertIsNotNone(self.qa_model.tokenizer)
        self.assertIsNotNone(self.qa_model.model)
        
    def test_summarization_model_initialization(self):
        """Test summarization model initialization"""
        self.assertEqual(self.sum_model.model_name, self.model_name)
        self.assertIsNotNone(self.sum_model.tokenizer)
        self.assertIsNotNone(self.sum_model.model)
        
    def test_qa_preprocessing(self):
        """Test QA input preprocessing"""
        input_data = {
            "question": "What is Python?",
            "context": "Python is a programming language."
        }
        processed = self.qa_model.preprocess_input(input_data)
        expected = "question: What is Python? context: Python is a programming language."
        self.assertEqual(processed, expected)
        
    def test_summarization_preprocessing(self):
        """Test summarization input preprocessing"""
        text = "This is a long article to be summarized."
        processed = self.sum_model.preprocess_input(text)
        expected = "summarize: This is a long article to be summarized."
        self.assertEqual(processed, expected)
        
    def test_qa_postprocessing(self):
        """Test QA output postprocessing"""
        output = "question: The answer is Python."
        processed = self.qa_model.postprocess_output(output)
        expected = "The answer is Python."
        self.assertEqual(processed, expected)
        
    def test_summarization_postprocessing(self):
        """Test summarization output postprocessing"""
        output = "summarize: This is a summary."
        processed = self.sum_model.postprocess_output(output)
        expected = "This is a summary."
        self.assertEqual(processed, expected)
        
    def test_model_info(self):
        """Test model info retrieval"""
        info = self.qa_model.get_model_info()
        
        self.assertIn("model_name", info)
        self.assertIn("total_parameters", info)
        self.assertIn("trainable_parameters", info)
        self.assertIn("model_size_mb", info)
        self.assertIn("device", info)
        
        self.assertGreater(info["total_parameters"], 0)

if __name__ == "__main__":
    unittest.main()
