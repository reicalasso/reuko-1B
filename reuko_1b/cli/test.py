"""
Testing CLI for Reuko-1B
"""

import argparse
import sys
from pathlib import Path
import json
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..pipeline.inference import ReukoInference
from ..pipeline.evaluator import ReukoEvaluator

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test Reuko-1B models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["qa", "summarization", "both"],
        default="both",
        help="Task to test"
    )
    
    parser.add_argument(
        "--qa-model",
        type=str,
        help="Path to QA model"
    )
    
    parser.add_argument(
        "--summarization-model",
        type=str,
        help="Path to summarization model"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input text file or single text"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./test_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on test dataset"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main testing function"""
    args = parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger(__name__, level=log_level)
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        # Initialize inference
        inference = ReukoInference(config)
        
        logger.info("=== Reuko-1B Testing ===")
        
        if args.evaluate:
            # Run evaluation
            evaluator = ReukoEvaluator(config)
            
            if args.task in ["qa", "both"]:
                qa_results = evaluator.evaluate_qa(args.qa_model)
                logger.info(f"QA Evaluation Results: {qa_results}")
                
            if args.task in ["summarization", "both"]:
                sum_results = evaluator.evaluate_summarization(args.summarization_model)
                logger.info(f"Summarization Evaluation Results: {sum_results}")
                
        else:
            # Interactive testing
            if args.input:
                if Path(args.input).exists():
                    # File input
                    with open(args.input, 'r') as f:
                        input_text = f.read().strip()
                else:
                    # Direct text input
                    input_text = args.input
                    
                results = []
                
                if args.task in ["qa", "both"]:
                    # Assume format: "question|context"
                    if "|" in input_text:
                        question, context = input_text.split("|", 1)
                        qa_result = inference.answer_question(question.strip(), context.strip())
                        results.append({"type": "qa", "result": qa_result})
                        
                if args.task in ["summarization", "both"]:
                    sum_result = inference.summarize_text(input_text)
                    results.append({"type": "summarization", "result": sum_result})
                
                # Save results
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"Results saved to {args.output}")
                
            else:
                # Interactive mode
                logger.info("Interactive mode - type 'quit' to exit")
                
                while True:
                    task = input("\nSelect task (qa/summarization): ").strip().lower()
                    
                    if task == "quit":
                        break
                        
                    if task == "qa":
                        question = input("Question: ")
                        context = input("Context: ")
                        result = inference.answer_question(question, context)
                        print(f"Answer: {result['answer']}")
                        
                    elif task == "summarization":
                        text = input("Text to summarize: ")
                        result = inference.summarize_text(text)
                        print(f"Summary: {result['summary']}")
                        
                    else:
                        print("Invalid task. Use 'qa' or 'summarization'")
        
        return 0
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
