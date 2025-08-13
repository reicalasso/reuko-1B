"""
Training CLI for Reuko-1B
"""

import argparse
import sys
from pathlib import Path
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..pipeline.trainer import ReukoTrainer

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Reuko-1B models",
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
        help="Task to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit"
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger(__name__, level=log_level)
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        # Override config with CLI args
        if args.output_dir:
            config.paths.output_dir = args.output_dir
        if args.model_name:
            config.model.name = args.model_name
        if args.epochs:
            config.training.num_epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
            
        logger.info("=== Reuko-1B Training ===")
        logger.info(f"Task: {args.task}")
        logger.info(f"Model: {config.model.name}")
        logger.info(f"Output: {config.paths.output_dir}")
        logger.info(f"Epochs: {config.training.num_epochs}")
        logger.info(f"Batch size: {config.training.batch_size}")
        
        if args.dry_run:
            logger.info("Dry run mode - exiting")
            return 0
            
        # Create trainer
        trainer = ReukoTrainer(config)
        
        # Train based on task
        if args.task == "qa":
            trainer.train_qa()
        elif args.task == "summarization":
            trainer.train_summarization()
        else:  # both
            trainer.train_qa()
            trainer.train_summarization()
            
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
