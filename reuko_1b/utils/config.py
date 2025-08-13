"""
Configuration Management for Reuko-1B
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "t5-small"
    max_input_length: int = 512
    max_output_length: int = 128
    num_beams: int = 4
    early_stopping: bool = True
    do_sample: bool = False

@dataclass 
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 2
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

@dataclass
class DataConfig:
    """Data configuration"""
    qa_train_size: int = 1000
    qa_val_size: int = 200
    summary_train_size: int = 1000
    summary_val_size: int = 200
    cache_dir: str = "./data_cache"
    
@dataclass
class PathConfig:
    """Path configuration"""
    output_dir: str = "./outputs"
    model_dir: str = "./models"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"

@dataclass
class ReukoConfig:
    """Main configuration class"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ReukoConfig':
        """Load config from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return cls.default()
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
        )
    
    @classmethod
    def default(cls) -> 'ReukoConfig':
        """Create default configuration"""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            paths=PathConfig(),
        )
    
    def to_yaml(self, config_path: str):
        """Save config to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training), 
            'data': asdict(self.data),
            'paths': asdict(self.paths),
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./config.yaml"
        self._config = None
    
    @property
    def config(self) -> ReukoConfig:
        """Get configuration (lazy loading)"""
        if self._config is None:
            self._config = ReukoConfig.from_yaml(self.config_path)
        return self._config
    
    def reload(self):
        """Reload configuration"""
        self._config = None
        logger.info("Configuration reloaded")
    
    def save_default_config(self):
        """Save default configuration to file"""
        default_config = ReukoConfig.default()
        default_config.to_yaml(self.config_path)
