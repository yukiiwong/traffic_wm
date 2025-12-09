"""
Configuration Management System

Supports YAML-based configuration for experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "data/processed/train_episodes.npz"
    val_path: Optional[str] = "data/processed/val_episodes.npz"
    test_path: Optional[str] = "data/processed/test_episodes.npz"
    episode_length: int = 30
    max_agents: int = 50
    input_dim: int = 10  # Extended features
    normalize: bool = True
    stats_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_dim: int = 256
    encoder_hidden: int = 128
    encoder_n_heads: int = 4
    encoder_n_layers: int = 2
    encoder_dropout: float = 0.1
    use_spatial_encoding: bool = True
    use_social_pooling: bool = True
    pooling_radius: float = 50.0

    dynamics_type: str = "gru"  # gru, lstm, transformer
    dynamics_hidden: int = 512
    dynamics_n_layers: int = 2
    dynamics_dropout: float = 0.1

    decoder_hidden: int = 128
    decoder_n_layers: int = 2
    decoder_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Learning rate scheduler
    scheduler_type: str = "cosine"  # cosine, step, plateau
    scheduler_eta_min: float = 1e-6
    scheduler_patience: int = 10

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Mixed precision training
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 or bfloat16

    # Distributed training
    use_ddp: bool = False
    world_size: int = 1
    rank: int = 0


@dataclass
class LossConfig:
    """Loss function configuration."""
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    existence_weight: float = 0.1
    contrastive_weight: float = 0.0
    huber_delta: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    context_length: int = 10
    rollout_length: int = 20
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10, 20])
    eval_frequency: int = 5  # Evaluate every N epochs
    save_visualizations: bool = True
    save_attention: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_frequency: int = 10
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "baseline"
    description: str = ""
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        # Extract top-level fields
        experiment_name = config_dict.get('experiment_name', 'baseline')
        description = config_dict.get('description', '')
        seed = config_dict.get('seed', 42)

        # Create sub-configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))

        return cls(
            experiment_name=experiment_name,
            description=description,
            seed=seed,
            data=data_config,
            model=model_config,
            training=training_config,
            loss=loss_config,
            evaluation=eval_config,
            logging=logging_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return asdict(self)

    def to_yaml(self, save_path: str):
        """
        Save configuration to YAML file.

        Args:
            save_path: Path to save YAML file
        """
        config_dict = self.to_dict()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        print(f"Configuration saved to {save_path}")

    def to_json(self, save_path: str):
        """
        Save configuration to JSON file.

        Args:
            save_path: Path to save JSON file
        """
        config_dict = self.to_dict()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {save_path}")

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Description: {self.description}",
            f"Seed: {self.seed}",
            "\nData:",
            f"  Train: {self.data.train_path}",
            f"  Val: {self.data.val_path}",
            f"  Input dim: {self.data.input_dim}",
            "\nModel:",
            f"  Latent dim: {self.model.latent_dim}",
            f"  Dynamics: {self.model.dynamics_type}",
            f"  Spatial encoding: {self.model.use_spatial_encoding}",
            f"  Social pooling: {self.model.use_social_pooling}",
            "\nTraining:",
            f"  Batch size: {self.training.batch_size}",
            f"  Epochs: {self.training.n_epochs}",
            f"  Learning rate: {self.training.learning_rate}",
            f"  Mixed precision: {self.training.use_amp}",
        ]
        return "\n".join(lines)


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from file (YAML or JSON).

    Args:
        config_path: Path to configuration file

    Returns:
        ExperimentConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix in ['.yaml', '.yml']:
        return ExperimentConfig.from_yaml(str(config_path))
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return ExperimentConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unsupported configuration format: {config_path.suffix}")


def create_default_config(save_path: str = "config.yaml"):
    """
    Create and save default configuration file.

    Args:
        save_path: Path to save configuration
    """
    config = ExperimentConfig()
    config.to_yaml(save_path)
    print(f"Default configuration created at {save_path}")


if __name__ == '__main__':
    # Create default configuration
    config = ExperimentConfig(
        experiment_name="improved_baseline",
        description="Improved world model with spatial encoding and social pooling"
    )

    # Save to YAML
    config.to_yaml("test_config.yaml")

    # Load back
    loaded_config = load_config("test_config.yaml")

    print("\nLoaded configuration:")
    print(loaded_config)

    # Test dictionary conversion
    config_dict = config.to_dict()
    print(f"\nConfiguration has {len(config_dict)} top-level keys")
