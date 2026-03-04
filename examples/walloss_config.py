"""
Configuration module for Walloss model integration with DSRL training.

This module provides configuration dataclasses and factory functions
for creating WallossPolicyAdapter instances.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os


@dataclass
class WallossModelConfig:
    """Configuration for Walloss model loading and inference.

    Attributes:
        model_path: Path to the pretrained Walloss model checkpoint
        processor_path: Path to the processor (tokenizer/image processor)
        action_tokenizer_path: Path to action tokenizer (optional, for diffusion mode)
        norm_stats_path: Path to normalization statistics JSON file
        train_config_path: Path to YAML train config file (recommended)
        predict_mode: Prediction mode - "fast" or "diffusion"
        action_horizon: Number of action steps to predict (query_freq in DSRL)
        action_dim: Dimension of action space
        agent_pos_dim: Dimension of proprioceptive state
        dataset_name: Dataset name for normalization stats lookup
        camera_keys: List of camera keys to use for observation
        device: Device to run model on
        dtype: Data type for model weights
        max_length: Maximum sequence length for text
        min_pixels: Minimum pixels for image resizing
        max_pixels: Maximum pixels for image resizing
        image_factor: Factor for smart resize
        train_config: Inline train config dict (used if train_config_path not provided)
    """
    model_path: str
    processor_path: str
    norm_stats_path: str
    train_config_path: Optional[str] = None
    predict_mode: str = "fast"
    action_horizon: int = 32
    action_dim: int = 7
    agent_pos_dim: int = 8
    dataset_name: str = "physical-intelligence/libero"
    camera_keys: List[str] = field(default_factory=lambda: ["face_view", "left_wrist_view"])
    action_tokenizer_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_length: int = 768
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 16384 * 28 * 28
    image_factor: int = 28
    train_config: Optional[Dict[str, Any]] = None


# Default configurations for different environments
DEFAULT_LIBERO_CONFIG = WallossModelConfig(
    model_path="/data/disk0/Models/wall-x-libero",
    processor_path="/data/disk0/Models/wall-x-libero",
    norm_stats_path="/data/disk0/Models/wall-x-libero/norm_stats.json",
    predict_mode="fast",
    action_horizon=32,
    action_dim=7,
    agent_pos_dim=8,
    dataset_name="physical-intelligence/libero",
    camera_keys=["face_view", "left_wrist_view"],
)

DEFAULT_ALOHA_CONFIG = WallossModelConfig(
    model_path="/data/disk0/Models/wall-x-aloha",
    processor_path="/data/disk0/Models/wall-x-aloha",
    norm_stats_path="/data/disk0/Models/wall-x-aloha/norm_stats.json",
    predict_mode="fast",
    action_horizon=32,
    action_dim=14,
    agent_pos_dim=14,
    dataset_name="lerobot/aloha_mobile_cabinet",
    camera_keys=["face_view", "left_wrist_view", "right_wrist_view"],
)


def create_walloss_adapter(
    config_path: Optional[str] = None,
    env: str = "libero",
    **kwargs
) -> "WallossPolicyAdapter":
    """Create a WallossPolicyAdapter instance.

    Args:
        config_path: Optional path to YAML config file (not implemented yet)
        env: Environment name ("libero" or "aloha_cube") for default config
        **kwargs: Override default config values

    Returns:
        WallossPolicyAdapter instance configured for the specified environment
    """
    from examples.walloss_policy_adapter import WallossPolicyAdapter

    # Select default config based on environment
    if env == "libero":
        config = DEFAULT_LIBERO_CONFIG
    elif env == "aloha_cube":
        config = DEFAULT_ALOHA_CONFIG
    else:
        raise ValueError(f"Unknown environment: {env}. Supported: 'libero', 'aloha_cube'")

    # Override with kwargs
    config_dict = {
        k: kwargs.get(k, getattr(config, k))
        for k in [
            "model_path", "processor_path", "norm_stats_path", "train_config_path",
            "predict_mode", "action_horizon", "action_dim", "agent_pos_dim",
            "dataset_name", "camera_keys", "action_tokenizer_path", "device", "dtype",
            "max_length", "min_pixels", "max_pixels", "image_factor", "train_config"
        ]
    }

    new_config = WallossModelConfig(**config_dict)

    return WallossPolicyAdapter(new_config)
