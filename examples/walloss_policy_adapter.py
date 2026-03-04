"""
Walloss Policy Adapter for DSRL Training.

This module provides an adapter class that wraps the Walloss model
to provide a π0-compatible interface for use in DSRL training loops.

Key differences between π0 and Walloss:
- π0 is a JAX model with `infer(obs, noise)` interface
- Walloss is a PyTorch model requiring complex batch format input

This adapter handles:
1. Input format conversion: π0 obs format -> Walloss batch format
2. Model inference with proper device management
3. Output format conversion: Walloss output -> π0 compatible format
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import yaml

import torch
from transformers import AutoProcessor

# Import Walloss components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wall-x"))

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.utils import load_norm_stats, preprocesser_call
from wall_x.serving.policy.utils import process_images, format_text_with_vision_tokens
from qwen_vl_utils.vision_process import smart_resize
from transformers import BatchFeature

from examples.walloss_config import WallossModelConfig

logger = logging.getLogger(__name__)


class WallossPolicyAdapter:
    """Adapter that provides π0-compatible interface for Walloss model.

    This adapter wraps the Walloss PyTorch model and provides an `infer()`
    method that accepts π0-style observations and noise, returning actions
    in the expected format.

    Example:
        >>> adapter = WallossPolicyAdapter(config)
        >>> result = adapter.infer(obs_pi_zero, noise=noise)
        >>> actions = result["actions"]  # Shape: (action_horizon, action_dim)
    """

    def __init__(self, config: WallossModelConfig):
        """Initialize the Walloss policy adapter.

        Args:
            config: WallossModelConfig containing model paths and settings
        """
        self.config = config

        logger.info(f"Loading Walloss model from {config.model_path}")

        # Load train_config from YAML file if path provided, otherwise use inline config
        if config.train_config_path and os.path.exists(config.train_config_path):
            logger.info(f"Loading train_config from {config.train_config_path}")
            with open(config.train_config_path, "r") as f:
                train_config = yaml.safe_load(f)
        elif config.train_config:
            train_config = config.train_config
        else:
            # Create minimal train_config for inference
            train_config = self._create_default_train_config(config)

        # Store train_config for later use
        self.train_config = train_config

        # Get dataset_name from train_config (for load_norm_stats)
        if "data" in train_config and "lerobot_config" in train_config["data"]:
            self.dataset_name = train_config["data"]["lerobot_config"]["repo_id"]
        else:
            self.dataset_name = config.dataset_name

        # Load model
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            config.model_path,
            train_config=train_config,
            action_tokenizer_path=config.action_tokenizer_path,
        )
        self.model.eval()

        # Set dtype
        if config.dtype == "bfloat16":
            self.model = self.model.bfloat16()
        elif config.dtype == "float16":
            self.model = self.model.half()

        self.model = self.model.to(config.device)

        # Fixed action dim for internal padding (Walloss uses 20)
        self.fixed_action_dim = 20

        # Store dimensions
        self.action_dim = config.action_dim
        self.agent_pos_dim = config.agent_pos_dim
        self.pred_horizon = config.action_horizon
        self.device = config.device
        self.predict_mode = config.predict_mode
        self.camera_keys = config.camera_keys

        # Image preprocessing config
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        self.image_factor = config.image_factor
        self.max_length = config.max_length

        # Load processor
        logger.info("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(
            config.processor_path, use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"

        # Load normalization statistics
        if config.norm_stats_path and os.path.exists(config.norm_stats_path):
            logger.info(f"Loading normalization stats from {config.norm_stats_path}")
            logger.info(f"Using dataset_name: {self.dataset_name}")
            self.norm_stats = load_norm_stats(config.norm_stats_path, self.dataset_name)
        else:
            logger.warning("No normalization stats found, using identity normalization")
            self.norm_stats = None

        logger.info(
            f"Walloss model loaded. Device: {config.device}, "
            f"Action dim: {config.action_dim}, Horizon: {config.action_horizon}"
        )

    def infer(self, obs: Dict, noise=None) -> Dict:
        """Infer actions from observation.

        This method provides π0-compatible interface for DSRL training.
        It converts π0 observation format to Walloss batch format,
        runs inference, and returns actions in expected format.

        Args:
            obs: Observation dictionary in π0 format:
                - "observation/image": Main camera image (H, W, 3) uint8
                - "observation/wrist_image": Wrist camera image (H, W, 3) uint8 (optional)
                - "observation/state": Robot state (state_dim,) float32
                - "prompt": Task instruction text
            noise: Noise tensor for diffusion mode (ignored in fast mode)
                   Shape: (1, noise_steps, latent_dim) or similar

        Returns:
            Dictionary containing:
                - "actions": numpy array of shape (action_horizon, action_dim)
        """
        try:
            # Convert π0 observation format to Walloss format
            walloss_obs = self._convert_obs_format(obs)

            # Prepare batch for Walloss model
            input_batch = self._prepare_batch(walloss_obs)

            # Convert noise to torch tensor if provided (for diffusion mode)
            initial_noise = None
            if noise is not None and self.predict_mode == "diffusion":
                if isinstance(noise, np.ndarray):
                    initial_noise = torch.from_numpy(noise.copy()).float()
                elif isinstance(noise, torch.Tensor):
                    initial_noise = noise.float()
                elif hasattr(noise, '__array__'):  # JAX array or similar
                    # Convert JAX array to numpy then to torch
                    initial_noise = torch.from_numpy(np.array(noise)).float()
                else:
                    logger.warning(f"Unsupported noise type: {type(noise)}, ignoring")

                # Reshape noise to match expected dimensions
                # DSRL passes noise with shape (1, 50, action_dim) or similar
                # Walloss expects (batch_size, pred_horizon, fixed_action_dim) where fixed_action_dim=20
                if initial_noise is not None:
                    # Ensure 3D tensor
                    if initial_noise.dim() == 2:
                        initial_noise = initial_noise.unsqueeze(0)

                    # Trim or pad to match pred_horizon
                    current_horizon = initial_noise.shape[1]
                    if current_horizon > self.pred_horizon:
                        # Trim to pred_horizon
                        initial_noise = initial_noise[:, :self.pred_horizon, :]
                    elif current_horizon < self.pred_horizon:
                        # Pad by repeating the last frame
                        padding = initial_noise[:, -1:, :].repeat(1, self.pred_horizon - current_horizon, 1)
                        initial_noise = torch.cat([initial_noise, padding], dim=1)

                    # Ensure last dim matches fixed_action_dim (20)
                    # If > 20: trim, if < 20: repeat to fill
                    current_dim = initial_noise.shape[2]
                    if current_dim > self.fixed_action_dim:
                        initial_noise = initial_noise[:, :, :self.fixed_action_dim]
                    elif current_dim < self.fixed_action_dim:
                        # Repeat the values to fill the dimension
                        repeat_count = self.fixed_action_dim // current_dim
                        remainder = self.fixed_action_dim % current_dim
                        initial_noise = initial_noise.repeat(1, 1, repeat_count)
                        if remainder > 0:
                            initial_noise = torch.cat([initial_noise, initial_noise[:, :, :remainder]], dim=2)

                    logger.debug(f"Noise reshaped from shape {noise.shape if hasattr(noise, 'shape') else 'unknown'} to {initial_noise.shape}")

            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    **input_batch,
                    action_dim=(
                        self.action_dim
                        if self.predict_mode == "fast"
                        else self.fixed_action_dim
                    ),
                    pred_horizon=self.pred_horizon,
                    mode="predict",
                    predict_mode=self.predict_mode,
                    initial_noise=initial_noise,
                )

            # Extract and format actions
            if outputs["predict_action"] is None:
                predicted_actions = np.zeros(
                    [self.pred_horizon, self.action_dim], dtype=np.float32
                )
            else:
                predicted_actions = (
                    outputs["predict_action"][0, :, : self.action_dim]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )

            return {"actions": predicted_actions}

        except Exception as e:
            logger.error(f"Error during Walloss inference: {e}")
            raise

    def _convert_obs_format(self, pi0_obs: Dict) -> Dict:
        """Convert π0 observation format to Walloss observation format.

        π0 format (Libero):
            - "observation/image": Main camera image
            - "observation/wrist_image": Wrist camera image
            - "observation/state": Robot state
            - "prompt": Task instruction

        Walloss format:
            - Camera keys (e.g., "face_view", "left_wrist_view"): Images
            - "state": Robot state
            - "prompt": Task instruction
            - "dataset_names": Dataset identifier
        """
        walloss_obs = {}

        # Handle images based on environment
        if "observation/image" in pi0_obs:
            # Libero-style observation
            walloss_obs["face_view"] = pi0_obs["observation/image"]
            if "observation/wrist_image" in pi0_obs:
                walloss_obs["left_wrist_view"] = pi0_obs["observation/wrist_image"]
        elif "images" in pi0_obs:
            # Aloha-style observation
            images_dict = pi0_obs["images"]
            if "cam_high" in images_dict:
                walloss_obs["face_view"] = self._transpose_image(images_dict["cam_high"])
        else:
            # Try direct camera keys
            for key in self.camera_keys:
                if key in pi0_obs:
                    walloss_obs[key] = pi0_obs[key]

        # Handle state
        if "observation/state" in pi0_obs:
            walloss_obs["state"] = pi0_obs["observation/state"]
        elif "state" in pi0_obs:
            walloss_obs["state"] = pi0_obs["state"]

        # Handle prompt
        if "prompt" in pi0_obs:
            walloss_obs["prompt"] = pi0_obs["prompt"]
        else:
            walloss_obs["prompt"] = "Perform the task"

        # Add dataset name
        walloss_obs["dataset_names"] = self.dataset_name

        return walloss_obs

    def _transpose_image(self, img: np.ndarray) -> np.ndarray:
        """Transpose image from (C, H, W) to (H, W, C) if needed."""
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            return np.transpose(img, (1, 2, 0))
        return img

    def _prepare_batch(self, obs: Dict) -> BatchFeature:
        """Prepare observation into Walloss model input format.

        This follows the same logic as wall_x.serving.policy.utils.prepare_batch
        but simplified for single-inference use case.
        """
        # Extract and process images
        images = []
        for key in self.camera_keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    # Handle dimensions
                    if img.ndim > 3:
                        img = np.squeeze(img)
                    if img.ndim == 3 and img.shape[0] in [1, 3]:
                        img = np.transpose(img, (1, 2, 0))
                    # Convert to PIL Image
                    from PIL import Image
                    if img.dtype == np.uint8:
                        img = Image.fromarray(img)
                    else:
                        img = Image.fromarray((img * 255).astype(np.uint8))
                images.append(img)

        # Apply smart resize to images
        resized_images = process_images(
            images, self.image_factor, self.min_pixels, self.max_pixels
        )

        # Format text with vision tokens
        formatted_text = format_text_with_vision_tokens(
            obs["prompt"],
            self.camera_keys[:len(resized_images)],
            self.predict_mode,
            self.pred_horizon,
        )

        # Use processor to prepare inputs
        inputs = preprocesser_call(
            processor=self.processor,
            text=[formatted_text],
            images=[resized_images],
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Add moe_token_types
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        moe_token_types = inputs.input_ids == action_token_id
        inputs["moe_token_types"] = moe_token_types

        # Handle robot state/proprioception
        if "state" in obs:
            state = obs["state"]
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state.copy()).float()
            elif not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            # Normalize state if stats available
            if self.norm_stats is not None:
                state = self._normalize_state(state)

            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.dim() == 2:
                state = state.unsqueeze(1)  # [batch, 1, state_dim]

            # Pad to 20 dimensions if needed
            if state.shape[-1] < self.fixed_action_dim:
                padding = torch.zeros(
                    state.shape[0], state.shape[1],
                    self.fixed_action_dim - state.shape[-1]
                )
                state = torch.cat([state, padding], dim=-1)

            # Create mask for valid dimensions
            agent_pos_mask = torch.ones_like(state)
            if state.shape[-1] > self.agent_pos_dim:
                agent_pos_mask[:, :, self.agent_pos_dim:] = 0

            inputs["proprioception"] = state
            inputs["agent_pos_mask"] = agent_pos_mask
        else:
            # Create dummy state if not provided
            batch_size = inputs["input_ids"].shape[0]
            inputs["proprioception"] = torch.zeros(
                batch_size, 1, self.fixed_action_dim
            )
            inputs["agent_pos_mask"] = torch.zeros(
                batch_size, 1, self.fixed_action_dim
            )
            if self.agent_pos_dim > 0:
                inputs["agent_pos_mask"][:, :, :self.agent_pos_dim] = 1

        # Add dataset name (must be a list for the model's unnormalize_data function)
        inputs["dataset_names"] = [obs.get("dataset_names", self.dataset_name)]

        # Add dof_mask
        dof_mask = torch.ones([inputs["input_ids"].shape[0], self.pred_horizon, self.fixed_action_dim])
        dof_mask[:, :, self.action_dim:] = 0
        inputs["dof_mask"] = dof_mask

        # Move all tensors to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        return BatchFeature(data=dict(inputs)).to(self.device)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using loaded statistics."""
        if self.norm_stats is None:
            return state

        stats = self.norm_stats["state"]
        # Normalize to [-1, 1] range
        normalized = 2 * (state - stats.min) / (stats.delta + 1e-8) - 1
        return torch.clamp(normalized, -1, 1)

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to original range."""
        if self.norm_stats is None:
            return action

        stats = self.norm_stats["action"]
        # Denormalize from [-1, 1] to original range
        denormalized = (action + 1) / 2 * stats.delta + stats.min
        return denormalized

    def _create_default_train_config(self, config: WallossModelConfig) -> Dict:
        """Create a minimal train_config for model loading.

        This is used when no train_config_path or train_config is provided.
        """
        # Determine dataset name for lerobot_config
        dataset_name = config.dataset_name
        if dataset_name == "physical-intelligence/libero":
            lerobot_repo_id = "lerobot/libero_goal_image"
        elif dataset_name == "lerobot/aloha_mobile_cabinet":
            lerobot_repo_id = "lerobot/aloha_mobile_cabinet"
        else:
            lerobot_repo_id = dataset_name

        return {
            "data": {
                "use_lerobot": True,
                "lerobot_config": {
                    "repo_id": lerobot_repo_id,
                    "root": None,
                    "episodes": None,
                    "image_transforms": None,
                    "delta_timestamps": None,
                    "tolerance_s": 1e-4,
                    "revision": None,
                    "force_cache_sync": False,
                    "download_videos": True,
                    "video_backend": None,
                },
                "action_horizon": config.action_horizon,
            },
            "enable_customized_robot_config": True,
            "customized_robot_config": {
                "name": lerobot_repo_id,
                "customized_dof_config": {
                    "action_eef": 6,
                    "action_gripper": 1,
                },
                "customized_agent_pos_config": {
                    "state_eef_with_gripper": 8,
                },
            },
            "norm_stats_path": config.norm_stats_path,
        }
