#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CRaFT Configuration Module

Defines the configuration dataclass for CRaFT training parameters.
"""

from dataclasses import dataclass, field


@dataclass
class CraftConfig:
    """
    Configuration for CRaFT (Constrained Retention Fine-Tuning) training.
    
    This config controls the dual-objective optimization with retention constraints:
    - Task loss (L_task): standard supervised learning on new task data
    - Retention loss (L_retain): performance preservation on anchor/old task data
    - Primal-dual optimization: dynamically balance the two objectives via Lagrangian multiplier λ
    
    Attributes:
        # Anchor dataset configuration
        anchor_dataset_path: Path to the anchor/retention dataset (HuggingFace dataset or local path)
        anchor_batch_size: Batch size for anchor data sampling
        anchor_sample_ratio: Ratio of anchor samples per training step (0.0-1.0)
        
        # Loss weighting
        initial_lambda: Initial value for Lagrangian multiplier λ (retention loss weight)
        lambda_lr: Learning rate for λ updates in primal-dual optimization
        lambda_max: Maximum allowed value for λ (prevents unbounded growth)
        
        # Retention constraint
        epsilon_start: Initial retention loss threshold ε (relaxed at start)
        epsilon_end: Final retention loss threshold ε (tightened over training)
        epsilon_decay_steps: Number of steps to anneal ε from start to end
        
        # Gradient surgery
        use_grad_projection: Whether to apply gradient projection when task/retention gradients conflict
        conflict_threshold: Cosine similarity threshold to detect gradient conflict (typically < 0)
        projection_mode: How to merge gradients after projection ("weighted", "equal", "task_priority")
        
        # Logging and debugging
        log_craft_metrics_freq: Frequency (in steps) to log CRaFT-specific metrics
        save_lambda_history: Whether to save λ trajectory for analysis
    """
    
    # Anchor dataset configuration
    anchor_dataset_path: str = ""
    anchor_batch_size: int = 16
    anchor_sample_ratio: float = 0.5  # Sample anchor data 50% of the time
    
    # Loss weighting (primal-dual)
    initial_lambda: float = 1.0
    lambda_lr: float = 0.01
    lambda_max: float = 10.0
    
    # Retention constraint (epsilon schedule)
    epsilon_start: float = 1.0  # Relaxed constraint at start
    epsilon_end: float = 0.1    # Tight constraint at end
    epsilon_decay_steps: int = 10000
    
    # Gradient surgery
    use_grad_projection: bool = True
    conflict_threshold: float = -0.1  # Negative cosine similarity indicates conflict
    projection_mode: str = "weighted"  # Options: "weighted", "equal", "task_priority"
    
    # Logging
    log_craft_metrics_freq: int = 100
    save_lambda_history: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.anchor_sample_ratio <= 1.0:
            raise ValueError(f"anchor_sample_ratio must be in [0, 1], got {self.anchor_sample_ratio}")
        
        if self.initial_lambda < 0:
            raise ValueError(f"initial_lambda must be non-negative, got {self.initial_lambda}")
        
        if self.lambda_max <= 0:
            raise ValueError(f"lambda_max must be positive, got {self.lambda_max}")
        
        if self.epsilon_start < self.epsilon_end:
            raise ValueError(
                f"epsilon_start ({self.epsilon_start}) must be >= epsilon_end ({self.epsilon_end})"
            )
        
        if self.projection_mode not in ["weighted", "equal", "task_priority"]:
            raise ValueError(
                f"projection_mode must be one of ['weighted', 'equal', 'task_priority'], "
                f"got {self.projection_mode}"
            )

