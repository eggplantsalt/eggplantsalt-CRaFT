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
Gradient Surgery Module

Implements gradient projection and conflict resolution for multi-objective optimization.
Based on PCGrad and related gradient surgery techniques.
"""

import torch
from torch import Tensor


def compute_dot(grad1: list[Tensor], grad2: list[Tensor]) -> Tensor:
    """
    Compute the dot product (cosine similarity direction) between two gradient vectors.
    
    Args:
        grad1: List of gradient tensors for objective 1 (one per parameter)
        grad2: List of gradient tensors for objective 2 (one per parameter)
    
    Returns:
        Scalar tensor representing the dot product of flattened gradients.
        Positive value indicates alignment, negative indicates conflict.
    
    TODO: Implement gradient flattening and dot product computation.
    """
    raise NotImplementedError("compute_dot: to be implemented in next phase")


def project_if_conflict(
    grad_task: list[Tensor],
    grad_retain: list[Tensor],
    conflict_threshold: float = -0.1,
) -> tuple[list[Tensor], list[Tensor], bool]:
    """
    Project gradients if they are in conflict (negative dot product).
    
    Implements PCGrad-style projection:
    - If dot(g_task, g_retain) < threshold: project g_task onto normal plane of g_retain
    - Otherwise: keep original gradients
    
    Args:
        grad_task: Gradients from task loss (new data)
        grad_retain: Gradients from retention loss (anchor data)
        conflict_threshold: Threshold for detecting conflict (typically < 0)
    
    Returns:
        Tuple of (projected_grad_task, projected_grad_retain, conflict_detected)
    
    TODO: Implement gradient projection logic.
    """
    raise NotImplementedError("project_if_conflict: to be implemented in next phase")


def merge_grads(
    grad_task: list[Tensor],
    grad_retain: list[Tensor],
    lambda_weight: float,
    mode: str = "weighted",
) -> list[Tensor]:
    """
    Merge task and retention gradients according to specified mode.
    
    Modes:
    - "weighted": g_final = g_task + λ * g_retain
    - "equal": g_final = 0.5 * (g_task + g_retain)
    - "task_priority": g_final = g_task + min(λ, 1.0) * g_retain
    
    Args:
        grad_task: Gradients from task loss
        grad_retain: Gradients from retention loss
        lambda_weight: Current Lagrangian multiplier value
        mode: Merging strategy
    
    Returns:
        List of merged gradient tensors
    
    TODO: Implement gradient merging strategies.
    """
    raise NotImplementedError("merge_grads: to be implemented in next phase")

