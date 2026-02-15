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
Retention Loss Module

Computes retention loss on anchor/old task data to measure performance preservation.
"""

import torch
from torch import Tensor


def compute_retention_loss(
    policy,
    anchor_batch: dict,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute retention loss on anchor data batch.
    
    The retention loss measures how well the model preserves performance on
    previously learned tasks/data. It uses the same loss function as the task loss
    but is computed on anchor (old) data.
    
    Args:
        policy: The policy model (must have forward method that returns loss)
        anchor_batch: Batch of anchor data (same format as training batch)
        reduction: Loss reduction mode ("mean", "sum", "none")
    
    Returns:
        Retention loss tensor
    
    TODO: Implement retention loss computation by calling policy.forward on anchor batch.
    This should be straightforward as it reuses the policy's existing loss computation.
    """
    raise NotImplementedError("compute_retention_loss: to be implemented in next phase")

