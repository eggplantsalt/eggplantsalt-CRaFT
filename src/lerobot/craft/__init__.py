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
CRaFT (Constrained Retention Fine-Tuning) Training Framework

This package implements the CRaFT training methodology for continual learning
with retention constraints. It provides gradient surgery, primal-dual optimization,
and anchor-based retention loss computation.
"""

from lerobot.craft.craft_config import CraftConfig

__all__ = ["CraftConfig"]

