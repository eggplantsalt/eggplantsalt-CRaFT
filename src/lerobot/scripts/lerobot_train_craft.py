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
CRaFT Training Script

This script implements Constrained Retention Fine-Tuning (CRaFT) for continual learning.
It extends the baseline lerobot_train.py with dual-objective optimization:
- Task loss: standard supervised learning on new task data
- Retention loss: performance preservation on anchor/old task data

Key differences from baseline training:
1. Loads anchor dataset for retention loss computation
2. Performs two backward passes per step (task + retention)
3. Applies gradient surgery (projection) when gradients conflict
4. Uses primal-dual optimization to dynamically adjust loss weights via λ
5. Anneals retention constraint ε over training

Current status: DRY-RUN SCAFFOLD
- Can load policy and run forward pass on one batch
- CRaFT-specific logic (dual backward, projection, λ update) marked as TODO
"""

import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.craft import CraftConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def update_policy_craft(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    task_batch: Any,
    anchor_batch: Any | None,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    craft_config: CraftConfig,
    current_lambda: float,
    current_epsilon: float,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict, float]:
    """
    Performs a single CRaFT training step with dual-objective optimization.
    
    CRaFT training loop (per step):
    1. Forward pass on task batch → L_task
    2. Backward pass on L_task → ∇L_task
    3. Forward pass on anchor batch → L_retain
    4. Backward pass on L_retain → ∇L_retain
    5. Gradient surgery: project if conflict detected
    6. Merge gradients: g_final = ∇L_task + λ * ∇L_retain
    7. Optimizer step with g_final
    8. Update λ based on constraint violation: λ ← λ + λ_lr * (L_retain - ε)
    
    Args:
        train_metrics: MetricsTracker for logging
        policy: Policy model to train
        task_batch: Batch from new task dataset
        anchor_batch: Batch from anchor/retention dataset (can be None for dry-run)
        optimizer: Optimizer instance
        grad_clip_norm: Gradient clipping norm
        accelerator: Accelerator for distributed training
        craft_config: CRaFT configuration
        current_lambda: Current Lagrangian multiplier value
        current_epsilon: Current retention loss threshold
        lr_scheduler: Optional learning rate scheduler
        lock: Optional lock for thread-safe updates
    
    Returns:
        Tuple of (updated_metrics, output_dict, updated_lambda)
    
    TODO: Implement full CRaFT training logic in next phase.
    Current implementation: baseline single-objective training (task loss only).
    """
    start_time = time.perf_counter()
    policy.train()
    
    # ============================================================================
    # PHASE 1: Task Loss (Standard Training)
    # ============================================================================
    with accelerator.autocast():
        task_loss, output_dict = policy.forward(task_batch)
    
    # Backward pass for task loss
    accelerator.backward(task_loss)
    
    # ============================================================================
    # TODO: PHASE 2: Retention Loss (CRaFT-specific)
    # ============================================================================
    # if anchor_batch is not None:
    #     # 1. Compute retention loss
    #     from lerobot.craft.retention_loss import compute_retention_loss
    #     with accelerator.autocast():
    #         retention_loss = compute_retention_loss(policy, anchor_batch)
    #     
    #     # 2. Backward pass for retention loss (accumulate gradients)
    #     accelerator.backward(retention_loss)
    #     
    #     # 3. Extract gradients for both objectives
    #     task_grads = [p.grad.clone() for p in policy.parameters() if p.grad is not None]
    #     # Note: Need to separate task and retention gradients properly
    #     
    #     # 4. Gradient surgery (projection if conflict)
    #     if craft_config.use_grad_projection:
    #         from lerobot.craft.grad_surgery import project_if_conflict, merge_grads
    #         task_grads_proj, retain_grads_proj, conflict = project_if_conflict(
    #             task_grads, retention_grads, craft_config.conflict_threshold
    #         )
    #         final_grads = merge_grads(
    #             task_grads_proj, retain_grads_proj, current_lambda, craft_config.projection_mode
    #         )
    #     else:
    #         # Simple weighted combination
    #         final_grads = [g_t + current_lambda * g_r for g_t, g_r in zip(task_grads, retention_grads)]
    #     
    #     # 5. Replace gradients with merged gradients
    #     for param, grad in zip(policy.parameters(), final_grads):
    #         param.grad = grad
    #     
    #     # 6. Update lambda (primal-dual)
    #     from lerobot.craft.primal_dual import update_lambda
    #     current_lambda = update_lambda(
    #         current_lambda,
    #         retention_loss.item(),
    #         current_epsilon,
    #         craft_config.lambda_lr,
    #         craft_config.lambda_max,
    #     )
    #     
    #     # Log retention metrics
    #     output_dict["retention_loss"] = retention_loss.item()
    #     output_dict["lambda"] = current_lambda
    #     output_dict["epsilon"] = current_epsilon
    
    # ============================================================================
    # Gradient Clipping and Optimizer Step (Standard)
    # ============================================================================
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )
    
    with lock if lock is not None else nullcontext():
        optimizer.step()
    
    optimizer.zero_grad()
    
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
    
    # Update metrics
    train_metrics.loss = task_loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    
    return train_metrics, output_dict, current_lambda


@parser.wrap()
def train_craft(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function for CRaFT training.
    
    This function extends the baseline train() with CRaFT-specific components:
    - Loads anchor dataset for retention loss
    - Initializes CRaFT config (λ, ε schedule, etc.)
    - Calls update_policy_craft() instead of update_policy()
    
    Current status: DRY-RUN mode
    - Runs 1 batch forward pass and exits
    - Validates that policy loading and data pipeline work correctly
    """
    cfg.validate()
    
    # Create Accelerator if not provided
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )
    
    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        logging.info(colored("=" * 80, "cyan"))
        logging.info(colored("CRaFT Training (DRY-RUN MODE)", "cyan", attrs=["bold"]))
        logging.info(colored("=" * 80, "cyan"))
        logging.info(pformat(cfg.to_dict()))
    
    # Initialize wandb
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # ============================================================================
    # Dataset Loading (Task Dataset)
    # ============================================================================
    if is_main_process:
        logging.info("Creating task dataset")
        dataset = make_dataset(cfg)
    
    accelerator.wait_for_everyone()
    
    if not is_main_process:
        dataset = make_dataset(cfg)
    
    # ============================================================================
    # TODO: Load Anchor Dataset (CRaFT-specific)
    # ============================================================================
    # Initialize CRaFT config (placeholder for now)
    craft_config = CraftConfig()
    
    # TODO: Load anchor dataset in next phase
    # if craft_config.anchor_dataset_path:
    #     from lerobot.craft.anchor_cache import create_anchor_dataloader
    #     anchor_dataloader = create_anchor_dataloader(
    #         craft_config.anchor_dataset_path,
    #         craft_config.anchor_batch_size,
    #         num_workers=cfg.num_workers,
    #     )
    #     anchor_dl_iter = cycle(anchor_dataloader)
    # else:
    #     anchor_dl_iter = None
    anchor_dl_iter = None  # Placeholder for dry-run
    
    # ============================================================================
    # Environment (for evaluation)
    # ============================================================================
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    # ============================================================================
    # Policy Creation
    # ============================================================================
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    
    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
    
    accelerator.wait_for_everyone()
    
    # ============================================================================
    # Processors
    # ============================================================================
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    
    # ============================================================================
    # Optimizer and Scheduler
    # ============================================================================
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    
    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # ============================================================================
    # DataLoader
    # ============================================================================
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    
    # Prepare with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    
    policy.train()
    
    # ============================================================================
    # Metrics Tracking
    # ============================================================================
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )
    
    # ============================================================================
    # CRaFT State Initialization
    # ============================================================================
    current_lambda = craft_config.initial_lambda
    current_epsilon = craft_config.epsilon_start
    
    if is_main_process:
        logging.info(colored("=" * 80, "green"))
        logging.info(colored("DRY-RUN: Running 1 batch forward pass", "green", attrs=["bold"]))
        logging.info(colored("=" * 80, "green"))
        logging.info(f"Initial λ: {current_lambda}")
        logging.info(f"Initial ε: {current_epsilon}")
    
    # ============================================================================
    # DRY-RUN: Single Batch Forward Pass
    # ============================================================================
    start_time = time.perf_counter()
    batch = next(dl_iter)
    batch = preprocessor(batch)
    train_tracker.dataloading_s = time.perf_counter() - start_time
    
    if is_main_process:
        logging.info(f"Batch loaded in {train_tracker.dataloading_s:.3f}s")
        logging.info(f"Batch keys: {list(batch.keys())}")
    
    # Run one training step (dry-run mode: no anchor batch)
    train_tracker, output_dict, current_lambda = update_policy_craft(
        train_tracker,
        policy,
        batch,
        anchor_batch=None,  # No anchor data in dry-run
        optimizer=optimizer,
        grad_clip_norm=cfg.optimizer.grad_clip_norm,
        accelerator=accelerator,
        craft_config=craft_config,
        current_lambda=current_lambda,
        current_epsilon=current_epsilon,
        lr_scheduler=lr_scheduler,
    )
    
    step += 1
    train_tracker.step()
    
    if is_main_process:
        logging.info(colored("=" * 80, "green"))
        logging.info(colored("DRY-RUN SUCCESSFUL!", "green", attrs=["bold"]))
        logging.info(colored("=" * 80, "green"))
        logging.info(train_tracker)
        logging.info(f"Task loss: {output_dict.get('loss', 'N/A')}")
        logging.info(f"Updated λ: {current_lambda}")
        logging.info("")
        logging.info(colored("Next steps:", "yellow", attrs=["bold"]))
        logging.info("  1. Implement anchor dataset loading (anchor_cache.py)")
        logging.info("  2. Implement retention loss computation (retention_loss.py)")
        logging.info("  3. Implement gradient surgery (grad_surgery.py)")
        logging.info("  4. Implement primal-dual updates (primal_dual.py)")
        logging.info("  5. Uncomment TODO sections in update_policy_craft()")
    
    if eval_env:
        close_envs(eval_env)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    if is_main_process:
        logging.info("End of CRaFT dry-run")


def main():
    register_third_party_plugins()
    train_craft()


if __name__ == "__main__":
    main()

