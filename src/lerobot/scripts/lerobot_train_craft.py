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
    执行单步 CRaFT 训练（双目标优化，支持 Hidden State Anchoring）
    
    【CRaFT 训练流程】
    1. 前向传播（任务数据）→ L_task
    2. 反向传播 L_task → ∇L_task（保存梯度）
    3. 前向传播（锚点数据）→ 提取 student hidden states
    4. 计算 L_retain（hidden state loss）
    5. 反向传播 L_retain → ∇L_retain（保存梯度）
    6. 梯度手术：检测冲突并投影
    7. 合并梯度：g_final = ∇L_task + λ * ∇L_retain
    8. 优化器更新（使用 g_final）
    9. 更新 λ：λ ← λ + λ_lr * (L_retain - ε)
    
    【Hidden State Anchoring】
    - 不再使用 teacher tokens/labels
    - 改用 teacher hidden states（从 AnchorCache 加载）
    - Student 提取相同层的 hidden states 并 pooling
    - 计算 MSE/Cosine loss
    
    【参数】
    train_metrics: 训练指标跟踪器
    policy: 策略模型
    task_batch: 任务数据批次
    anchor_batch: 锚点数据批次（可为 None，表示不计算保留损失）
    optimizer: 优化器
    grad_clip_norm: 梯度裁剪范数
    accelerator: 分布式训练加速器
    craft_config: CRaFT 配置
    current_lambda: 当前 λ 值
    current_epsilon: 当前 ε 阈值
    lr_scheduler: 学习率调度器（可选）
    lock: 线程锁（可选）
    
    【返回值】
    (更新后的指标, 输出字典, 更新后的 λ)
    """
    from lerobot.craft.grad_surgery import compute_dot, project_if_conflict, merge_grads
    from lerobot.craft.primal_dual import update_lambda
    from lerobot.craft.retention_loss import compute_retention_loss_hidden, extract_student_hidden_with_pooling
    
    start_time = time.perf_counter()
    policy.train()
    
    # ============================================================================
    # PHASE 1: 任务损失（标准训练）
    # ============================================================================
    with accelerator.autocast():
        task_loss, output_dict = policy.forward(task_batch)
    
    # 第一次反向传播：计算任务梯度
    accelerator.backward(task_loss)
    
    # 保存任务梯度（需要 clone，因为后续会清零）
    task_grads = [
        p.grad.clone() if p.grad is not None else None
        for p in policy.parameters()
    ]
    
    # ============================================================================
    # PHASE 2: 保留损失（CRaFT 特有 - Hidden State Anchoring）
    # ============================================================================
    retention_loss_value = None
    grad_conflict_detected = False
    grad_dot_product = None
    
    if anchor_batch is not None and craft_config.enabled:
        # 检测 cache 类型
        is_hidden_state_cache = "teacher_hidden" in anchor_batch
        
        if is_hidden_state_cache:
            # ========================================================================
            # Hidden State Anchoring（新版本）
            # ========================================================================
            # 清零梯度，准备第二次反向传播
            optimizer.zero_grad()
            
            # 提取 student hidden states
            meta = anchor_batch["meta"]
            layers_to_extract = meta["layers_to_save"]
            
            with accelerator.autocast():
                student_hidden = extract_student_hidden_with_pooling(
                    policy,
                    anchor_batch,
                    layers_to_extract,
                    meta,
                )
                
                # 计算 hidden state retention loss
                teacher_hidden = anchor_batch["teacher_hidden"].to(student_hidden.device)
                retention_loss = compute_retention_loss_hidden(
                    student_hidden,
                    teacher_hidden,
                    loss_type="mse",  # 可配置
                    reduction="mean",
                )
            
            # 第二次反向传播：计算保留梯度
            accelerator.backward(retention_loss)
            
        else:
            # ========================================================================
            # Token-level Distillation（旧版本，向后兼容）
            # ========================================================================
            optimizer.zero_grad()
            
            with accelerator.autocast():
                retention_loss, _ = policy.forward(anchor_batch)
            
            accelerator.backward(retention_loss)
        
        # 保存保留梯度
        retention_grads = [
            p.grad.clone() if p.grad is not None else None
            for p in policy.parameters()
        ]
        
        # 清零梯度，准备设置合并后的梯度
        optimizer.zero_grad()
        
        # ========================================================================
        # 梯度手术：冲突检测与投影
        # ========================================================================
        task_grads_filtered = [g for g in task_grads if g is not None]
        retention_grads_filtered = [g for g in retention_grads if g is not None]
        
        if len(task_grads_filtered) > 0 and len(retention_grads_filtered) > 0:
            # 计算梯度点积
            grad_dot_product = compute_dot(task_grads_filtered, retention_grads_filtered).item()
            
            # 如果启用梯度投影且检测到冲突
            if craft_config.use_grad_projection:
                task_grads_proj, retention_grads_proj, grad_conflict_detected = project_if_conflict(
                    task_grads,
                    retention_grads,
                    craft_config.conflict_threshold,
                )
            else:
                task_grads_proj = task_grads
                retention_grads_proj = retention_grads
            
            # 合并梯度
            final_grads = merge_grads(
                task_grads_proj,
                retention_grads_proj,
                current_lambda,
                craft_config.projection_mode,
            )
        else:
            final_grads = task_grads
        
        # 设置合并后的梯度到模型参数
        for param, grad in zip(policy.parameters(), final_grads):
            if grad is not None:
                param.grad = grad
        
        # ========================================================================
        # 原对偶优化：更新 λ
        # ========================================================================
        retention_loss_value = retention_loss.item()
        
        # 在分布式训练中，需要对 retention_loss 求平均
        if accelerator.num_processes > 1:
            retention_loss_tensor = torch.tensor(retention_loss_value, device=accelerator.device)
            retention_loss_tensor = accelerator.gather(retention_loss_tensor).mean()
            retention_loss_value = retention_loss_tensor.item()
        
        # 更新 λ
        current_lambda = update_lambda(
            current_lambda,
            retention_loss_value,
            current_epsilon,
            craft_config.lambda_lr,
            craft_config.lambda_max,
        )
        
        # 记录 CRaFT 特定指标
        output_dict["retention_loss"] = retention_loss_value
        output_dict["lambda"] = current_lambda
        output_dict["epsilon"] = current_epsilon
        output_dict["grad_conflict"] = 1.0 if grad_conflict_detected else 0.0
        output_dict["cache_type"] = "hidden_state" if is_hidden_state_cache else "token_level"
        if grad_dot_product is not None:
            output_dict["grad_dot"] = grad_dot_product
    else:
        # 没有锚点数据或未启用 CRaFT：使用任务梯度
        for param, grad in zip(policy.parameters(), task_grads):
            if grad is not None:
                param.grad = grad
    
    # ============================================================================
    # 梯度裁剪和优化器更新（标准流程）
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
    
    # ============================================================================
    # 更新指标
    # ============================================================================
    train_metrics.loss = task_loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    
    return train_metrics, output_dict, current_lambda


@parser.wrap()
def train_craft(
    cfg: TrainPipelineConfig,
    craft_config: CraftConfig | None = None,
    accelerator: Accelerator | None = None,
):
    """
    CRaFT 训练主函数
    
    【功能说明】
    扩展 baseline 训练，增加 CRaFT 特有组件：
    - 加载锚点数据集用于保留损失计算
    - 初始化 CRaFT 配置（λ、ε 调度等）
    - 使用 update_policy_craft() 替代标准 update_policy()
    - 实现 K-step 策略（每 K 步计算一次保留损失）
    
    【参数】
    cfg: 训练配置
    craft_config: CRaFT 配置（如果为 None，使用默认配置）
    accelerator: 分布式训练加速器（可选）
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
    # 加载锚点数据集（CRaFT 特有）
    # ============================================================================
    # 初始化 CRaFT 配置
    if craft_config is None:
        craft_config = CraftConfig()
    
    # 加载 AnchorCache
    anchor_dl_iter = None
    if craft_config.enabled and craft_config.anchor_cache_dir:
        if is_main_process:
            logging.info(colored("=" * 80, "cyan"))
            logging.info(colored("加载 AnchorCache", "cyan", attrs=["bold"]))
            logging.info(colored("=" * 80, "cyan"))
            logging.info(f"AnchorCache 目录: {craft_config.anchor_cache_dir}")
            logging.info(f"锚点批次大小: {craft_config.anchor_batch_size}")
            logging.info(f"保留频率 (K-step): 每 {craft_config.retention_freq} 步")
        
        try:
            from lerobot.craft.anchor_cache import AnchorCacheDataset
            
            anchor_dataset = AnchorCacheDataset(
                cache_dir=craft_config.anchor_cache_dir,
                transform=preprocessor,  # 使用与任务数据相同的预处理
            )
            
            anchor_dataloader = torch.utils.data.DataLoader(
                anchor_dataset,
                batch_size=craft_config.anchor_batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=True,
            )
            
            # 使用 cycle 使其无限循环
            anchor_dl_iter = cycle(anchor_dataloader)
            
            if is_main_process:
                logging.info(colored(f"✓ AnchorCache 加载成功: {len(anchor_dataset)} 样本", "green"))
        except Exception as e:
            if is_main_process:
                logging.warning(colored(f"⚠ AnchorCache 加载失败: {e}", "yellow"))
                logging.warning(colored("将以 baseline 模式运行（无保留损失）", "yellow"))
            craft_config.enabled = False
    elif craft_config.enabled:
        if is_main_process:
            logging.warning(colored("⚠ CRaFT 已启用但未提供 anchor_cache_dir", "yellow"))
            logging.warning(colored("将以 baseline 模式运行", "yellow"))
        craft_config.enabled = False
    
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
    # CRaFT 状态初始化
    # ============================================================================
    current_lambda = craft_config.initial_lambda
    
    # 计算 epsilon 衰减步数
    epsilon_decay_steps = craft_config.epsilon_decay_steps
    if epsilon_decay_steps == 0:
        epsilon_decay_steps = cfg.steps
    
    if is_main_process:
        logging.info(colored("=" * 80, "cyan"))
        logging.info(colored("CRaFT 训练配置", "cyan", attrs=["bold"]))
        logging.info(colored("=" * 80, "cyan"))
        logging.info(f"CRaFT 启用: {craft_config.enabled}")
        if craft_config.enabled:
            logging.info(f"初始 λ: {current_lambda}")
            logging.info(f"λ 学习率: {craft_config.lambda_lr}")
            logging.info(f"λ 最大值: {craft_config.lambda_max}")
            logging.info(f"ε 起始值: {craft_config.epsilon_start}")
            logging.info(f"ε 最终值: {craft_config.epsilon_end}")
            logging.info(f"ε 衰减步数: {epsilon_decay_steps}")
            logging.info(f"梯度投影: {craft_config.use_grad_projection}")
            logging.info(f"冲突阈值: {craft_config.conflict_threshold}")
            logging.info(f"合并模式: {craft_config.projection_mode}")
        logging.info(colored("=" * 80, "cyan"))
    
    # Lambda 历史记录（用于分析）
    lambda_history = [] if craft_config.save_lambda_history else None
    
    # ============================================================================
    # 训练循环
    # ============================================================================
    if is_main_process:
        logging.info(colored("=" * 80, "green"))
        logging.info(colored("开始训练", "green", attrs=["bold"]))
        logging.info(colored("=" * 80, "green"))
    
    while step < cfg.steps:
        # ========================================================================
        # 数据加载
        # ========================================================================
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        # ========================================================================
        # 计算当前 epsilon（退火调度）
        # ========================================================================
        from lerobot.craft.primal_dual import epsilon_schedule
        
        current_epsilon = epsilon_schedule(
            step,
            craft_config.epsilon_start,
            craft_config.epsilon_end,
            epsilon_decay_steps,
            schedule_type="linear",
        )
        
        # ========================================================================
        # K-step 策略：每 K 步计算一次保留损失
        # ========================================================================
        anchor_batch = None
        if craft_config.enabled and anchor_dl_iter is not None:
            if step % craft_config.retention_freq == 0:
                # 加载锚点批次
                anchor_batch = next(anchor_dl_iter)
                # 注意：anchor_batch 已经在 AnchorCacheDataset 中预处理过
        
        # ========================================================================
        # 训练步骤（CRaFT 或 baseline）
        # ========================================================================
        train_tracker, output_dict, current_lambda = update_policy_craft(
            train_tracker,
            policy,
            batch,
            anchor_batch=anchor_batch,
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
        
        # 保存 lambda 历史
        if lambda_history is not None and craft_config.enabled:
            lambda_history.append({
                "step": step,
                "lambda": current_lambda,
                "epsilon": current_epsilon,
                "retention_loss": output_dict.get("retention_loss"),
            })
        
        # ========================================================================
        # 日志记录
        # ========================================================================
        if step % cfg.log_freq == 0 and is_main_process:
            log_msg = f"Step {step}/{cfg.steps} | {train_tracker}"
            
            # CRaFT 特定指标
            if craft_config.enabled and "retention_loss" in output_dict:
                log_msg += f" | L_ret={output_dict['retention_loss']:.3f}"
                log_msg += f" | λ={current_lambda:.3f}"
                log_msg += f" | ε={current_epsilon:.3f}"
                
                if "grad_conflict" in output_dict and output_dict["grad_conflict"] > 0:
                    log_msg += " | conflict=✓"
                
                if "grad_dot" in output_dict:
                    log_msg += f" | cos={output_dict['grad_dot']:.3f}"
            
            logging.info(log_msg)
            
            # WandB 日志
            if wandb_logger is not None:
                wandb_log_dict = {
                    "train/loss": train_tracker.loss.avg,
                    "train/grad_norm": train_tracker.grad_norm.avg,
                    "train/lr": train_tracker.lr.avg,
                    "train/step": step,
                }
                
                if craft_config.enabled and "retention_loss" in output_dict:
                    wandb_log_dict.update({
                        "craft/retention_loss": output_dict["retention_loss"],
                        "craft/lambda": current_lambda,
                        "craft/epsilon": current_epsilon,
                        "craft/grad_conflict": output_dict.get("grad_conflict", 0),
                    })
                    if "grad_dot" in output_dict:
                        wandb_log_dict["craft/grad_dot"] = output_dict["grad_dot"]
                
                wandb_logger.log(wandb_log_dict, step=step)
        
        # ========================================================================
        # 评估
        # ========================================================================
        if cfg.eval_freq > 0 and step % cfg.eval_freq == 0 and eval_env is not None:
            if is_main_process:
                logging.info(colored(f"Evaluating at step {step}", "yellow", attrs=["bold"]))
            
            policy.eval()
            
            with torch.no_grad():
                eval_info = eval_policy_all(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    max_episodes_rendered=cfg.eval.n_episodes_rendered,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step:06d}",
                    start_seed=cfg.seed,
                )
            
            if is_main_process:
                logging.info(f"Eval success rate: {eval_info['aggregated']['pc_success']:.2%}")
                
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "eval/success_rate": eval_info["aggregated"]["pc_success"],
                            "eval/avg_reward": eval_info["aggregated"]["avg_sum_reward"],
                        },
                        step=step,
                    )
            
            policy.train()
        
        # ========================================================================
        # 保存检查点
        # ========================================================================
        if cfg.save_checkpoint and step % cfg.save_freq == 0:
            if is_main_process:
                logging.info(colored(f"Saving checkpoint at step {step}", "yellow"))
            
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, step)
            save_checkpoint(
                checkpoint_dir,
                policy,
                optimizer,
                lr_scheduler,
                step,
                cfg,
                accelerator=accelerator,
            )
            
            # 保存 CRaFT 状态
            if craft_config.enabled and is_main_process:
                craft_state = {
                    "lambda": current_lambda,
                    "epsilon": current_epsilon,
                    "step": step,
                }
                if lambda_history is not None:
                    craft_state["lambda_history"] = lambda_history
                
                torch.save(craft_state, checkpoint_dir / "craft_state.pt")
            
            if is_main_process:
                update_last_checkpoint(cfg.output_dir, checkpoint_dir)
    
    # ============================================================================
    # 训练结束
    # ============================================================================
    if is_main_process:
        logging.info(colored("=" * 80, "green"))
        logging.info(colored("训练完成！", "green", attrs=["bold"]))
        logging.info(colored("=" * 80, "green"))
        
        # 保存最终检查点
        if cfg.save_checkpoint:
            final_checkpoint_dir = cfg.output_dir / "final"
            save_checkpoint(
                final_checkpoint_dir,
                policy,
                optimizer,
                lr_scheduler,
                step,
                cfg,
                accelerator=accelerator,
            )
            
            # 保存 CRaFT 最终状态
            if craft_config.enabled:
                craft_state = {
                    "lambda": current_lambda,
                    "epsilon": current_epsilon,
                    "step": step,
                }
                if lambda_history is not None:
                    craft_state["lambda_history"] = lambda_history
                
                torch.save(craft_state, final_checkpoint_dir / "craft_state.pt")
                
                # 保存 lambda 历史为 CSV（便于分析）
                if lambda_history is not None:
                    import csv
                    with open(final_checkpoint_dir / "lambda_history.csv", "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["step", "lambda", "epsilon", "retention_loss"])
                        writer.writeheader()
                        writer.writerows(lambda_history)
                    
                    logging.info(f"Lambda 历史已保存到: {final_checkpoint_dir / 'lambda_history.csv'}")
    
    if eval_env:
        close_envs(eval_env)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    if is_main_process:
        logging.info("CRaFT 训练结束")


def main():
    register_third_party_plugins()
    train_craft()


if __name__ == "__main__":
    main()

