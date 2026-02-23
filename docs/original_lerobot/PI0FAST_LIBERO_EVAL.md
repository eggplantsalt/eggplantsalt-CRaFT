# 使用原始 LeRobot 在 LIBERO 上评测 Pi0Fast

> 目标: 不依赖 CRaFT, 直接用 LeRobot 的 Pi0Fast 预训练权重在 LIBERO 基准上做评测, 并能解释每个关键参数和输出指标。

---

## 适用范围与前置概念

- 本文面向 LeRobot 原生流程, 不涉及 CRaFT 的训练/缓存/评测脚本。
- 评测对象为 LIBERO 仿真基准, 使用 `lerobot-eval` 统一评测入口。
- 推荐权重: `lerobot/pi0fast-libero` (Pi0Fast 在 LIBERO 上微调后的权重)。

LIBERO 套件说明:
- `libero_object`: 以物体操作为主
- `libero_spatial`: 强调空间关系
- `libero_goal`: 目标条件变化
- `libero_10`: 10 个长时序任务子集
- `libero_90`: 90 个短时序任务子集

参考来源:
- Pi0Fast 说明与 LIBERO 评测命令: [docs/source/pi0fast.mdx](docs/source/pi0fast.mdx)
- LIBERO 评测入口与参数说明: [docs/source/libero.mdx](docs/source/libero.mdx)
- 统一评测脚本入口: [src/lerobot/scripts/lerobot_eval.py](src/lerobot/scripts/lerobot_eval.py)

---

## 步骤 1: 环境与依赖

### 1.1 Python 环境

- Python >= 3.10
- 推荐使用虚拟环境

### 1.2 安装 LeRobot + Pi0Fast + LIBERO 依赖

在仓库根目录执行:

```bash
pip install -e .
pip install -e ".[pi]"
pip install -e ".[libero]"
```

说明:
- `.[pi]` 安装 Pi0Fast 依赖
- `.[libero]` 安装 LIBERO 仿真依赖

### 1.3 MuJoCo 渲染后端

LIBERO 使用 MuJoCo 渲染。

- 无界面服务器: 设置 `MUJOCO_GL=egl`
- 本地桌面(Windows/macOS/Linux): 通常使用默认 `glfw`

示例 (PowerShell):

```powershell
$env:MUJOCO_GL = "glfw"
```

---

## 步骤 2: 选择评测权重

推荐使用已在 LIBERO 上微调的权重:

- `lerobot/pi0fast-libero`

可选: 也可使用 `lerobot/pi0fast-base` 进行 sanity check, 但 LIBERO 表现会明显下降。

---

## 步骤 3: 关键数据接口与命名约定

LIBERO 在 LeRobot 中会输出以下观测键, Pi0Fast 预训练权重对图像键名非常敏感:

- `observation.images.image`: 主摄像头 (agentview)
- `observation.images.image2`: 腕部摄像头 (wrist)
- `observation.state`: 机器人状态向量 (由环境处理器拼接)

Pi0Fast 在 LIBERO 上的权重要求图像键名对齐到训练时的命名。评测时需要使用 `--rename_map` 把环境输出的键映射到模型期望的键:

- `observation.images.image` -> `observation.images.base_0_rgb`
- `observation.images.image2` -> `observation.images.left_wrist_0_rgb`

如果不设置 `--rename_map`, 通常会出现 KeyError 或归一化层找不到统计量。

---

## 步骤 4: 单套件评测 (示例)

以下示例评测 `libero_object` 套件:

```bash
lerobot-eval \
  --policy.path=lerobot/pi0fast-libero \
  --policy.max_action_tokens=256 \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

参数解释:
- `--policy.path`: 模型权重来源, 可是 Hub repo 或本地目录
- `--policy.max_action_tokens`: Pi0Fast 每个动作块的最大 token 数
- `--policy.gradient_checkpointing=false`: 评测时关闭以提升推理速度
- `--env.type=libero`: 选择 LIBERO 环境
- `--env.task`: 套件名称
- `--eval.batch_size`: 并行环境数量, 显存不足时调小
- `--eval.n_episodes`: 总评测回合数
- `--rename_map`: 图像键名映射, 确保与权重统计对齐

---

## 步骤 5: 多套件评测 (官方复现配置)

评测多个套件, 对应官方复现命令:

```bash
lerobot-eval \
  --policy.path=lerobot/pi0fast-libero \
  --policy.max_action_tokens=256 \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_object,libero_spatial,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

说明:
- `--eval.n_episodes` 可以改成 5/10 等更稳定的统计
- 如果希望评测子任务, 可追加 `--env.task_ids=[0,1,2]`

---

## 步骤 6: 输出与结果解读

默认输出目录由 LeRobot 自动生成:

```
outputs/eval/YYYY-MM-DD/HH-MM-SS_libero_pi0_fast/
```

常见输出文件:
- `eval_info.json`: 汇总指标, 包含成功率/回报/耗时等
- `videos/`: 评测视频 (若启用渲染/保存)

核心指标含义:
- 成功率: `pc_success` 或 success 相关字段, 反映任务完成比例
- 回报: `avg_sum_reward` 或 `avg_max_reward`, 衡量策略在回合中的累计奖励
- 速度: `eval_ep_s` 或 `eval_s`, 用于估计评测耗时

可指定输出目录:

```bash
lerobot-eval ... --output_dir=outputs/eval/pi0fast_libero
```

---

## 关键参数速查

- `--env.task`: 评测套件
- `--env.task_ids`: 评测指定任务索引
- `--env.control_mode`: `relative` 或 `absolute`
- `--env.episode_length`: 手动限制最大步数
- `--eval.batch_size`: 并行环境数量
- `--eval.n_episodes`: 总评测回合数
- `--seed`: 随机种子, 便于复现
- `--policy.max_action_tokens`: Pi0Fast 最大动作 token 数

---

## 常见问题与排查

### 1) 找不到 LIBERO 或 MuJoCo 相关依赖

```bash
pip install -e ".[libero]"
```

### 2) 图像键不匹配导致 KeyError

确保添加 `--rename_map`:

```bash
--rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

### 3) 显存不足

- 调小 `--eval.batch_size`
- 减少 `--eval.n_episodes`
- 只评测单套件

### 4) 评测速度过慢

- 关闭 `--policy.gradient_checkpointing`
- 减少 `--eval.n_episodes`
- 减少 `--env.task_ids`

---

## 推荐的验证顺序

1. `libero_object` + `n_episodes=1` 做快速 sanity check
2. 增大 `n_episodes` 得到稳定成功率
3. 扩展到多套件评测

