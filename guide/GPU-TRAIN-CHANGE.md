# 🚀 GPU 性能优化任务：创建 `train_gpu.py`

## 📋 背景信息

### 当前硬件环境（虚拟机）

| 组件 | 规格 | 说明 |
| :--- | :--- | :--- |
| **CPU** | Intel Xeon Gold 6348 @ 2.60 GHz | 28 核心，服务器级处理器 |
| **GPU** | NVIDIA A40-2Q | 2GB 显存（虚拟化切分），专业计算卡 |
| **内存** | 16 GB | 充足 |
| **计费模式** | **按小时计费** | 速度越快、成本越低 |
| **当前 GPU 利用率** | **仅 4%** | 严重浪费 |

### 问题现状

当前 `train.py` 在 GPU 上运行时，`fps` 仅为 **31**，与 CPU 训练（27-30 FPS）几乎无异，GPU 利用率仅 **4%**。

**根本原因**：
- 网络规模太小（256 神经元 × 2 层）
- `batch_size` 太小（64）
- `n_steps` 太小（2048）
- 单环境串行收集经验

**结果**：GPU 处于“饥饿”状态，数据从 CPU 拷贝到显存的开销甚至超过了计算收益。

---

## 🎯 优化目标

| 指标 | 当前值 | 目标值 |
| :--- | :--- | :--- |
| GPU 利用率 | 4% | **60-85%** |
| 训练速度（FPS） | 31 | **200-500**（视配置） |
| 每窗口训练时间 | 15-20 分钟 | **2-5 分钟** |

---

## 📁 需要生成的文件

在项目根目录下创建 **`src/train_gpu.py`**，与 `train.py` 同级。

**要求**：
1. 保留 `train.py` 的全部功能逻辑（数据加载、环境、评估、日志、checkpoint 保存）。
2. 针对 GPU 进行性能优化，充分利用 A40-2Q（显存 2GB）。
3. 保持与现有 checkpoint 格式兼容（使用 `stable-baselines3` 2.0.0，与 `venv_gpu` 环境一致）。
4. 支持从指定窗口恢复训练（`--resume` 参数）。
5. 添加性能监控（打印 GPU 利用率、显存占用）。

---

## 🔧 具体优化策略

### 1. 增大 PPO 超参数（显存 2GB 限制内）

| 参数 | 当前值（`train.py`） | 建议优化值（`train_gpu.py`） | 说明 |
| :--- | :--- | :--- | :--- |
| `batch_size` | 64 | **256** | 让 GPU 一次处理更多样本 |
| `n_steps` | 2048 | **4096** 或 **8192** | 单次 rollout 收集更多经验 |
| `policy_kwargs.net_arch` | `[256, 256]` | **`[512, 512]`** 或 **`[1024, 1024]`** | 增大网络容量，更好地利用 GPU 并行 |
| `learning_rate` | 3e-4 | **3e-4**（保持不变） | |
| `gamma` | 0.99 | 0.99 | |
| `max_grad_norm` | 0.5 | 0.5 | |

**注意**：A40-2Q 显存仅 2GB，`batch_size=512` + `n_steps=8192` 可能触发 `CUDA out of memory`，需要从保守值开始尝试。

### 2. 多环境并行（关键优化）

当前使用 `DummyVecEnv`（单环境串行）。改为 **`SubprocVecEnv`** 并行运行多个环境，让 CPU 同时收集经验，GPU 专注训练。

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_vec_env(df, news_df, window_days, n_envs=4):
    def _factory():
        return TradingEnv(df=df, news_df=news_df, window_days=window_days)
    return SubprocVecEnv([_factory for _ in range(n_envs)])
```

- **建议 `n_envs`**：**4**（A40-2Q 显存限制下，4 个环境比较安全）
- 如果显存充足，可尝试 8。

### 3. 启用 `torch.compile`（PyTorch 2.0+ 支持）

如果 PyTorch 版本 >= 2.0，可以尝试：

```python
import torch
torch.compile(model.policy)
```

### 4. 调整 `policy_kwargs` 使用 GPU 友好激活函数

```python
policy_kwargs = {
    "net_arch": dict(pi=[512, 512], vf=[512, 512]),
    "activation_fn": torch.nn.ReLU,  # GPU 上 ReLU 比 Tanh 更快
}
```

### 5. 添加 GPU 性能监控

在训练循环中，每 10 次迭代打印 GPU 利用率：

```python
if hasattr(torch.cuda, 'utilization'):
    print(f"GPU Util: {torch.cuda.utilization()}%")
```

（注：`torch.cuda.utilization()` 在 Windows 上可能不可用，可以使用 `nvidia-smi` 或 `pynvml` 库）

---

## 📋 命令行参数（与 `train.py` 保持一致）

```bash
python -m src.train_gpu --output models/news_gpu --device cuda [--resume N] [--test]
```

| 参数 | 说明 |
| :--- | :--- |
| `--output` | 输出目录（默认 `models/news_gpu`） |
| `--device` | 默认 `cuda`（在 `train_gpu.py` 中硬编码为 `cuda`） |
| `--resume N` | 从第 N 个窗口恢复训练（加载对应的 checkpoint） |
| `--test` | 只训练 1 个窗口，用于快速验证 |

---

## ⚠️ 注意事项

1. **显存限制**：A40-2Q 只有 2GB 显存，如果 OOM，请降低 `batch_size`、`n_steps` 或 `n_envs`。
2. **Checkpoint 兼容性**：由于 `venv_gpu` 使用 SB3 2.0.0，与 `venv_cpu` 的 SB3 1.7.0 checkpoint **不兼容**。因此：
   - `train_gpu.py` 应该**从头开始训练**（或只加载 SB3 2.0.0 生成的 checkpoint）。
   - 如果使用 `--resume`，只能恢复 `train_gpu.py` 自己生成的 checkpoint（`models/news_gpu/` 目录）。
3. **News 数据**：保持与 `train.py` 一致，从 `enhanced_data.parquet` 加载。