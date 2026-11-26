# llm-quant-edge-sprint

面向 LLM 量化与端侧部署的求职冲刺实验仓库。

## Day 1: FP16 推理（Mac/MPS，非正式基线）

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 环境：MacBook，conda 环境 `jiong`，`torch 2.15.0`，backend `mps`，`dtype=float16`
- 脚本：`day01_fp16_baseline/infer_fp16.py`
- 提示：`用三句话解释什么是 KV Cache，以及它为什么能加快自回归解码。`
- 设置：`do_sample=False`，`max_new_tokens=128`
- 结果（首次加载含下模型时间）：
  - `load_s=102.61`
  - `new_tokens=54`，`gen_s=0.727`，`tokens_per_s=74.27`
  - `mps_allocated_gb=2.90`

这组数字只能证明流程跑通，不能当作作品 A 的正式基线。正式对照需在 A800（或至少一块 NVIDIA GPU）上，用同一脚本、同一提示再跑，并同时记录 TTFT 或总耗时、tokens/s、峰值显存，以及 generate 原文。

generate 原文仅作可复现对照，其中对 KV Cache 的解释并不准确，不要当作定义。