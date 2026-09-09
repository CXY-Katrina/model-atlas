# MiniMax-M3 RoPE 维度核对

核对日期：2026-09-08

## 结论

需要区分文本解码器和视觉塔：

- **MiniMax-M3 文本解码器确实使用 50% Partial RoPE。**准确参数不是“RoPE 维度是 64 和 128”，而是 `head_dim = 128`、`rotary_dim = 64`、`partial_rotary_factor = 0.5`。因此每个 Q/K head 的前 64 个通道参与 RoPE，后 64 个通道原样通过。
- **MiniMax-M3-VL 视觉塔不是这套 64/128 Partial RoPE。**Transformers 的视觉 RoPE 文档明确说明最终角度覆盖整个 head 维度，不使用 partial rotation。

建议页面使用这句话：

> 每个文本 Q/K head：128 维 = 前 64 维 Partial NeoX RoPE + 后 64 维原样通过。

## 一手证据

### 1. MiniMax 官方 checkpoint 配置

MiniMax 官方 `MiniMaxAI/MiniMax-M3` checkpoint 的 `text_config` 同时给出：

```json
"head_dim": 128,
"rotary_dim": 64,
"partial_rotary_factor": 0.5
```

来源：[MiniMax-M3 `config.json`，固定版本 `f0e1c1e`](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/f0e1c1e04d40177e4673a22097036854f536e9c0/config.json#L14-L22)。该模型的[初始发布提交](https://huggingface.co/MiniMaxAI/MiniMax-M3/commit/3a41b311ffa5719cef48fed3974ccf2cc03733ea)也已经是这组值，并非后续框架自行推断。

### 2. Transformers 的维度计算和真实执行路径

Transformers 配置把 `rotary_dim` 定义为“参与 RoPE 的 head 通道数，其余通道保持不变”，默认 `head_dim = 128`、`rotary_dim = 64`：[配置定义，固定版本 `0df4ef3`](https://github.com/huggingface/transformers/blob/0df4ef369d324d4072e2910c673671eed4e92459/src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py#L30-L37)、[默认维度](https://github.com/huggingface/transformers/blob/0df4ef369d324d4072e2910c673671eed4e92459/src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py#L82-L90)。

实现先计算：

```python
dim = int(head_dim * partial_rotary_factor)  # 128 * 0.5 = 64
```

来源：[Transformers RoPE 参数计算](https://github.com/huggingface/transformers/blob/0df4ef369d324d4072e2910c673671eed4e92459/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L298-L308)。随后 `apply_rotary_pos_emb` 按 `rotary_dim` 拆出 `q_rot/q_pass` 与 `k_rot/k_pass`，只旋转 `*_rot`，再与 `*_pass` 拼接：[Transformers 拆分、旋转与拼接](https://github.com/huggingface/transformers/blob/0df4ef369d324d4072e2910c673671eed4e92459/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L365-L396)。

### 3. 当前网页固定引用的 vLLM 版本

仓库固定引用 vLLM commit `edd4c8176cfd98ece8a29beda574378c42971967`。该版本的 MiniMax-M3 实现明确写有：

```python
# Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
```

并以 `head_dim` 和 `partial_rotary_factor` 创建 RoPE，再把 `self.rotary_emb.rotary_dim` 传给融合 kernel：[vLLM MiniMax-M3 实现](https://github.com/vllm-project/vllm/blob/edd4c8176cfd98ece8a29beda574378c42971967/vllm/models/minimax_m3/nvidia/model.py#L312-L378)。vLLM 的 `get_rope` 实际执行 `rotary_dim = int(head_size * partial_rotary_factor)`：[vLLM `get_rope`](https://github.com/vllm-project/vllm/blob/edd4c8176cfd98ece8a29beda574378c42971967/vllm/model_executor/layers/rotary_embedding/__init__.py#L59-L80)。

### 4. 为什么容易记成“不是半旋转”

有两种常见混淆：

1. `rotate_half()` 是标准 NeoX RoPE 在 **64 维旋转子空间内部**进行 32+32 配对的实现细节；它不表示只旋转 32 维。真正参与 RoPE 的仍是 64 个通道。
2. 同一 MiniMax-M3-VL 的**视觉塔**采用全 head 维度的 axial RoPE。Transformers 源码明确写明视觉侧没有 partial rotation：[视觉 RoPE 说明](https://github.com/huggingface/transformers/blob/0df4ef369d324d4072e2910c673671eed4e92459/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L995-L1004)。网页当前的 `Partial RoPE (Q/K)` 节点描述的是文本 decoder，不是视觉塔。

## 判定

对当前网页展示的 **MiniMax-M3 文本 decoder attention**，使用 `head_dim = 128`、`rotary_dim = 64` 和“前 64 维旋转、后 64 维直通”的表达是正确的，并与 MiniMax checkpoint、Transformers 和仓库固定的 vLLM 版本一致。
