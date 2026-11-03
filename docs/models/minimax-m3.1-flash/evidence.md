---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: hf-config.json@main(2026-10-26, sha256 3bea745c2d1f602a1d0f25532717330cfe856dd6f1074907314bf3540c53fa54); vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879
Content-Status: verified
---

# 证据索引

vLLM 源码引用格式：`文件@8fae0ff5:符号（行号）`。本地冻结快照位于
`.scratch/add-minimax-m3.1-flash/sources/`（仅供本次核验，不作为页面内容）：

| 快照 | sha256 |
| --- | --- |
| vllm-minimax_m3-__init__.py | 6a7ead66… |
| vllm-nvidia-model.py | a5ca5463… |
| vllm-common-sparse_attention.py | 32eb23f7… |
| vllm-common-indexer.py | b4ac1543… |
| vllm-nvidia-sparse_attention_msa.py | 7d1f8a05… |
| vllm-nvidia-indexer_msa.py | 94479e53… |
| vllm-nvidia-mtp.py | 7c796ee6… |
| hf-config.json | 3bea745c… |

> 注：快照取自本地工作树（Windows CRLF 行尾），`git show 8fae0ff5:...`
> 存储为 LF，字节哈希不同但语义一致（`git diff` 为空）。如需字节级一致，
> 改用 `git show` 输出冻结。

## 页面/图节点 → 源码

| 节点/边 ID | 阶段 | 语义/shape | 源文件@SHA:符号 | 配置/权重证据 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| overview: embed→60×decoder→norm→lm_head | 全程 | [B,S,H] → [B,S,V] | nvidia/model.py@8fae0ff5: MiniMaxM3Model.forward（L878–932） | hf-config: hidden 6144 / vocab 200064 | 已核实 |
| overview: L0–2 sparse+dense MLP | 层族 | 3 层 | nvidia/model.py@8fae0ff5: _sparse_attention_layer_ids（L99–107）、_is_moe_layer（L110–115） | hf-config: sparse_attention_freq 全 1；moe_layer_freq 前三 0 | 已核实 |
| overview: L3–59 sparse+MoE | 层族 | 57 层 | 同上 | 同上 | 已核实 |
| norm（input/post/final） | 每层 | [B,S,6144] | nvidia/model.py@8fae0ff5: MiniMAXGemmaRMSNorm（L118–146） | hf-config: rms_norm_eps 1e−6, use_gemma_norm true | 已核实 |
| 五路投影 qkv+idx | 每层 | [Nq,9856] | nvidia/model.py@8fae0ff5: L444–455 | hf-config: 64Q/4KV/4idx/128idx_dim | 已核实 |
| Q/K norm + partial RoPE + cache insert | 每层 | 融合 kernel | nvidia/model.py@8fae0ff5: forward L625–650 | hf-config: rope_theta 1e7, partial 0.5, qk_norm_type per_head | 已核实 |
| 主 KV cache（NVFP4） | 每层 | 128 token/page | nvidia/model.py@8fae0ff5: use_nvfp4_kv（L504）、kv scales（L648–649）；common/sparse_attention.py: NVFP4 仅 MSA（select_main_backend_and_impl_cls） | 部署 --kv-cache-dtype nvfp4；博客 34,560 B/token | 已核实 |
| index K side cache | 每层 | key-only [T,128] | common/indexer.py@8fae0ff5: MiniMaxM3IndexerCache（L102–151） | hf-config: sparse_disable_index_value 全 1 | 已核实 |
| indexer score/mask/block-max | 每层 | [B,N_idx,S,T]→blocks | common/indexer.py@8fae0ff5: MiniMaxM3Indexer（L578–） | hf-config: score_type max, block 128 | 已核实 |
| top-16 选择 | 每层 | [token,head,16] int32 | nvidia/model.py@8fae0ff5: topk_indices_buffer（L826–846）；common/indexer.py: index_topk | hf-config: sparse_topk_blocks 16, local 1, init 0 | 已核实 |
| sparse attend（块内精确） | 每层 | ≤2048 keys/query | nvidia/sparse_attention_msa.py@8fae0ff5: MiniMaxM3SparseMSAImpl（L193–）；common/sparse_attention.py: MiniMaxM3SparseTritonImpl | 博客 MSA primer | 已核实 |
| o_proj | 每层 | [6144,8192] | nvidia/model.py@8fae0ff5: L459–466 | hf-config | 已核实 |
| dense MLP（L0–2） | FFN | 12288 宽 | nvidia/model.py@8fae0ff5: MiniMaxM3MLP（L149–192） | hf-config: dense_intermediate_size 12288 | 已核实 |
| router（FP32 logits） | MoE 层 | [Nq,128] | nvidia/model.py@8fae0ff5: MiniMaxM3MoE.forward（L279–289） | hf-config: 128 experts, sigmoid | 已核实 |
| FusedMoE Top-4 + shared | MoE 层 | [Nq,6144] | nvidia/model.py@8fae0ff5: FusedMoEFactory（L251–272） | hf-config: top-4, bias true, scaling 2.0 | 已核实 |
| MXFP8 线性层量化 | 权重 | [1,32] 块 scale | nvidia/model.py@8fae0ff5: weight_scale_inv→weight_scale（L975–976） | hf-config: quantization_config | 已核实 |
| NVFP4 专家量化 | 权重 | nvfp4-pack | FusedMoE 量化层（vLLM 量化栈） | hf-config: moe_quant_algo NVFP4 | 已核实（格式键）；逐专家字节布局未展开 |
| DSpark drafter | 外部投机解码 | 4096 滑窗 | vllm/config/speculative.py（dspark 方法） | 博客 quick-start；HF 模型卡 "the release includes the draft head" | 已核实（接口级 + 模型卡） |
| 权重 shard 定位 | 权重 | — | — | safetensors index 未获取 | **未验证** |

## 公式 → 源码

| 公式 ID | 源码锚点 | 核验状态 |
| --- | --- | --- |
| F-NORM | nvidia/model.py L118–146 | 已核实 |
| F-PROJ5 | nvidia/model.py L444–455、L952–961 | 已核实 |
| F-QKNORM | nvidia/model.py L469–470、L482–487 | 已核实 |
| F-ROPE | nvidia/model.py L473–488；θ=1e7 见 hf-config | 已核实 |
| F-IDXSCORE | common/indexer.py MiniMaxM3Indexer | 已核实（块 max/未来屏蔽）；缩放位置以 kernel 为准 |
| F-TOPK | common/indexer.py index_topk；init 0 / local 1 | 已核实 |
| F-ATTN | nvidia/sparse_attention_msa.py + common/ops/sparse_attn.py | 已核实（数学分解标注） |
| F-MLP | nvidia/model.py L149–192；activation.py SiluAndMulWithClamp | 已核实 |
| F-MOE | nvidia/model.py L195–289 | 已核实 |
| F-SLOT | common/sparse_attention.py metadata builder；vLLM runner | 已核实 |

## 后端差异

| 项 | NVIDIA | AMD |
| --- | --- | --- |
| 模型文件 | nvidia/model.py | amd/model.py（未验证） |
| NVFP4 KV | SM100 MSA（QMUL4 / CUTLASS decode） | FlyDSL MSA kernel（未验证） |
| indexer DCP | 有（博客） | 有（未验证） |