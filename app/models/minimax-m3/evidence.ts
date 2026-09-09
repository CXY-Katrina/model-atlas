import type { CodeSection, CodeSymbol, CodeDetail, IoBinding } from "../types";
import { CODE_URL, ACTIVATION_URL, LINEAR_URL, TRANSFORMERS_MINIMAX_M3_URL, TRANSFORMERS_MOE_URL, TRANSFORMERS_QKV_PROJECTION_URL, TRANSFORMERS_INDEX_PROJECTION_URL, TRANSFORMERS_SPARSE_ATTENTION_URL, TRANSFORMERS_INDEX_SELECTION_URL, TRANSFORMERS_BLOCK_MASK_URL, VLLM_INDEXER_URL, VLLM_SPARSE_ATTENTION_URL, NORM_FORWARD_URL, DECODER_FORWARD_URL, FLASHINFER_GEMMA_NORM_URL } from "./sources";

const NORM_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"MiniMAXGemmaRMSNorm.forward：选择普通或 fused kernel",location:"nvidia/model.py · L130–142",url:NORM_FORWARD_URL,code:`def forward(self, x, residual=None):
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
    if residual is None:
        return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
    # mutates x and residual in place
    gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
    return x, residual`},
  {stage:"2 · RESIDUAL",title:"MiniMaxM3DecoderLayer.forward：residual 的创建与更新位置",location:"nvidia/model.py · L752–778",url:DECODER_FORWARD_URL,code:`if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

hidden_states = self.self_attn(...)
hidden_states, residual = fused_allreduce_gemma_rms_norm(
    hidden_states, residual, self.post_attention_layernorm
)`},
  {stage:"3 · ENTER",title:"FlashInfer gemma_rmsnorm：kernel 的实际数学定义",location:"flashinfer.norm.gemma_rmsnorm",url:FLASHINFER_GEMMA_NORM_URL,code:`RMS(x) = sqrt(mean(x²) + eps)
out[i] = (x[i] / RMS(x)) * (weight[i] + 1)`},
];

const NORM_SYMBOLS: CodeSymbol[] = [
  {symbol:"x / hidden_states",resolvesTo:"待归一化分支",meaning:"首个分支直接归一化 x；fused 分支先把 x 加入 residual。"},
  {symbol:"residual",resolvesTo:"残差累加器 r′",meaning:"首次为空时保存当前 hidden_states；后续 fused 调用原地更新为 residual + x。"},
  {symbol:"self.weight",resolvesTo:"γ",meaning:"checkpoint 保存 γ；Gemma kernel 实际使用 γ+1 作为逐元素缩放。"},
  {symbol:"self.variance_epsilon",resolvesTo:"ε=10⁻⁶",meaning:"计算 RMS 时用于数值稳定。"},
];

const RESIDUAL_MERGE_SECTIONS: CodeSection[] = [
  {stage:"1 · EXIT",title:"DecoderLayer.forward：当前层先返回两条独立流",location:"nvidia/model.py · L776–778",url:`${CODE_URL}#L776-L778`,code:`ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
hidden_states = ffn(hidden_states)   # Yffn / Ymoe
return hidden_states, residual       # residual is U`},
  {stage:"2 · ENTER",title:"下一 Decoder Layer：fused norm 内完成 residual merge",location:"nvidia/model.py · L758–767",url:`${CODE_URL}#L758-L767`,code:`if self.fuse_input_allreduce and residual is not None:
    hidden_states, residual = fused_allreduce_gemma_rms_norm(
        hidden_states, residual, self.input_layernorm
    )
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)`},
];

const RESIDUAL_MERGE_SYMBOLS: CodeSymbol[] = [
  {symbol:"hidden_states",resolvesTo:"Yffn / Ymoe",meaning:"当前 FFN 计算分支的输出。"},
  {symbol:"residual",resolvesTo:"U",meaning:"Attention 后沿 Layer 边界保留的 residual stream。"},
  {symbol:"logical Xₗ₊₁",resolvesTo:"U + Yffn / Ymoe",meaning:"图中 Add 的语义；实际融合进下一层 input RMSNorm 或 Final Norm。"},
];

const ATTENTION_RESIDUAL_SECTIONS: CodeSection[] = [
  {stage:"1 · CALL",title:"DecoderLayer.forward：Layer 内融合 Attention residual 与 post-norm",location:"nvidia/model.py · L773–775",url:`${CODE_URL}#L773-L775`,code:`hidden_states, residual = fused_allreduce_gemma_rms_norm(
    hidden_states, residual, self.post_attention_layernorm
)`},
];

const ATTENTION_RESIDUAL_SYMBOLS: CodeSymbol[] = [
  {symbol:"hidden_states",resolvesTo:"Yattn",meaning:"L768–771 的 self_attn 输出。"},
  {symbol:"residual",resolvesTo:"Xₗ → U",meaning:"fused kernel 原地执行 residual += hidden_states。"},
  {symbol:"returned hidden_states",resolvesTo:"Û",meaning:"同一个 fused kernel 随后对更新后的 U 执行 Gemma RMSNorm。"},
];

const MLP_SECTIONS: CodeSection[] = [
  {stage:"2 · CALL",title:"MiniMaxM3MLP.forward：调用顺序",location:"nvidia/model.py · L165–171",url:`${CODE_URL}#L165-L171`,code:`def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x, _ = self.down_proj(x)
    return x`},
  {stage:"3 · ENTER",title:"SiluAndMulWithClamp.forward_native：展开 self.act_fn",location:"activation.py · L214–218",url:`${ACTIVATION_URL}#L214-L218`,code:`def forward_native(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
    up = torch.clamp(
        x[..., d:],
        min=-self.swiglu_limit,
        max=self.swiglu_limit,
    )
    return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)`},
];

const MLP_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.gate_up_proj",resolvesTo:"MergedColumnParallelLinear",meaning:"一次并行 GEMM 产生 packed [gate | up]，随后沿最后一维平分。"},
  {symbol:"self.act_fn",resolvesTo:"SiluAndMulWithClamp",meaning:"不是未说明的黑盒 SiLU；内部完成 split、clamp、sigmoid 与逐元素乘法。"},
  {symbol:"self.down_proj",resolvesTo:"RowParallelLinear",meaning:"把激活后的中间维投回 hidden_size，并按配置归并 TP 结果。"},
  {symbol:"swiglu_limit / alpha / beta",resolvesTo:"7.0 / 1.702 / 1.0",meaning:"来自 MiniMax-M3 config，并直接传入激活算子。"},
];

const GATE_UP_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"MiniMaxM3MLP.__init__：创建 fused column-parallel 投影",location:"nvidia/model.py · L157–163",url:`${CODE_URL}#L157-L163`,code:`self.gate_up_proj = MergedColumnParallelLinear(
    config.hidden_size,              # H = 6144
    [intermediate_size] * 2,         # 2 × H_dense, H_dense = 12288
    bias=False,
    prefix=f"{prefix}.gate_up_proj",
)`},
  {stage:"2 · CALL",title:"MiniMaxM3MLP.forward：只调用 gate_up_proj",location:"nvidia/model.py · L184–185",url:`${CODE_URL}#L184-L185`,code:`def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = self.gate_up_proj(x)`},
  {stage:"3 · ENTER",title:"ColumnParallelLinear：按 TP 切输出维并执行 GEMM",location:"linear.py · L460–467, L569–587",url:`${LINEAR_URL}#L460-L587`,code:`self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [
    divide(output_size, self.tp_size) for output_size in self.output_sizes
]

output_parallel = self.quant_method.apply(self, input_, bias)
output = output_parallel  # gather_output=False`},
];

const GATE_UP_SYMBOLS: CodeSymbol[] = [
  {symbol:"x / Û",resolvesTo:"[B,S,H], H=6144",meaning:"每个 TP rank 都读取完整 hidden 输入。"},
  {symbol:"output_sizes",resolvesTo:"[H_dense,H_dense]",meaning:"gate 与 up 的全局宽度各为 H_dense=12288。"},
  {symbol:"output_partition_sizes",resolvesTo:"[H_dense/TP,H_dense/TP]",meaning:"MergedColumnParallelLinear 沿输出维切分，每 rank 只产生两块局部投影。"},
  {symbol:"gate_up",resolvesTo:"[B,S,2H_dense/TP]",meaning:"这里只产生 packed 线性投影；Split、clamp 和 sigmoid 属于后续节点。"},
];

const SWIGLU_SECTIONS: CodeSection[] = [MLP_SECTIONS[1]];
const SWIGLU_SYMBOLS: CodeSymbol[] = [MLP_SYMBOLS[1],MLP_SYMBOLS[3]];

const DOWN_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"MiniMaxM3MLP.__init__：创建 row-parallel down projection",location:"nvidia/model.py · L164–170",url:`${CODE_URL}#L164-L170`,code:`self.down_proj = RowParallelLinear(
    intermediate_size,       # H_dense = 12288
    config.hidden_size,      # H = 6144
    bias=False,
    reduce_results=reduce_results,
    prefix=f"{prefix}.down_proj",
)`},
  {stage:"2 · CALL",title:"MiniMaxM3MLP.forward：调用 down_proj",location:"nvidia/model.py · L187",url:`${CODE_URL}#L187`,code:`x, _ = self.down_proj(x)`},
];
const DOWN_SYMBOLS: CodeSymbol[] = [MLP_SYMBOLS[2]];

const ATTENTION_SECTIONS: CodeSection[] = [
  {stage:"1 · PROJECT",title:"Attention.forward：packed QKV 投影",location:"nvidia/model.py · MiniMaxM3Attention.forward",url:CODE_URL,code:`qkv, _ = self.qkv_proj(hidden_states)
ops.fused_minimax_m3_qknorm_rope_kv_insert(
    qkv, positions, self.q_norm.weight, self.k_norm.weight,
    self.attn.kv_cache, ...
)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)`},
  {stage:"2 · ATTEND",title:"Q/K/V 进入 attention backend",location:"nvidia/model.py · MiniMaxM3Attention.forward",url:CODE_URL,code:`attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
return output`},
];

const ATTENTION_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.qkv_proj",resolvesTo:"QKVParallelLinear",meaning:"checkpoint 的 q_proj/k_proj/v_proj 在运行时合并为一次投影。"},
  {symbol:"fused_minimax_m3_qknorm_rope_kv_insert",resolvesTo:"Q/K RMSNorm + partial RoPE + KV cache insert",meaning:"positions、norm 权重和 cache 写入在融合 kernel 中一起消费。"},
  {symbol:"self.attn",resolvesTo:"vLLM Attention backend",meaning:"causal、长度与 block table 由 runtime metadata 提供，不要求物化稠密 mask。"},
];

export const QKV_INDEX_PROJECTION_SECTIONS: CodeSection[] = [
  {stage:"1 · FUSED LAYOUT",title:"vLLM：一次 GEMM 的五段输出布局",location:"linear.py · MinimaxM3QKVParallelLinearWithIndexer · L1319–1401",url:`${LINEAR_URL}#L1319-L1401`,code:`# One column-parallel GEMM emits:
# [q | k | v | index_q | index_k]
q = self.num_heads * self.head_size
kv = self.num_kv_heads * self.head_size
index_q = self.num_index_heads * self.index_head_size
index_k = self.index_head_size
self.output_sizes = [q * tp_size, kv * tp_size, kv * tp_size,
                     index_q * tp_size, index_k * tp_size]

ColumnParallelLinear.__init__(
    self, input_size=self.hidden_size,
    output_size=sum(self.output_sizes), gather_output=False,
)`},
  {stage:"2 · PROJECT",title:"vLLM：执行包含 Index Q/K 的 packed 投影",location:"nvidia/model.py · MiniMaxM3SparseAttention.forward · L565–581",url:`${CODE_URL}#L565-L581`,code:`# qkv 的名称沿用历史命名，实际包含五段：
# [q | k | v | index_q | index_k]
qkv, _ = self.qkv_proj(hidden_states)

# 第二返回值 _ 是 bias；bias=False，因此为 None。
# 五路投影结果全部位于 qkv。
# 后续 fused_minimax_m3_qknorm_rope_kv_insert
# 按五段偏移读取这个 packed tensor。`},
  {stage:"3 · TRANSFORMERS QKV",title:"Transformers：主 Q/K/V 的独立可读投影",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLAttention · L422–445",url:TRANSFORMERS_QKV_PROJECTION_URL,code:`self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)

query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)`},
  {stage:"4 · TRANSFORMERS INDEX",title:"Transformers：Index Q/K 的独立可读投影",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer · L538–563",url:TRANSFORMERS_INDEX_PROJECTION_URL,code:`self.q_proj = nn.Linear(hidden_size, index_n_heads * index_head_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, index_head_dim, bias=False)

idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)`},
];

export const QKV_INDEX_PROJECTION_SYMBOLS: CodeSymbol[] = [
  {symbol:"qkv",resolvesTo:"packed [Q | K | V | Qidx | Kidx]",meaning:"变量名叫 qkv，但在稀疏层中实际保存五路投影结果。"},
  {symbol:"bias / _",resolvesTo:"None",meaning:"线性层的第二返回值是 bias；这里 bias=False，与 Index 输出无关。"},
  {symbol:"index_q",resolvesTo:"Qidx · 4 个 index heads × 128",meaning:"用于计算稀疏块选择分数的 query 投影。"},
  {symbol:"index_k",resolvesTo:"Kidx · 1 个共享 index head × 128",meaning:"写入 Indexer cache，并与 Qidx 计算候选 block 分数。"},
];

const INDEX_NORM_ROPE_SECTIONS: CodeSection[] = [
  {stage:"TRANSFORMERS · PREPARE",title:"Index Q/K：投影后执行 Norm 与 RoPE",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer.forward · L559–565",url:TRANSFORMERS_INDEX_SELECTION_URL,code:`idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
idx_q = self.q_norm(idx_q).transpose(1, 2)
idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
idx_k = self.k_norm(idx_k).transpose(1, 2)
idx_q, idx_k = apply_rotary_pos_emb(
    idx_q, idx_k, cos[..., :self.head_dim], sin[..., :self.head_dim]
)`},
];

const INDEX_CACHE_SECTIONS: CodeSection[] = [
  {stage:"VLLM · CACHE SPEC",title:"独立的 key-only Index K side cache",location:"common/indexer.py · MiniMaxM3IndexerCache · L101–151",url:`${VLLM_INDEXER_URL}#L101-L151`,code:`class MiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    # one index-key vector per token; no value cache
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config):
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )`},
  {stage:"VLLM · OWNERSHIP",title:"Indexer 实现持有并注册自己的 Index K cache",location:"common/indexer.py · MiniMaxM3IndexerImpl.__init__ · L384–391",url:`${VLLM_INDEXER_URL}#L384-L391`,code:`self.index_cache = MiniMaxM3IndexerCache(
    head_dim=index_head_dim,
    prefix=f"{prefix}.index_cache",
    cache_config=cache_config,
    indexer_kv_dtype=indexer_kv_dtype,
    backend_cls=type(self).indexer_backend_cls,
)`},
];

const INDEX_SCORE_SECTIONS: CodeSection[] = [
  {stage:"VLLM · SCORE INPUT",title:"Index score kernel 读取完整 Index K cache",location:"common/indexer.py · MiniMaxM3IndexerTritonImpl.forward · L413–479",url:`${VLLM_INDEXER_URL}#L413-L479`,code:`index_md = attn_metadata[self.index_cache.prefix]
iq = index_query[:num_tokens].view(
    -1, self.num_index_heads, self.index_head_dim
)
index_k_cache = self.index_cache.kv_cache

score = minimax_m3_index_score(
    iq[nd:], index_k_cache, p.block_table,
    p.cu_seqlens_q, p.seq_lens, p.context_lens,
    p.max_query_len, p.max_seq_len, self.num_kv_heads,
)`},
  {stage:"TRANSFORMERS · SCORE",title:"每个 query/KV group 计算 Index token scores",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer.forward · L570–575",url:TRANSFORMERS_INDEX_SELECTION_URL,code:`k_len = idx_k.shape[2]
scores = torch.matmul(
    idx_q.float(), idx_k.float().transpose(-1, -2)
)  # [B, H_idx, S_q, S_k]`},
];

const INDEX_FUTURE_MASK_SECTIONS: CodeSection[] = [
  {stage:"TRANSFORMERS · INDEX MASK",title:"选块前先排除未来 key 与补齐槽",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer.forward · L575–579",url:TRANSFORMERS_INDEX_SELECTION_URL,code:`k_positions = torch.arange(k_len, device=idx_q.device)
token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
scores = scores.masked_fill(token_future, float("-inf"))
if pad:
    scores = F.pad(scores, (0, pad), value=float("-inf"))`},
];

const INDEX_SELECTION_SECTIONS: CodeSection[] = [
  {stage:"TRANSFORMERS · BLOCK MAX",title:"每 128 个 key 聚合成一个 block score",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer.forward · L580–582",url:TRANSFORMERS_INDEX_SELECTION_URL,code:`scores = scores.view(
    batch, self.num_heads, q_len, num_key_blocks, self.block_size
)
block_scores = scores.amax(dim=-1)`},
  {stage:"TRANSFORMERS · TOP-K",title:"保证 local block 可见后选择 Top-16",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLIndexer.forward · L584–597",url:TRANSFORMERS_INDEX_SELECTION_URL,code:`q_block = position_ids // self.block_size
local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
block_scores.scatter_(-1, local_idx, float("inf"))

topk_scores, block_indices = block_scores.topk(self.topk_blocks, dim=-1)
return block_indices.masked_fill(topk_scores == float("-inf"), -1)`},
];

const SPARSE_MASK_SECTIONS: CodeSection[] = [
  {stage:"TRANSFORMERS · BLOCK MASK",title:"把每组 block_indices 展开到 query heads",location:"modeling_minimax_m3_vl.py · build_block_mask · L599–625",url:TRANSFORMERS_BLOCK_MASK_URL,code:`bias.scatter_(-1, safe_block_indices, 0.0)
block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)
block_keep = block_keep.repeat_interleave(
    num_attention_heads // n_idx_heads, dim=1
)`},
  {stage:"TRANSFORMERS · COMPOSE",title:"再与 padding 或 causal token mask 合并",location:"modeling_minimax_m3_vl.py · build_block_mask · L626–635",url:TRANSFORMERS_BLOCK_MASK_URL,code:`if attention_mask is not None:
    padding_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    keep = block_keep & padding_mask
else:
    token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
    keep = block_keep & ~token_future
return torch.zeros(keep.shape, dtype=dtype).masked_fill(~keep, min_dtype)`},
];

const SPARSE_PAGED_ATTENTION_SECTIONS: CodeSection[] = [
  {stage:"VLLM · DIRECT PAGED READ",title:"Sparse Attention 直接消费 KV cache 与 Top-16 indices",location:"common/sparse_attention.py · MiniMaxM3SparseTritonImpl.forward · L418–470",url:`${VLLM_SPARSE_ATTENTION_URL}#L418-L470`,code:`topk = layer.topk_indices_buffer[:num_tokens].transpose(0, 1)
minimax_m3_sparse_attn(
    q[nd:],
    kv_cache,
    topk[:, nd:num_tokens, :],
    p.block_table,
    p.cu_seqlens_q,
    p.seq_lens,
    p.context_lens,
    p.max_query_len,
    self.num_kv_heads,
    self.scale,
    out[nd:],
)`},
  {stage:"TRANSFORMERS · DIRECT DISPATCH",title:"完整 K/V 与 block_indices 直接进入 Attention backend",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLAttention.forward · L456–487",url:TRANSFORMERS_SPARSE_ATTENTION_URL,code:`block_indices = self.indexer(
    hidden_states, position_embeddings, past_key_values, position_ids
)
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    block_indices=block_indices,
)`},
];

const INDEX_SELECTION_SYMBOLS: CodeSymbol[] = [
  {symbol:"H_idx",resolvesTo:"4 index heads = 4 KV groups",meaning:"每个 Index/KV group 独立为每个 query 选择 block。"},
  {symbol:"block_size",resolvesTo:"128 keys",meaning:"token scores 每 128 个 key 做一次 max pooling。"},
  {symbol:"block_indices",resolvesTo:"[B,4,S,16]",meaning:"每组、每个 query 的 Top-16 逻辑 key-block 索引；无效槽为 −1。"},
];

const INDEX_CACHE_SYMBOLS: CodeSymbol[] = [
  {symbol:"index_cache",resolvesTo:"MiniMaxM3IndexerCache",meaning:"与主 Paged KV Cache 分开的 Indexer side cache。"},
  {symbol:"kv_cache",resolvesTo:"key-only · one vector/token",meaning:"名字沿用 KV cache 接口，但这里只保存 Index K，不保存 Index V。"},
  {symbol:"indexer_kv_dtype",resolvesTo:"bf16 或 fp8_e4m3",meaning:"Index score 路径可独立选择 side-cache 存储精度。"},
];

const SPARSE_MASK_SYMBOLS: CodeSymbol[] = [
  {symbol:"block_keep",resolvesTo:"Top-16 block selection",meaning:"决定哪些 key blocks 属于当前 query/KV group 的候选集合。"},
  {symbol:"attention_mask",resolvesTo:"padding 或 causal token bounds",meaning:"在候选 blocks 内继续排除 padding 与未来 token。"},
];

const QK_NORM_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"为每个 Q/K head 创建 Gemma RMSNorm",location:"nvidia/model.py · MiniMaxM3Attention.__init__",url:`${CODE_URL}#L315-L317`,code:`self.q_norm = MiniMAXGemmaRMSNorm(
    self.head_dim, eps=config.rms_norm_eps
)
self.k_norm = MiniMAXGemmaRMSNorm(
    self.head_dim, eps=config.rms_norm_eps
)`},
  {stage:"2 · FUSED",title:"融合算子接收 Q/K Norm 权重与 ε",location:"nvidia/model.py · MiniMaxM3Attention.forward",url:`${CODE_URL}#L341-L354`,code:`ops.fused_minimax_m3_qknorm_rope_kv_insert(
    qkv,
    self.q_norm.weight,
    self.k_norm.weight,
    self.rotary_emb.cos_sin_cache,
    positions,
    self.num_heads,
    self.num_kv_heads,
    self.rotary_emb.rotary_dim,
    self.q_norm.variance_epsilon,
    kv_cache_dtype="auto",
)`},
];

const QK_NORM_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.q_norm / self.k_norm",resolvesTo:"per-head MiniMAXGemmaRMSNorm",meaning:"分别对每个 Q head 与 K head 的 128 维向量归一化。"},
  {symbol:"self.q_norm.weight / self.k_norm.weight",resolvesTo:"γQ / γK",meaning:"checkpoint 中独立保存的 Q/K Gemma RMSNorm 缩放权重。"},
  {symbol:"self.q_norm.variance_epsilon",resolvesTo:"ε = 10⁻⁶",meaning:"融合算子执行 Q/K RMSNorm 时使用的数值稳定项。"},
];

const VLLM_ROPE_SECTION: CodeSection = {
  stage:"1 · FUSED",
  title:"vLLM：融合 Q/K Norm + Partial RoPE",
  location:"nvidia/model.py · MiniMaxM3Attention.forward",
  url:CODE_URL,
  code:`ops.fused_minimax_m3_qknorm_rope_kv_insert(
    qkv,
    self.q_norm.weight,
    self.k_norm.weight,
    self.rotary_emb.cos_sin_cache,
    positions,
    self.num_heads,
    self.num_kv_heads,
    self.rotary_emb.rotary_dim,
    self.q_norm.variance_epsilon,
    kv_cache_dtype="auto",
)`,
};

const TRANSFORMERS_ROPE_SECTION: CodeSection = {
  stage:"2 · REFERENCE",
  title:"Transformers：Partial RoPE 可读实现",
  location:"modeling_minimax_m3_vl.py · apply_rotary_pos_emb · L365",
  url:TRANSFORMERS_MINIMAX_M3_URL,
  code:`rotary_dim = cos.shape[-1]
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
q = torch.cat([q_rot, q_pass], dim=-1)
k = torch.cat([k_rot, k_pass], dim=-1)`,
};

const TRANSFORMERS_ROPE_SYMBOL: CodeSymbol = {
  symbol:"rotary_dim / q_pass / k_pass",
  resolvesTo:"64 / 后 64 维 Q / 后 64 维 K",
  meaning:"Transformers 将参与旋转的前半段与直接保留的后半段显式拆开，便于核对 vLLM 融合算子的数学语义。",
};

const ROUTER_SECTIONS: CodeSection[] = [
  {stage:"1 · ROUTE",title:"MiniMaxM3MoE.forward：计算 router_logits",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`router_logits, _ = self.gate(hidden_states)`},
];

const ROUTER_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.gate",resolvesTo:"GateLinear",meaning:"对每个 token 做一次 FP32 线性投影。"},
  {symbol:"router_logits",resolvesTo:"[B,S,128] FP32",meaning:"这是 Python 层实际产生并传给 FusedMoE 的唯一 Router 输出。"},
];

const ROUTED_EXPERT_SECTIONS: CodeSection[] = [
  {stage:"1 · CALL",title:"FusedMoE 消费 router_logits",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)`},
  {stage:"2 · CONFIG",title:"FusedMoEFactory：Top-4 路由与专家配置",location:"nvidia/model.py · MiniMaxM3MoE.__init__",url:CODE_URL,code:`FusedMoEFactory(
    num_experts=128,
    top_k=4,
    hidden_size=6144,
    intermediate_size=3072,
    scoring_func="sigmoid",
    e_score_correction_bias=self.e_score_correction_bias,
    activation="swigluoai_uninterleave",
    routed_scaling_factor=2.0,
)`},
  {stage:"3 · TRANSFORMERS ROUTER",title:"MiniMaxM3VLTopKRouter：可读路由实现",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLTopKRouter.forward · L210–219",url:TRANSFORMERS_MOE_URL,code:`router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)
routing_weights = F.sigmoid(router_logits.float())
scores_for_choice = routing_weights + self.e_score_correction_bias
_, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
top_k_weights = routing_weights.gather(1, top_k_index)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)`},
  {stage:"4 · TRANSFORMERS EXPERTS",title:"MiniMaxM3VLExperts：专家计算与加权归并",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLExperts.forward · L178–200",url:TRANSFORMERS_MOE_URL,code:`top_k_pos, token_idx = torch.where(mask[expert_idx])
current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
final.index_add_(0, token_idx, current.to(final.dtype))`},
  {stage:"5 · TRANSFORMERS SCALE",title:"MiniMaxM3VLSparseMoeBlock：应用 routed scaling",location:"modeling_minimax_m3_vl.py · MiniMaxM3VLSparseMoeBlock.forward · L228–236",url:TRANSFORMERS_MOE_URL,code:`hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
hidden_states = hidden_states * self.routed_scaling_factor`},
];

const ROUTED_EXPERT_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.experts",resolvesTo:"FusedMoE",meaning:"消费 hidden_states 与 router_logits，并在内部完成 Top-4 路由、专家计算与加权归并。"},
  {symbol:"router_logits",resolvesTo:"FP32 Router 的 [B,S,128] 输出",meaning:"FusedMoE 根据它计算 sigmoid 分数、校正 Top-4 选择和混合权重。"},
  {symbol:"scores_for_choice",resolvesTo:"sigmoid(router_logits) + correction bias",meaning:"只用于决定 Top-4 expert；bias 不直接进入最终混合权重。"},
  {symbol:"top_k_index / top_k_weights",resolvesTo:"选中专家索引 / 归一化混合权重",meaning:"vLLM 中是 fused kernel 内部量；Transformers 参考实现将两者显式展开。"},
  {symbol:"activation",resolvesTo:"swigluoai_uninterleave",meaning:"routed-expert fused kernel 中的 SwiGLU-OAI 实现布局。"},
];

const SHARED_EXPERT_SECTIONS: CodeSection[] = [
  {stage:"1 · SHARED",title:"共享专家复用 MiniMaxM3MLP",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`shared_hidden_states = self.shared_experts(hidden_states)`},
];

const MOE_SUM_SECTIONS: CodeSection[] = [
  {stage:"1 · ADD",title:"Routed 与 Shared 输出相加",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`final_hidden_states = final_hidden_states + shared_hidden_states
return final_hidden_states.view(num_tokens, hidden_dim)`},
];

export const CODE_BY_ID: Record<string, CodeDetail> = {};
for(const id of ["d-norm","d-postnorm","s-norm","s-postnorm"]) CODE_BY_ID[id]={sections:NORM_SECTIONS,symbols:NORM_SYMBOLS};
CODE_BY_ID["d-gateup"]={sections:GATE_UP_SECTIONS,symbols:GATE_UP_SYMBOLS};
CODE_BY_ID["d-swiglu"]={sections:SWIGLU_SECTIONS,symbols:SWIGLU_SYMBOLS};
CODE_BY_ID["d-down"]={sections:DOWN_SECTIONS,symbols:DOWN_SYMBOLS};
for(const id of ["d-add2","s-addout"]) CODE_BY_ID[id]={sections:RESIDUAL_MERGE_SECTIONS,symbols:RESIDUAL_MERGE_SYMBOLS};
for(const id of ["d-add1","s-addattn"]) CODE_BY_ID[id]={sections:ATTENTION_RESIDUAL_SECTIONS,symbols:ATTENTION_RESIDUAL_SYMBOLS};
for(const id of ["d-qkv","d-split","d-ropeq","d-ropek","d-cache","d-qk","d-scale","d-mask","d-softmax","d-pv","d-oproj","s-split","s-rope","s-cache","s-qk","s-scale","s-mask","s-softmax","s-pv","s-oproj"]) CODE_BY_ID[id]={sections:ATTENTION_SECTIONS,symbols:ATTENTION_SYMBOLS};
for(const id of ["d-qnorm","d-knorm","s-mainnorm"]) CODE_BY_ID[id]={sections:QK_NORM_SECTIONS,symbols:QK_NORM_SYMBOLS};
for(const id of ["d-ropeq","d-ropek","s-rope"]) CODE_BY_ID[id]={sections:[VLLM_ROPE_SECTION,TRANSFORMERS_ROPE_SECTION],symbols:[ATTENTION_SYMBOLS[1],TRANSFORMERS_ROPE_SYMBOL]};
CODE_BY_ID["s-idxnorm"]={sections:INDEX_NORM_ROPE_SECTIONS,symbols:INDEX_SELECTION_SYMBOLS};
CODE_BY_ID["s-idxcache"]={sections:INDEX_CACHE_SECTIONS,symbols:INDEX_CACHE_SYMBOLS};
CODE_BY_ID["s-idxscore"]={sections:INDEX_SCORE_SECTIONS,symbols:INDEX_SELECTION_SYMBOLS};
CODE_BY_ID["s-idxmask"]={sections:INDEX_FUTURE_MASK_SECTIONS,symbols:INDEX_SELECTION_SYMBOLS};
for(const id of ["s-blockmax","s-topk"]) CODE_BY_ID[id]={sections:INDEX_SELECTION_SECTIONS,symbols:INDEX_SELECTION_SYMBOLS};
for(const id of ["s-qk","s-pv"]) CODE_BY_ID[id]={sections:SPARSE_PAGED_ATTENTION_SECTIONS,symbols:ATTENTION_SYMBOLS};
CODE_BY_ID["s-mask"]={sections:SPARSE_MASK_SECTIONS,symbols:SPARSE_MASK_SYMBOLS};
CODE_BY_ID["s-router"]={sections:ROUTER_SECTIONS,symbols:ROUTER_SYMBOLS};
CODE_BY_ID["s-experts"]={sections:ROUTED_EXPERT_SECTIONS,symbols:ROUTED_EXPERT_SYMBOLS};
CODE_BY_ID["s-shared"]={sections:[...SHARED_EXPERT_SECTIONS,...MLP_SECTIONS],symbols:MLP_SYMBOLS};
CODE_BY_ID["s-sum"]={sections:MOE_SUM_SECTIONS,symbols:[]};

export const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
  "d-input":[{kind:"external",label:"Xₗ · hidden_states",shape:"[B,S,6144]",from:"上一 decoder layer；L0 时来自 embedding fusion"}],
  "s-input":[{kind:"external",label:"Xₗ · hidden_states",shape:"[B,S,6144]",from:"上一 decoder layer 输出"}],
  "d-position":[{kind:"external",label:"num_computed_tokens + query offsets",shape:"[B] + [Nq]",from:"vLLM GPUModelRunner 请求调度状态"}],
  "s-position":[{kind:"external",label:"num_computed_tokens + query offsets",shape:"[B] + [Nq]",from:"vLLM GPUModelRunner 请求调度状态"}],
  "d-attnmeta":[{kind:"external",label:"query_start_loc · seq_lens · causal",shape:"[B+1] + [B] + bool",from:"vLLM CommonAttentionMetadata"}],
  "s-attnmeta":[{kind:"external",label:"query_start_loc · seq_lens · causal",shape:"[B+1] + [B] + bool",from:"vLLM CommonAttentionMetadata"}],
  "d-slots":[{kind:"external",label:"positions + block_table",shape:"[Nq] + [B,Nblocks]",from:"runner positions 与 KV cache manager"}],
  "s-slots":[{kind:"external",label:"positions + block_table",shape:"[Nq] + [B,Nblocks]",from:"runner positions 与 KV cache manager"}],
  "d-ropeq":[{kind:"upstream",label:"Q̃",shape:"[B,64,S,128]",from:"Q RMSNorm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "d-ropek":[{kind:"upstream",label:"K̃",shape:"[B,4,S,128]",from:"K RMSNorm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "d-cache":[{kind:"upstream",label:"Kᵣ",shape:"[B,4,S,128]",from:"Partial RoPE (K) 输出"},{kind:"upstream",label:"V",shape:"[B,4,S,128]",from:"Split Q / K / V 输出"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "d-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE (Q) 输出"},{kind:"upstream",label:"visible K (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "d-mask":[{kind:"upstream",label:"scaled local scores",shape:"[B,64/TP,S,T]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "d-pv":[{kind:"upstream",label:"local attention probability P",shape:"[B,64/TP,S,T]",from:"Softmax 输出"},{kind:"upstream",label:"visible V (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "s-rope":[{kind:"upstream",label:"Q̃ · K̃",shape:"Q/K unchanged",from:"Main Q/K Norm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "s-cache":[{kind:"upstream",label:"Kᵣ · V",shape:"KV pages",from:"Partial RoPE 与 Split 5 outputs"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "s-idxnorm":[{kind:"upstream",label:"Qidx · Kidx",shape:"[B,4,S,128] · [B,1,T,128]",from:"Split 5 outputs"},{kind:"external",label:"position embeddings",shape:"cos · sin",from:"Build Position IDs / RoPE cache"}],
  "s-idxcache":[{kind:"upstream",label:"current rotated Index K",shape:"[B,S,128]",from:"Index Q/K Gemma RMSNorm + RoPE 输出"},{kind:"external",label:"index slot_mapping",shape:"[Nq]",from:"Indexer metadata builder"}],
  "s-idxscore":[{kind:"upstream",label:"Index Q query",shape:"[B,4,S,128]",from:"Index Q/K Gemma RMSNorm + RoPE 输出"},{kind:"upstream",label:"cached Index K history",shape:"[B,T,128]",from:"Index K Cache 输出"}],
  "s-idxmask":[{kind:"upstream",label:"Index token scores",shape:"[B,4,S,T]",from:"Index Q × Kᵀ 输出"},{kind:"external",label:"position_ids",shape:"[B,S]",from:"当前 query/key 的因果位置"}],
  "s-blockmax":[{kind:"upstream",label:"causal Index scores",shape:"[B,4,S,T]",from:"Mask Future Index Keys 输出"}],
  "s-topk":[{kind:"upstream",label:"local block scores",shape:"[B,max(1,4/TP),S,Nblocks]",from:"Block Max 输出"},{kind:"external",label:"local / init priority",shape:"logical block flags",from:"Indexer 配置：local_blocks=1, init_blocks=0"}],
  "s-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE 输出"},{kind:"upstream",label:"paged K",shape:"KV pages",from:"Paged KV Cache 输出"},{kind:"upstream",label:"block_indices",shape:"[B,4,S,16]",from:"Top-16 Blocks 输出"}],
  "s-mask":[{kind:"upstream",label:"scaled local selected scores",shape:"[B,64/TP,S,Ksel]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "s-pv":[{kind:"upstream",label:"local selected attention P",shape:"[B,64/TP,S,Ksel]",from:"Softmax 输出"},{kind:"upstream",label:"paged V",shape:"KV pages",from:"Paged KV Cache 输出；按同一 Top-16 顺序读取"}],
  "s-router":[{kind:"upstream",label:"post-attn normalized hidden Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出"}],
  "s-experts":[{kind:"upstream",label:"normalized hidden + router logits",shape:"[B,S,6144] + [B,S,128]",from:"Post-attn RMSNorm 与 FP32 Router 输出"}],
  "s-shared":[{kind:"upstream",label:"all normalized tokens Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出；不经过 Top-K"}],
  "s-sum":[{kind:"upstream",label:"weighted routed output",shape:"[B,S,6144]",from:"Fused Top-4 Routing + Experts 输出"},{kind:"upstream",label:"shared output",shape:"[B,S,6144]",from:"Shared Expert ×1 输出"}],
  "d-add2":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Yffn · FFN branch",shape:"[B,S,6144]",from:"Down Projection 输出"}],
  "s-addout":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Ymoe · MoE branch",shape:"[B,S,6144]",from:"Add Routed + Shared 输出"}],
  "d-add1":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn · attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
  "s-addattn":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn · sparse attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
};

export const NEXT_BY_ID: Record<string,string> = {
  "d-input":"Gemma RMSNorm","d-position":"Partial RoPE (Q/K)","d-attnmeta":"Apply Causal / Pad Bounds","d-slots":"Paged KV Cache","d-norm":"QKV Projection","d-qkv":"Split Q / K / V","d-split":"Q RMSNorm · K RMSNorm · Paged KV Cache","d-qnorm":"Partial RoPE (Q)","d-knorm":"Partial RoPE (K)","d-ropeq":"Q × Kᵀ","d-ropek":"Paged KV Cache","d-cache":"Q × Kᵀ · P × V","d-qk":"Scale 1/√128","d-scale":"Apply Causal / Pad Bounds","d-mask":"Softmax","d-softmax":"P × V","d-pv":"O Projection","d-oproj":"Attention Residual Merge","d-add1":"Post-attn Gemma RMSNorm","d-postnorm":"Gate + Up Projection","d-gateup":"Split Gate / Up","d-gatesplit":"SwiGLU-OAI","d-swiglu":"Down Projection","d-down":"Decoder Layer Residual Merge","d-add2":"下一 decoder layer / Final Norm",
  "s-input":"Gemma RMSNorm","s-position":"Partial RoPE","s-attnmeta":"Indexer 与 Sparse Attention mask","s-slots":"Paged KV Cache","s-norm":"QKV + Index Projection","s-packed":"Split 5 outputs","s-split":"Index Q/K Gemma RMSNorm + RoPE · Main Q/K Gemma RMSNorm · Paged KV Cache","s-idxnorm":"Index Q query · Index K Cache","s-idxcache":"Index Q × cached Kᵀ","s-idxscore":"Mask Future Index Keys","s-idxmask":"Block Max","s-blockmax":"Top-K_block Blocks","s-topk":"Q × paged Kᵀ · Top-K_block","s-mainnorm":"Partial RoPE","s-rope":"Paged KV Cache · Q × paged Kᵀ · Top-K_block","s-cache":"Q × paged Kᵀ · Top-K_block · P × paged V","s-qk":"Scale 1/√Dₕ","s-scale":"Apply Token Causal / Pad Mask","s-mask":"Softmax","s-softmax":"P × paged V · same Top-K_block","s-pv":"O Projection","s-oproj":"Attention Residual Merge","s-addattn":"Post-attn Gemma RMSNorm","s-postnorm":"FP32 Router Logits · Fused Top-4 Routing + Experts · Shared Expert","s-router":"router_logits → Fused Top-4 Routing + Experts","s-experts":"Add Routed + Shared","s-shared":"Add Routed + Shared","s-sum":"Decoder Layer Residual Merge","s-addout":"下一 decoder layer / Final Norm",
};
