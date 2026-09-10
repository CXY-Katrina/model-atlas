import type { CodeSection, CodeSymbol, CodeDetail, IoBinding } from "../types";
import { CODE_URL, ATTENTION_URL, MOE_URL, MTP_URL, DECODER_IHC_FORWARD_URL, HC_PRE_URL, HC_POST_URL, HC_HEAD_URL, MOE_FACTORY_URL, DENSE_FF_URL, MTP_FORWARD_URL, MODEL_FORWARD_URL, INDEXER_PREPARE_URL, SINK_MATH_URL, MLA_ATTENTION_URL, V1_FLASHMLA_SPARSE_URL } from "./sources";

const HC_PRE_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"HYV4HCPreLayer.forward:归约 + post 门",location:"nvidia/hc.py · L97–139",url:HC_PRE_URL,code:`shape = x.size()  # [num_tokens, hc, d]
hc = self.hc_mult
hc_eps = self.hc_eps
x_flat = x.flatten(1).float()  # [num_tokens, hc*d]
rsqrt = torch.rsqrt(
    x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
)
mixes = self.hc_fn(x_flat)[0] * rsqrt  # [num_tokens, 2*hc]
pre_raw = mixes[..., :hc]
post_raw = mixes[..., hc : 2 * hc]
pre = torch.sigmoid(
    pre_raw * self.hc_scale[0].float() + self.hc_base[:hc].float()
) + hc_eps
post = (
    self.magnitude
    * torch.sigmoid(
        post_raw * self.hc_scale[1].float() + self.hc_base[hc : 2 * hc].float()
    )
    + hc_eps
)
y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
return y.to(x.dtype), post`},
  {stage:"2 · CALL",title:"HYV4DecoderLayer._forward_ihc:两个子块各一次",location:"nvidia/model.py · L187–209",url:DECODER_IHC_FORWARD_URL,code:`hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)`,
  },
];

const HC_PRE_SYMBOLS: CodeSymbol[] = [
  {symbol:"x / x_flat",resolvesTo:"4 通道输入与展平",meaning:"[T,4,6144] → [T,24576],FP32 计算。"},
  {symbol:"hc_scale / hc_base",resolvesTo:"门仿射参数",meaning:"scale[2] 初始 0.01;base[:4]=−log 3、base[4:8]=0。"},
  {symbol:"magnitude",resolvesTo:"hc_magnitude = 2.0",meaning:"post 门的幅度,乘在 σ 之后。"},
  {symbol:"self.layernorm_epsilon",resolvesTo:"1e−5",meaning:"取自 rms_norm_eps,与子块内 RMSNorm 数值一致。"},
];

const HC_POST_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"HYV4HCPostLayer.forward:无参数散回",location:"nvidia/hc.py · L163–186",url:HC_POST_URL,code:`dtype = x.dtype
x = x.float(); residual = residual.float(); post = post.float()
post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)  # [num_tokens, hc, d]
y = post_gated + residual
return y.to(dtype)`},
];

const HC_POST_SYMBOLS: CodeSymbol[] = [
  {symbol:"post",resolvesTo:"pre 层输出的 post 门",meaning:"2·σ(·)+ε,幅度 2.0。"},
  {symbol:"residual",resolvesTo:"本子块入口的 4 通道 hidden",meaning:"pre 返回的第三个值,未作修改。"},
];

const HC_HEAD_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"HYV4HCHeadLayer.forward:4 通道合并",location:"nvidia/hc.py · L251–277",url:HC_HEAD_URL,code:`shape, x_dtype = x.size(), x.dtype
x = x.flatten(1).float()  # [num_tokens, hc*d]
rsqrt = torch.rsqrt(
    x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
)
mixes = self.hc_head_fn(x)[0] * rsqrt  # [num_tokens, hc]
pre = (
    torch.sigmoid(
        mixes * self.hc_head_scale.float() + self.hc_head_base.float()
    )
    + self.hc_eps
)
y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
return y.to(x_dtype)`},
  {stage:"2 · CALL",title:"HYV4Model.forward:hc_head 在 final norm 之前",location:"nvidia/model.py · L388–393",url:MODEL_FORWARD_URL,code:`if self.enable_ihc:
    hidden_states = self.hc_head(hidden_states)
else:
    hidden_states = hidden_states + residual
return self.norm(hidden_states)`},
];

const NORM_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"RMSNorm(hidden, eps=rms_norm_eps)",location:"nvidia/model.py · HYV4DecoderLayer.__init__",url:CODE_URL,code:`self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)`},
];

const NORM_SYMBOLS: CodeSymbol[] = [
  {symbol:"config.rms_norm_eps",resolvesTo:"ε = 1e−5",meaning:"checkpoint rms_norm_eps。"},
  {symbol:"weight",resolvesTo:"γ",meaning:"checkpoint 中 input_layernorm.weight / post_attention_layernorm.weight。"},
];

const APROJ_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"融合 down 投影的构造",location:"nvidia/attention.py · L359–371",url:ATTENTION_URL,code:`self.fused_qkv_a_proj = MergedColumnParallelLinear(
    self.hidden_size,
    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
    bias=False, quant_config=quant_config, disable_tp=True,
    prefix=f"{prefix}.fused_qkv_a_proj",)`},
  {stage:"2 · FORWARD",title:"forward:切分三个输出",location:"nvidia/attention.py · L669–683",url:ATTENTION_URL,code:`qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
q_c, kv_lora = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
q_c = self.q_a_layernorm(q_c)
q = self.q_b_proj(q_c)[0]
kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_c_normed = self.kv_a_layernorm(kv_c)`},
  {stage:"3 · LOAD",title:"checkpoint 名称 → 融合权重",location:"nvidia/model.py · stacked mapping",url:CODE_URL,code:`".q_a_proj": (".fused_qkv_a_proj", 0),
".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),`},
];

const APROJ_SYMBOLS: CodeSymbol[] = [
  {symbol:"q_lora_rank",resolvesTo:"2048",meaning:"q lora 宽度。"},
  {symbol:"kv_lora_rank + qk_rope_head_dim",resolvesTo:"512 + 64 = 576",meaning:"第二个 shard 的输出宽度。"},
  {symbol:"disable_tp",resolvesTo:"TP 复制",meaning:"down 投影整层复制,不做张量并行切分。"},
];

const ROPE_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"两条 RoPE 都是 interleaved",location:"nvidia/attention.py · L416–435",url:ATTENTION_URL,code:`self.rotary_emb = get_rope(
    qk_rope_head_dim, max_position=max_position_embeddings,
    rope_parameters=config.rope_parameters, is_neox_style=False,
)
# The checkpoint stores indexer q_pe/k_pe in interleaved (Megatron/PTM)
# layout ... Using NeoX here loses the relative-position dependence and
# corrupts the DSA top-k selection.`},
  {stage:"2 · FORWARD",title:"只旋转 rope 段",location:"nvidia/attention.py · L685–690",url:ATTENTION_URL,code:`q = q.view(-1, self.num_local_heads, self.qk_head_dim)
k_pe = k_pe.unsqueeze(1)
q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
    positions, q[..., self.qk_nope_head_dim:], k_pe)`},
];

const ROPE_SYMBOLS: CodeSymbol[] = [
  {symbol:"is_neox_style=False",resolvesTo:"interleaved",meaning:"checkpoint 为 Megatron/PTM 交错布局;NeoX 会破坏 top-k。"},
  {symbol:"q[..., 192:]",resolvesTo:"每 head 的 rope 64 维",meaning:"nope 段不旋转。"},
];

const INDEXER_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"wq_b 与融合 wk_weights_proj",location:"nvidia/attention.py · L150–168",url:ATTENTION_URL,code:`self.wq_b = ReplicatedLinear(self.q_lora_rank, self.head_dim * self.n_head, ...)
# Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
self.wk_weights_proj = MergedColumnParallelLinear(
    hidden_size, [self.head_dim, self.n_head], bias=False,
    quant_config=None, disable_tp=True, ...)
self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
self.softmax_scale = self.head_dim ** -0.5`},
  {stage:"2 · PREPARE",title:"prepare_inputs:indexer RoPE + 量化与权重折叠",location:"nvidia/attention.py · L212–263(节选)",url:INDEXER_PREPARE_URL,code:`q, _ = self.wq_b(qr)
q = q.view(-1, self.n_head, self.head_dim)
q_nope, q_pe = torch.split(q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
kw, _ = self.wk_weights_proj(hidden_states)
k = self.k_norm(kw[:, :self.head_dim])
weights = kw[:, self.head_dim:]
k_nope, k_pe = torch.split(k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))  # indexer 独立 RoPE
q = torch.cat([q_nope, q_pe.reshape(-1, self.n_head, self.rope_dim)], dim=-1)
k = torch.cat([k_nope, k_pe.reshape(-1, 1, self.rope_dim).squeeze(-2)], dim=-1)
q_fp8, q_scale = per_token_group_quant_fp8(
    q.view(-1, self.head_dim), self.quant_block_size,
    column_major_scales=False, use_ue8m0=self.scale_fmt is not None)
weights = (weights.unsqueeze(-1) * q_scale.view(-1, self.n_head, 1)
           * self.softmax_scale * self.n_head ** -0.5)
return hidden_states, q_fp8, k, weights.squeeze(-1)`},
  {stage:"3 · SELECT",title:"_indexer_and_attn:full 与 shared 的分岔",location:"nvidia/attention.py · L736–765",url:ATTENTION_URL,code:`if self.indexer is not None and self.is_sparse and not self.skip_topk:
    self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
out.copy_(self.mla_attn(q, kv_c_normed, k_pe, output_shape=out.shape))`},
];

const INDEXER_SYMBOLS: CodeSymbol[] = [
  {symbol:"skip_topk",resolvesTo:"shared 层标记",meaning:"由 indexer_types 推导;57 层为 true,MTP 层恒 false。"},
  {symbol:"softmax_scale",resolvesTo:"128^(-1/2)",meaning:"再乘 N_idx^(-1/2) 折进 w。"},
  {symbol:"quant_block_size",resolvesTo:"128",meaning:"q 的 FP8 组量化粒度,scale 格式 ue8m0。"},
];

const SINK_SECTIONS: CodeSection[] = [
  {stage:"1 · PARAM",title:"learnable_sink_param 的创建",location:"nvidia/attention.py · L482–518",url:ATTENTION_URL,code:`if self.learnable_sink:
    sink_backend = self._resolve_sink_backend(kv_cache_dtype)
    self.learnable_sink_param = nn.Parameter(torch.empty(
        self.num_local_heads, dtype=torch.float32))
    sinks = self.learnable_sink_param
    self._force_sparse_mqa()
extra_impl_args = {} if sinks is None else {"sinks": sinks}`},
  {stage:"2 · KERNEL",title:"FlashMLA sparse:折进 softmax 分母",location:"nvidia/flashmla_sparse.py · HYV4FlashMLASparseImpl",url:SINK_MATH_URL,code:`The sink enters as the 'sinks' impl kwarg ... consumed by the
FlashMLA kernels, which fold it into the softmax denominator:
out *= exp(lse) / (exp(lse) + exp(sink))`},
];

const SINK_SYMBOLS: CodeSymbol[] = [
  {symbol:"learnable_sink_param",resolvesTo:"每 head FP32 sink",meaning:"checkpoint 初始 0;TP 下保存本地分片。"},
  {symbol:"_force_sparse_mqa",resolvesTo:"prefill 也走稀疏",meaning:"dense prefill 后端不能应用 sink,部分应用会破坏架构一致性。"},
];

const GATE_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"gated MLA(elementwise)",location:"nvidia/attention.py · L710–733",url:ATTENTION_URL,code:`gate_score = self.linear_gate(hidden_states)[0]
attn_out = attn_out * torch.sigmoid(gate_score)
out, _ = self.o_proj(attn_out)
return out`},
  {stage:"2 · INIT",title:"elementwise 门的宽度 = v_head_dim/head",location:"nvidia/attention.py · L455–470",url:ATTENTION_URL,code:`if config.gating_type == "elementwise":
    self.gate_projection_size_per_head = self.v_head_dim
self.linear_gate = ColumnParallelLinear(
    self.hidden_size, self.num_heads * self.gate_projection_size_per_head, ...)`,
  },
];

const GATE_SYMBOLS: CodeSymbol[] = [
  {symbol:"gating_type",resolvesTo:"elementwise",meaning:"checkpoint 配置;每 head 256 通道与 V 对齐。"},
  {symbol:"hidden_states",resolvesTo:"子块输入",meaning:"门从注意力子块的输入投影,不是从注意力输出。"},
];

const MOE_ROUTER_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"FP32 gate 与 expert_bias",location:"nvidia/moe.py · L122–146",url:MOE_URL,code:`self.gate = GateLinear(config.hidden_size, config.num_experts,
    bias=False, out_dtype=torch.float32, params_dtype=torch.float32, ...)
self.expert_bias = nn.Parameter(torch.empty(config.num_experts, dtype=torch.float32))`},
  {stage:"2 · LOAD",title:"checkpoint 名称映射",location:"nvidia/model.py · load_weights",url:CODE_URL,code:`if "gate.e_score_correction_bias" in name:
    name = name.replace("gate.e_score_correction_bias", "expert_bias")`},
];

const MOE_ROUTER_SYMBOLS: CodeSymbol[] = [
  {symbol:"expert_bias",resolvesTo:"e_score_correction_bias",meaning:"只影响 Top-8 选择,不进混合权重。"},
  {symbol:"params_dtype fp32",resolvesTo:"router 精度",meaning:"logits 全程 FP32。"},
];

const MOE_EXPERT_SECTIONS: CodeSection[] = [
  {stage:"1 · FACTORY",title:"FusedMoEFactory 参数",location:"nvidia/moe.py · L152–170",url:MOE_FACTORY_URL,code:`self.experts = FusedMoEFactory(
    num_experts=self.n_routed_experts, top_k=top_k,
    hidden_size=config.hidden_size, intermediate_size=intermediate_size,
    renormalize=config.route_norm, scoring_func="sigmoid",
    use_grouped_topk=True, num_expert_group=1, topk_group=1,
    routed_scaling_factor=router_scaling_factor,
    e_score_correction_bias=self.expert_bias,
    shared_experts=self.shared_experts, swiglu_limit=moe_swiglu_limit,)`},
  {stage:"2 · CLAMP",title:"clamped SwiGLU 只作用 routed",location:"vllm FusedMoE(swiglu_limit=10)",url:MOE_FACTORY_URL,code:`gate = clamp(gate, max=limit)
up = clamp(up, -limit, limit)
output = silu(gate) * up
# Dense layers and shared experts are NOT clamped.`},
  {stage:"3 · LOAD",title:"融合 experts 张量切 w1/w3",location:"nvidia/model.py · load_weights",url:CODE_URL,code:`if "experts.gate_up_proj" in name:
    chunks = loaded_weight.chunk(2, dim=-2)
    success_w1 = self.load_fused_expert_weights(..., chunks[0], "w1", ...)
    success_w3 = self.load_fused_expert_weights(..., chunks[1], "w3", ...)`,
  },
];

const MOE_EXPERT_SYMBOLS: CodeSymbol[] = [
  {symbol:"swiglu_limit",resolvesTo:"10.0",meaning:"gate 截 max,up 截 ±10;仅 routed experts。"},
  {symbol:"routed_scaling_factor",resolvesTo:"2.827",meaning:"归一化混合权重后整体相乘。"},
  {symbol:"num_expert_group/topk_group",resolvesTo:"1 / 1",meaning:"无分组约束,等价朴素 Top-8。"},
];

const SHARED_SECTIONS: CodeSection[] = [
  {stage:"1 · INIT",title:"shared experts 用普通 HYV4FeedForward",location:"nvidia/moe.py · L131–142",url:MOE_URL,code:`self.shared_experts = HYV4FeedForward(
    hidden_size=config.hidden_size,
    intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
    hidden_act=config.hidden_act, quant_config=quant_config,
    prefix=f"{prefix}.shared_experts", reduce_results=False)`},
  {stage:"2 · FF",title:"HYV4FeedForward.forward",location:"nvidia/moe.py · L62–66",url:DENSE_FF_URL,code:`def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    out = self.act_fn(gate_up)
    out, _ = self.down_proj(out)
    return out`},
];

const SUM_SECTIONS: CodeSection[] = [
  {stage:"1 · ADD",title:"routed 与 shared 相加(经 FusedMoE shared_experts 通道)",location:"nvidia/moe.py · forward",url:MOE_URL,code:`final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=router_logits)
return final_hidden_states.view(orig_shape)`},
];

const CACHE_SECTIONS: CodeSection[] = [
  {stage:"1 · CANONICALIZE",title:"_canonicalize_sparse_mla_kv_cache_dtype:量化请求升格 fp8_ds_mla",location:"vllm/model_executor/layers/attention/mla_attention.py · L349–362",url:MLA_ATTENTION_URL,code:`backend_name = attn_backend.get_name()
if backend_name == "FLASHMLA_SPARSE" and is_quantized_kv_cache(kv_cache_dtype):
    return "fp8_ds_mla"
if backend_name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in (
    "auto", "fp8", "fp8_e4m3",
):
    return "fp8_ds_mla"
return kv_cache_dtype`},
  {stage:"2 · BACKEND",title:"后端据此选择 fp8 内核",location:"vllm/v1/attention/backends/mla/flashmla_sparse.py · L261",url:V1_FLASHMLA_SPARSE_URL,code:`self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"`},
];

const DENSE_FF_SECTIONS: CodeSection[] = [
  {stage:"1 · SELECT",title:"L0 dense / L1–77 MoE 的选择",location:"nvidia/model.py · HYV4DecoderLayer.__init__",url:CODE_URL,code:`if config.mlp_layer_types[layer_idx] == "dense":
    self.mlp = HYV4FeedForward(..., intermediate_size=config.intermediate_size, ...)
else:
    self.mlp = HYV4MoEFused(config=config, quant_config=quant_config, ...)`},
  {stage:"2 · FF",title:"HYV4FeedForward.forward(无 clamp)",location:"nvidia/moe.py · L62–66",url:DENSE_FF_URL,code:`gate_up, _ = self.gate_up_proj(x)
out = self.act_fn(gate_up)
out, _ = self.down_proj(out)`},
];

const MTP_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"HYV4MultiTokenPredictorLayer.forward",location:"nvidia/mtp.py · L369–390",url:MTP_FORWARD_URL,code:`inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
hidden_states, _ = self.final_layernorm(hidden_states, residual)
return hidden_states`},
  {stage:"2 · INIT",title:"MTP 块关闭 iHC",location:"nvidia/mtp.py · L339–351",url:MTP_URL,code:`self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
mtp_config.enable_ihc = False  # checkpoint 无 MTP iHC 权重`},
];

export const CODE_BY_ID: Record<string, CodeDetail> = {};
for(const id of ["hcpre-attn","hcpre-mlp"]) CODE_BY_ID[id]={sections:HC_PRE_SECTIONS,symbols:HC_PRE_SYMBOLS};
for(const id of ["hcpost1","hcpost2"]) CODE_BY_ID[id]={sections:HC_POST_SECTIONS,symbols:HC_POST_SYMBOLS};
CODE_BY_ID["hc-head"]={sections:HC_HEAD_SECTIONS,symbols:[]};
for(const id of ["norm1","norm2"]) CODE_BY_ID[id]={sections:NORM_SECTIONS,symbols:NORM_SYMBOLS};
CODE_BY_ID["aproj"]={sections:APROJ_SECTIONS,symbols:APROJ_SYMBOLS};
CODE_BY_ID["rope"]={sections:ROPE_SECTIONS,symbols:ROPE_SYMBOLS};
for(const id of ["iwqb","iwk","ikn","iquant","iscore"]) CODE_BY_ID[id]={sections:INDEXER_SECTIONS,symbols:INDEXER_SYMBOLS};
CODE_BY_ID["sink"]={sections:SINK_SECTIONS,symbols:SINK_SYMBOLS};
CODE_BY_ID["gate"]={sections:GATE_SECTIONS,symbols:GATE_SYMBOLS};
CODE_BY_ID["router"]={sections:MOE_ROUTER_SECTIONS,symbols:MOE_ROUTER_SYMBOLS};
CODE_BY_ID["select"]={sections:MOE_EXPERT_SECTIONS,symbols:MOE_EXPERT_SYMBOLS};
CODE_BY_ID["experts"]={sections:MOE_EXPERT_SECTIONS,symbols:MOE_EXPERT_SYMBOLS};
CODE_BY_ID["shared"]={sections:[...SHARED_SECTIONS,...DENSE_FF_SECTIONS],symbols:[]};
CODE_BY_ID["sum"]={sections:SUM_SECTIONS,symbols:[]};
CODE_BY_ID["cache"]={sections:CACHE_SECTIONS,symbols:[]};
for(const id of ["gateup","act","down"]) CODE_BY_ID[id]={sections:DENSE_FF_SECTIONS,symbols:[]};
CODE_BY_ID["mtp"]={sections:MTP_SECTIONS,symbols:[]};

export const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
  "input":[{kind:"external",label:"Xₗ · hidden_states",shape:"[T,6144] → [T,4,6144]",from:"上一 decoder layer(4 通道);L0 来自 embedding 广播"}],
  "hcpre-attn":[{kind:"upstream",label:"4 通道 hidden",shape:"[T,4,6144]",from:"prepare_input 输出"}],
  "hcpre-mlp":[{kind:"upstream",label:"4 通道 hidden",shape:"[T,4,6144]",from:"hc_post · attn 输出"}],
  "norm1":[{kind:"upstream",label:"归约后 hidden",shape:"[T,6144]",from:"hc_pre · attn 输出"}],
  "norm2":[{kind:"upstream",label:"归约后 hidden",shape:"[T,6144]",from:"hc_pre · mlp 输出"}],
  "hcpost1":[{kind:"upstream",label:"Y_attn",shape:"[T,6144]",from:"o_proj 输出"},{kind:"upstream",label:"post 门",shape:"[T,4]",from:"hc_pre · attn 输出"},{kind:"upstream",label:"4 通道残差",shape:"[T,4,6144]",from:"本子块入口"}],
  "hcpost2":[{kind:"upstream",label:"Y_ffn / Y_moe",shape:"[T,6144]",from:"FFN 子块输出"},{kind:"upstream",label:"post 门",shape:"[T,4]",from:"hc_pre · mlp 输出"},{kind:"upstream",label:"4 通道残差",shape:"[T,4,6144]",from:"hc_post · attn 输出"}],
  "aproj":[{kind:"upstream",label:"X̂",shape:"[T,6144]",from:"input_layernorm 输出"}],
  "qan":[{kind:"upstream",label:"q_a",shape:"[T,2048]",from:"fused_qkv_a_proj shard 0"}],
  "qb":[{kind:"upstream",label:"q̃",shape:"[T,2048]",from:"q_a_layernorm 输出"}],
  "kvn":[{kind:"upstream",label:"kv_lora 压缩段",shape:"[T,512]",from:"fused_qkv_a_proj shard 1(576)在 norm 前切出 512 段;k_pe 64 不经本 norm"}],
  "kvb":[{kind:"upstream",label:"k̃(cache 历史)",shape:"[T,512]",from:"MLA 压缩 KV cache"}],
  "rope":[{kind:"upstream",label:"Q rope 段 + k_pe",shape:"[T,64,64]+[T,1,64]",from:"q_b_proj 与 kv_a 切分"},{kind:"external",label:"positions",shape:"[T]",from:"vLLM runner"}],
  "cache":[{kind:"upstream",label:"k̃ + k_pe",shape:"[T,512]+[T,64]",from:"kv_a_layernorm 输出"},{kind:"external",label:"slot_mapping",shape:"[T]",from:"KV cache manager"}],
  "iwqb":[{kind:"upstream",label:"q̃",shape:"[T,2048]",from:"q_a_layernorm 输出(indexer 与 q_b 共享)"}],
  "iwk":[{kind:"upstream",label:"hidden",shape:"[T,6144]",from:"input_layernorm 输出(与 q_a 投影同源)"}],
  "ikn":[{kind:"upstream",label:"k",shape:"[T,128]",from:"wk_weights_proj shard 0"}],
  "iquant":[{kind:"upstream",label:"Qidx + w",shape:"[T,32,128]+[T,32]",from:"wq_b 与 wk_weights_proj shard 1"}],
  "iscore":[{kind:"upstream",label:"FP8 q + 历史 k + w'",shape:"fp8 + [T,128] + [T,32]",from:"量化节点与 indexer key cache"}],
  "ishared":[{kind:"upstream",label:"topk_indices_buffer",shape:"[T,2048] int32",from:"最近前驱 full 层写入"}],
  "qk":[{kind:"upstream",label:"Q",shape:"[T,64,256]",from:"RoPE 输出"},{kind:"upstream",label:"候选 K(含 k_pe)",shape:"候选 2048",from:"压缩 KV cache"},{kind:"upstream",label:"候选索引",shape:"[T,2048] int32",from:"iscore / ishared"}],
  "sink":[{kind:"upstream",label:"scores",shape:"[64,≤2048]",from:"QKᵀ 输出"},{kind:"upstream",label:"sink_h",shape:"[64]",from:"learnable_sink_param"}],
  "pv":[{kind:"upstream",label:"P",shape:"[64,≤2048]",from:"softmax + sink 输出"},{kind:"upstream",label:"V",shape:"[T,64,256]",from:"kv_b_proj(cache 历史)"}],
  "gate":[{kind:"upstream",label:"heads",shape:"[T,64,256]",from:"P·V 输出"},{kind:"upstream",label:"hidden",shape:"[T,6144]",from:"子块输入(门控来源)"}],
  "oproj":[{kind:"upstream",label:"门控后 heads",shape:"[T,16384]",from:"gated MLA 输出"}],
  "gateup":[{kind:"upstream",label:"Û",shape:"[T,6144]",from:"post_attention_layernorm 输出"}],
  "act":[{kind:"upstream",label:"packed gate_up",shape:"[T,36864]",from:"gate/up 投影"}],
  "down":[{kind:"upstream",label:"激活",shape:"[T,18432]",from:"SwiGLU 输出"}],
  "router":[{kind:"upstream",label:"Û",shape:"[T,6144]",from:"post_attention_layernorm 输出"}],
  "select":[{kind:"upstream",label:"router_logits",shape:"[T,256]",from:"gate 输出"}],
  "experts":[{kind:"upstream",label:"Û + ŵ + 专家 id",shape:"[T,6144]+8",from:"post_attention_layernorm 与路由输出"}],
  "shared":[{kind:"upstream",label:"Û",shape:"[T,6144]",from:"post_attention_layernorm 输出;不经过 Top-K"}],
  "sum":[{kind:"upstream",label:"Y_routed",shape:"[T,6144]",from:"FusedMoE 输出"},{kind:"upstream",label:"Y_shared",shape:"[T,6144]",from:"shared expert 输出"}],
  "mtp":[{kind:"upstream",label:"hidden(t)",shape:"[T,6144]",from:"主干最后一层(hnorm)"},{kind:"upstream",label:"embed(t+1)",shape:"[T,6144]",from:"共享 embedding(enorm)"}],
};

export const NEXT_BY_ID: Record<string,string> = {
  "input":"hc_pre · attn",
  "hcpre-attn":"input_layernorm(归约)+ hc_post(post 门)",
  "norm1":"fused_qkv_a_proj",
  "aproj":"q_a_layernorm · kv_a_layernorm",
  "qan":"q_b_proj · indexer wq_b",
  "qb":"RoPE(q rope 段)",
  "kvn":"MLA 压缩 KV cache · RoPE(k_pe)",
  "kvb":"QKᵀ / P·V",
  "rope":"QKᵀ / MLA 压缩 KV cache(indexer 内另有独立 RoPE 实例,见 dsa-indexer 图)",
  "cache":"QKᵀ · kv_b_proj",
  "iwqb":"interleaved RoPE(q)",
  "iwk":"k_norm = LayerNorm ·(w 以文字标注给打分)",
  "ikn":"indexer key cache",
  "iquant":"加权打分 + Top-2048",
  "iscore":"QKᵀ 的候选索引",
  "ishared":"QKᵀ 的候选索引",
  "qk":"softmax + sink",
  "sink":"P·V",
  "pv":"gated MLA",
  "gate":"o_proj",
  "oproj":"hc_post · attn",
  "hcpost1":"hc_pre · mlp",
  "hcpre-mlp":"post_attention_layernorm(归约)+ hc_post(post 门)",
  "norm2":"gate/up 投影 或 FP32 router",
  "gateup":"SwiGLU(dense)",
  "act":"down 投影",
  "down":"hc_post · mlp",
  "router":"σ + Top-8 + renormalize",
  "select":"routed experts · 混合权重",
  "experts":"routed ⊕ shared",
  "shared":"routed ⊕ shared",
  "sum":"hc_post · mlp",
  "hcpost2":"下一 decoder layer / hc_head",
  "mtp":"共享 lm_head(draft logits)",
};
