import {
  AddCircle,
  Arrow,
  FORMULA_NOTE_DEFAULT,
  GraphSurface,
  InputWeightedOp,
  Op,
  RuntimeIORail,
  Tensor,
  type CodeDetail,
  type CodeSection,
  type CodeSymbol,
  type ConfigGroup,
  type EdgePort,
  type ExpandedStage,
  type FormulaTerm,
  type GraphEdge,
  type IoBinding,
  type OpKind,
  type OpNode,
  type StageOverview,
  type Weight,
} from "../atlas-shared";
import type { ModelModule, NavigatorProps, WorkbenchProps } from "./model-module";

// Hunyuan4 (hy_v4 / Hy4-preview) model module.
// Facts sourced only from:
//   * vLLM upstream commit 40824284bc (vllm/models/hy_v4/, PR #54160)
//   * official ModelScope repo Tencent-Hunyuan/Hy4-preview (config.json +
//     model.safetensors.index.json; 2006 tensors audited)
// Every shard string below is the real weight_map entry for the displayed
// layer; Hy4's sharding has no closed-form rule, so they are looked up
// per tensor instead of computed.

const VLLM_COMMIT = "40824284bc5da5b8f5db1c27f543cb46b70a6e9f";
const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/model.py`;
const ATTN_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/attention.py`;
const HC_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/hc.py`;
const MOE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/moe.py`;
const MTP_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/mtp.py`;
const FLASHMLA_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/flashmla_sparse.py`;
const INDEXER_OPS_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/sparse_attn_indexer.py`;
const MLA_ATTN_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/attention/mla_attention.py`;
const ROUTER_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`;
const MOE_UTILS_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/fused_moe/utils.py`;
const HPC_IHC_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/hpc/hpc_ihc.py`;
const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
const WEIGHTS_URL = "https://www.modelscope.cn/models/Tencent-Hunyuan/Hy4-preview";

type LayerKind = "l0" | "full" | "shared";
// Representative layer shown per kind; shard numbers audited from
// model.safetensors.index.json for exactly these layers.
const DISPLAY_LAYER: Record<LayerKind, number> = { l0: 0, full: 9, shared: 2 };

const shard = (n: number) => `model-${String(n).padStart(5, "0")}-of-00131.safetensors`;

// weight_map[file number] per displayed layer; keys are suffixes after
// `model.layers.<L>.`.
const SHARDS: Record<number, Record<string, number>> = {
  0: {
    "hc_attn_layer.hc_pre.hc_fn": 93, "hc_attn_layer.hc_pre.hc_base": 94, "hc_attn_layer.hc_pre.hc_scale": 89,
    "hc_mlp_layer.hc_pre.hc_fn": 93, "hc_mlp_layer.hc_pre.hc_base": 94, "hc_mlp_layer.hc_pre.hc_scale": 84,
    "input_layernorm.weight": 55, "post_attention_layernorm.weight": 55,
    "mlp.gate_proj.weight": 125, "mlp.up_proj.weight": 125, "mlp.down_proj.weight": 130,
    "self_attn.indexer.wq_b.weight": 119, "self_attn.indexer.wk.weight": 78, "self_attn.indexer.weights_proj.weight": 125,
    "self_attn.indexer.k_norm.weight": 89, "self_attn.indexer.k_norm.bias": 89,
    "self_attn.q_a_proj.weight": 70, "self_attn.kv_a_proj_with_mqa.weight": 119,
    "self_attn.q_a_layernorm.weight": 130, "self_attn.q_b_proj.weight": 58,
    "self_attn.kv_a_layernorm.weight": 104, "self_attn.kv_b_proj.weight": 73,
    "self_attn.o_proj.weight": 130, "self_attn.linear_gate.weight": 130,
    "self_attn.learnable_sink_param": 89,
  },
  2: {
    "hc_attn_layer.hc_pre.hc_fn": 99, "hc_attn_layer.hc_pre.hc_base": 123, "hc_attn_layer.hc_pre.hc_scale": 123,
    "hc_mlp_layer.hc_pre.hc_fn": 99, "hc_mlp_layer.hc_pre.hc_base": 99, "hc_mlp_layer.hc_pre.hc_scale": 123,
    "input_layernorm.weight": 123, "post_attention_layernorm.weight": 123,
    "mlp.gate.weight": 113, "mlp.gate.e_score_correction_bias": 99,
    "mlp.experts.gate_up_proj": 4, "mlp.experts.down_proj": 4,
    "mlp.shared_experts.gate_proj.weight": 88, "mlp.shared_experts.up_proj.weight": 88, "mlp.shared_experts.down_proj.weight": 88,
    "self_attn.q_a_proj.weight": 88, "self_attn.kv_a_proj_with_mqa.weight": 113,
    "self_attn.q_a_layernorm.weight": 113, "self_attn.q_b_proj.weight": 83,
    "self_attn.kv_a_layernorm.weight": 99, "self_attn.kv_b_proj.weight": 123,
    "self_attn.o_proj.weight": 69, "self_attn.linear_gate.weight": 69,
    "self_attn.learnable_sink_param": 113,
  },
  9: {
    "hc_attn_layer.hc_pre.hc_fn": 125, "hc_attn_layer.hc_pre.hc_base": 89, "hc_attn_layer.hc_pre.hc_scale": 84,
    "hc_mlp_layer.hc_pre.hc_fn": 125, "hc_mlp_layer.hc_pre.hc_base": 89, "hc_mlp_layer.hc_pre.hc_scale": 84,
    "input_layernorm.weight": 130, "post_attention_layernorm.weight": 130,
    "mlp.gate.weight": 78, "mlp.gate.e_score_correction_bias": 109,
    "mlp.experts.gate_up_proj": 25, "mlp.experts.down_proj": 25,
    "mlp.shared_experts.gate_proj.weight": 119, "mlp.shared_experts.up_proj.weight": 119, "mlp.shared_experts.down_proj.weight": 119,
    "self_attn.indexer.wq_b.weight": 119, "self_attn.indexer.wk.weight": 93, "self_attn.indexer.weights_proj.weight": 55,
    "self_attn.indexer.k_norm.weight": 94, "self_attn.indexer.k_norm.bias": 94,
    "self_attn.q_a_proj.weight": 119, "self_attn.kv_a_proj_with_mqa.weight": 78,
    "self_attn.q_a_layernorm.weight": 104, "self_attn.q_b_proj.weight": 73,
    "self_attn.kv_a_layernorm.weight": 89, "self_attn.kv_b_proj.weight": 70,
    "self_attn.o_proj.weight": 58, "self_attn.linear_gate.weight": 59,
    "self_attn.learnable_sink_param": 94,
  },
};

function wt(layer: number, suffix: string, shape: string, dtype: "BF16" | "F32", params: string, runtime?: string, note?: string): Weight {
  return { key: `model.layers.${layer}.${suffix}`, shape, dtype, shard: shard(SHARDS[layer][suffix]), params, runtime, note };
}

const hcWeights = (layer: number, block: "hc_attn_layer" | "hc_mlp_layer"): Weight[] => [
  wt(layer, `${block}.hc_pre.hc_fn`, "[8,24576]", "F32", "196.61K", "hc_pre.hc_fn · ReplicatedLinear", "FP32 计算；TP 下全卡复制"),
  wt(layer, `${block}.hc_pre.hc_base`, "[8]", "F32", "8", "hc_pre.hc_base", "init：前 4 位 −log3、后 4 位 0"),
  wt(layer, `${block}.hc_pre.hc_scale`, "[2]", "F32", "2", "hc_pre.hc_scale", "init 0.01"),
];

const LATEX_BY_ID: Record<string, string> = {
  "hc-attnpre": String.raw`\begin{aligned}r&=\Big(\tfrac{1}{C\cdot H}\sum_{j}x_j^2+\varepsilon_{rms}\Big)^{-\frac12}\\m&=W_{hc}\,\mathrm{flat}(x)\cdot r\\H_{pre,i}&=\sigma(m_i\,s_{0}+b_i)+\varepsilon_{hc},\quad i<C\\H_{post,i}&=M\,\sigma(m_{i+C}\,s_{1}+b_{i+C})+\varepsilon_{hc}\\y&=\sum_{i=1}^{C}H_{pre,i}\,x_i\end{aligned}`,
  "hc-mlppre": String.raw`\begin{aligned}r&=\Big(\tfrac{1}{C\cdot H}\sum_{j}x_j^2+\varepsilon_{rms}\Big)^{-\frac12}\\m&=W_{hc}\,\mathrm{flat}(x)\cdot r\\H_{pre,i}&=\sigma(m_i\,s_{0}+b_i)+\varepsilon_{hc},\quad i<C\\H_{post,i}&=M\,\sigma(m_{i+C}\,s_{1}+b_{i+C})+\varepsilon_{hc}\\y&=\sum_{i=1}^{C}H_{pre,i}\,x_i\end{aligned}`,
  "hc-attnpost": String.raw`y_{n,i,:}=H_{post,n,i}\,f(x)_{n,:}+x_{n,i,:}`,
  "hc-mlppost": String.raw`y_{n,i,:}=H_{post,n,i}\,f(x)_{n,:}+x_{n,i,:}`,
  "norm-in": String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\tfrac{1}{H}\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}\,\gamma_i\end{aligned}`,
  "norm-post": String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\tfrac{1}{H}\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}\,\gamma_i\end{aligned}`,
  "a-fused": String.raw`z=\hat x\,\big[W_{q_a}^{\top}\mid W_{kv_a}^{\top}\big]`,
  "a-split": String.raw`(q_c,kv_l)=\operatorname{Split}\!\big(z;\,R_q,\,L_k+D_r\big)`,
  "a-qnorm": String.raw`q_c^{n}=\operatorname{RMSNorm}(q_c)`,
  "a-qup": String.raw`q=q_c^{n}\,W_{q_b}^{\top}\in\mathbb{R}^{T\times N_h\cdot D_{qk}}`,
  "a-kvsplit": String.raw`(kv_c,k_{pe})=\operatorname{Split}\!\big(kv_l;\,L_k,\,D_r\big)`,
  "a-kvnorm": String.raw`c_n=\operatorname{RMSNorm}(kv_c)`,
  "a-kvb": String.raw`\begin{aligned}\text{prefill: }&\big[K^{nope}_h\mid V_h\big]=c_n\,W_{kv_b,h}^{\top}\\\text{decode: }&q^{lat}_h=q^{nope}_h\,W_{UK,h}^{\top},\quad o_h=q^{lat}_h\,W_{UV,h}\end{aligned}`,
  "a-rope": String.raw`\begin{aligned}\theta_{p,j}&=p\,\theta_{base}^{-2j/D_r}\\\binom{x_{2j}}{x_{2j+1}}&=\begin{bmatrix}\cos\theta_{p,j}&-\sin\theta_{p,j}\\ \sin\theta_{p,j}&\cos\theta_{p,j}\end{bmatrix}\binom{x_{2j}}{x_{2j+1}}\end{aligned}`,
  "a-cache": String.raw`\mathcal C\big[\mathrm{slot}(p)\big]\leftarrow\big[c_n\mid k_{pe}^{r}\big]\in\mathbb{R}^{L_k+D_r}`,
  "a-core": String.raw`\begin{aligned}a_{t,h,s}&=\frac{q_{t,h}^{\top}k_s}{\sqrt{D_{qk}}}\\o_{t,h}&=\frac{\sum_{s\in\mathcal I_t}e^{a_{t,h,s}}\,v_{s}}{\sum_{s\in\mathcal I_t}e^{a_{t,h,s}}+e^{\kappa_h}}\end{aligned}`,
  "a-gate": String.raw`o'=o\odot\sigma\big(\hat x\,W_g^{\top}\big)`,
  "a-oproj": String.raw`Y_{attn}=o'\,W_{o}^{\top}\in\mathbb{R}^{T\times H}`,
  "i-qup": String.raw`q^{idx}=\mathrm{flat}\big(W_{wq_b}\,q_c^{n}\big)\in\mathbb{R}^{T\times N_{idx}\times D_{idx}}`,
  "i-kw": String.raw`\big[k\mid w\big]=\hat x\,\big[W_{wk}^{\top}\mid W_{\pi}^{\top}\big]`,
  "i-knorm": String.raw`\tilde k=\gamma_{kn}\odot\frac{k-\mu(k)}{\sqrt{\sigma^2(k)+\varepsilon_{kn}}}+\beta_{kn}`,
  "i-rope": String.raw`\big(q_{pe}^{r},k_{pe}^{r}\big)=\operatorname{RoPE}\big(q_{pe},k_{pe};\mathbf p\big)`,
  "i-quant": String.raw`\begin{aligned}\big(q^{fp8},s_q\big)&=\operatorname{Quant}^{ue8m0}_{D_{idx}}(q)\\w_h&=\pi_h\cdot s_q\cdot D_{idx}^{-\frac12}\cdot N_{idx}^{-\frac12}\end{aligned}`,
  "i-logits": String.raw`S_{t,s}=\sum_{h=1}^{N_{idx}}w_h(t)\,\big\langle q^{fp8}_{h}(t),\,k^{fp8}(s)\big\rangle`,
  "i-topk": String.raw`\mathcal I_t=\operatorname{TopK}_{K_{idx}}\big(S_{t,:}\big)\ \longrightarrow\ \text{topk\_indices\_buffer}`,
  "i-buffer": String.raw`\mathcal I_t \leftarrow \text{topk\_indices\_buffer}[t]`,
  "m-router": String.raw`\begin{aligned}s&=\sigma\big(u\,W_{route}^{\top}\big)\\\mathcal E&=\operatorname{TopK}_K(s+b)\\\hat w_e&=\frac{s_e}{\sum_{j\in\mathcal E}s_j}\end{aligned}`,
  "m-experts": String.raw`\begin{aligned}\bar g_e&=\min(g_e,c)&\bar u_e&=\operatorname{clip}(u_e,-c,c)\\E_e(u)&=W_{2,e}\big[\operatorname{silu}(\bar g_e)\odot\bar u_e\big]\end{aligned}`,
  "m-shared": String.raw`E_{sh}(u)=W_{2}^{sh}\big[\operatorname{silu}(W_{1}^{sh}u)\odot W_{3}^{sh}u\big]`,
  "m-sum": String.raw`Y_{moe}=s_{route}\sum_{e\in\mathcal E}\hat w_e\,E_e(u)+E_{sh}(u)`,
  "d-gateup": String.raw`\begin{aligned}G&=\hat u\,W_{gate}^{\top}\\U&=\hat u\,W_{up}^{\top}\end{aligned}`,
  "d-split": String.raw`(G,U)=\operatorname{Split}(x_{gate\_up};\,H_{dense},\,H_{dense})`,
  "d-act": String.raw`Z=\operatorname{silu}(G)\odot U`,
  "d-down": String.raw`Y_{ffn}=Z\,W_{down}^{\top}\in\mathbb{R}^{T\times H}`,
  position: String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\end{aligned}`,
  attnmeta: String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  slots: String.raw`\begin{aligned}\ell&=\left\lfloor p/B_{block}\right\rfloor,\quad o=p\bmod B_{block}\\\mathrm{slot}(r,p)&=\mathrm{block\_table}[r,\ell]\cdot B_{block}+o\end{aligned}`,
};

const FORMULA_TERMS_BY_KIND: Record<OpKind, readonly FormulaTerm[]> = {
  io: [["x", "输入"], ["y", "输出"]],
  norm: [["x", "输入向量"], ["y", "归一化输出"], ["γ", "可训练缩放权重"], ["ε", "数值稳定项"]],
  linear: [["x", "输入张量"], ["W", "投影权重"], ["y", "线性投影输出"]],
  split: [["x", "待切分张量"], ["a,b,…", "沿最后一维得到的输出"]],
  rope: [["x", "Q 或 K 的 rope 切片"], ["p", "token position"], ["θ", "旋转角度"]],
  matmul: [["a", "左输入张量"], ["b", "右输入张量"], ["y", "矩阵乘输出"]],
  scale: [["x", "未缩放输入"], ["y", "缩放后输出"]],
  mask: [["x", "原始 score"], ["M", "mask"], ["y", "mask 后 score"]],
  softmax: [["x", "输入 score"], ["p", "归一化概率"]],
  activation: [["g", "gate 分支"], ["u", "up 分支"], ["y", "激活输出"]],
  route: [["s", "路由分数"], ["K", "选择数量"], ["𝓔 / ℐ", "选中的 expert 或 token id"]],
  cache: [["K / V", "写入 cache 的张量"], ["slot", "物理 cache 位置"], ["block_table", "逻辑块到物理页映射"]],
  add: [["x", "residual 分支"], ["f(x)", "当前计算分支"], ["y", "相加结果"]],
};

const T = (symbol: string, meaning: string): FormulaTerm => [symbol, meaning];
const COMMON_TERMS: readonly FormulaTerm[] = [
  T("H", "hidden_size = 6144"), T("T", "本轮 token 数"), T("C", "hc_mult = 4 残差通道"),
  T("θ_base", "rope_theta = 10⁷"), T("V", "vocab_size = 120832"),
];

const FORMULA_TERMS_BY_ID: Record<string, readonly FormulaTerm[]> = {
  "hc-attnpre": [...COMMON_TERMS, T("x", "iHC 通道状态 · [T,C,H]"), T("W_hc", "hc_fn.weight · [2C,C·H] FP32"), T("b", "hc_base · init 前半 −log3"), T("s₀/s₁", "hc_scale · init 0.01"), T("M", "hc_magnitude = 2.0（仅 post 半段）"), T("ε_hc", "hc_eps = 10⁻⁶"), T("H_pre", "各通道进入子块的比例"), T("y", "压缩后的子块输入 · [T,H]")],
  "hc-mlppre": [...COMMON_TERMS, T("x", "iHC 通道状态 · [T,C,H]"), T("W_hc", "hc_fn.weight · [2C,C·H] FP32"), T("b", "hc_base · init 前半 −log3"), T("s₀/s₁", "hc_scale · init 0.01"), T("M", "hc_magnitude = 2.0（仅 post 半段）"), T("ε_hc", "hc_eps = 10⁻⁶"), T("H_pre", "各通道进入子块的比例"), T("y", "压缩后的子块输入 · [T,H]")],
  "hc-attnpost": [T("f(x)", "attention 子块输出 · [T,H]"), T("H_post", "pre 层同时算出的 post gate（×2.0 幅度）"), T("x", "压缩前的 4 通道残差 · [T,C,H]"), T("y", "散射回的 4 通道 · [T,C,H]"), T("hc_post", "无参数：纯逐元素运算")],
  "hc-mlppost": [T("f(x)", "FFN / MoE 子块输出 · [T,H]"), T("H_post", "pre 层同时算出的 post gate（×2.0 幅度）"), T("x", "压缩前的 4 通道残差 · [T,C,H]"), T("y", "散射回的 4 通道 · [T,C,H]"), T("hc_post", "无参数：纯逐元素运算")],
  "norm-in": [T("x", "hc_attn_layer.pre 输出 y"), T("γ", "input_layernorm.weight"), T("ε", "rms_norm_eps = 10⁻⁵"), T("y", "注意力的输入 û")],
  "norm-post": [T("x", "hc_mlp_layer.pre 输出 y′"), T("γ", "post_attention_layernorm.weight"), T("ε", "rms_norm_eps = 10⁻⁵"), T("y", "FFN / MoE 的输入")],
  "a-fused": [T("û", "input_layernorm 后的 hidden"), T("W_qa", "q_a_proj · [R_q,H] replicated"), T("W_kva", "kv_a_proj_with_mqa · [L_k+D_r,H] replicated"), T("z", "qkv_lora · [T,R_q+L_k+D_r]"), T("R_q", "q_lora_rank = 2048"), T("L_k", "kv_lora_rank = 512"), T("D_r", "qk_rope_head_dim = 64")],
  "a-split": [T("q_c", "Q 低秩激活 · [T,R_q]"), T("kv_l", "KV latent + 解耦 k_pe · [T,L_k+D_r]")],
  "a-qnorm": [T("q_c", "Q 低秩激活"), T("γ", "q_a_layernorm.weight · [R_q]"), T("ε", "rms_norm_eps = 10⁻⁵"), T("q_cⁿ", "norm 后的 Q 低秩（indexer 也复用它）")],
  "a-qup": [T("W_qb", "q_b_proj · [N_h·D_qk,R_q]（TP 列切）"), T("N_h", "num_attention_heads = 64"), T("D_qk", "qk_head_dim = 192+64 = 256"), T("q", "每 head nope 192 在前 + rope 64 在后")],
  "a-kvsplit": [T("kv_c", "共享 KV 压缩向量 · [T,L_k]"), T("k_pe", "解耦的 per-token rope key · [T,D_r]")],
  "a-kvnorm": [T("kv_c", "KV latent"), T("γ", "kv_a_layernorm.weight · [L_k]"), T("c_n", "norm 后 latent（写 cache / 上投影都用它）")],
  "a-kvb": [T("W_kvb", "kv_b_proj · [N_h·(D_n+D_v),L_k]（TP 列切）"), T("W_UK/W_UV", "加载后按 head 拆分预转（decode 吸收）"), T("D_n", "qk_nope_head_dim = 192"), T("D_v", "v_head_dim = 256"), T("q^lat", "latent query · [B,N_h,L_k]")],
  "a-rope": [T("D_r", "只旋转每 head 的后 64 维"), T("θ_base", "rope_theta = 10⁷"), T("p", "token position"), T("interleaved", "is_neox_style=False（PTM 布局，NeoX 会破坏 DSA top-k）")],
  "a-cache": [T("C", "latent KV cache · 每 token L_k+D_r = 576 元素"), T("slot", "物理 cache 槽位"), T("fp8_ds_mla", "FP8 打包后 656 B/token"), T("k_pe^r", "RoPE 后的解耦 key")],
  "a-core": [T("ℐ_t", "indexer 选出的 Top-K token 集合"), T("K_idx", "index_topk = 2048"), T("κ_h", "learnable_sink_param · [N_h] FP32"), T("e^κ", "sink 折进 softmax 分母的虚拟项"), T("D_qk", "scale = 256^-0.5"), T("prefill", "flash_mla_sparse_fwd 按 indices 稀疏 MHA"), T("decode", "W_UK 吸收后对 latent cache 做 MQA")],
  "a-gate": [T("W_g", "linear_gate · [N_h·D_v,H]（TP 按头列切）"), T("û", "input_layernorm 后的 hidden（gate 输入）"), T("elementwise", "每 head 每维一个门（gating_type）"), T("o", "attention 输出 · [T,N_h·D_v]")],
  "a-oproj": [T("W_o", "o_proj · [H,N_h·D_v]（TP 行切）"), T("Y_attn", "hc_attn_layer.post 的输入")],
  "i-qup": [T("W_wqb", "indexer.wq_b · [N_idx·D_idx,R_q] replicated"), T("q_cⁿ", "复用 MLA 的 q_a_layernorm 输出"), T("N_idx", "index_n_heads = 32"), T("D_idx", "index_head_dim = 128（nope 64 + rope 64）")],
  "i-kw": [T("W_wk", "indexer.wk · [D_idx,H]"), T("W_π", "indexer.weights_proj · [N_idx,H]"), T("û", "input_layernorm 后的 hidden"), T("融合", "wk 与 weights_proj 合成单 GEMM wk_weights_proj")],
  "i-knorm": [T("γ_kn/β_kn", "k_norm.weight/bias · [D_idx]"), T("ε_kn", "硬编码 1e-6"), T("LayerNorm", "全模型唯一带 bias 的 norm")],
  "i-rope": [T("D_r", "indexer 内部 rope 维 = 64"), T("布局", "nope 64 在前 / pe 64 在后，转完拼回")],
  "i-quant": [T("s_q", "ue8m0 scale（分组恰为一个 head）"), T("π_h", "weights_proj 输出的 per-head 权重"), T("k 侧", "k 量化与 cache 写入融合（fp8 + fp32 scale = 132 B/token）")],
  "i-logits": [T("w_h", "折叠后的 per-head 权重"), T("N_idx", "32 头加权和，无激活函数"), T("kernel", "prefill: fp8_fp4_mqa_logits · decode: paged 版")],
  "i-topk": [T("K_idx", "index_topk = 2048"), T("buffer", "全模型唯一 [max_num_batched_tokens,2048] int32"), T("哨兵", "预填 −1 = 无 token")],
  "i-buffer": [T("buffer", "topk_indices_buffer · 上一个 full 层写入"), T("skip_topk", "shared 层不构建 indexer，只按行读"), T("MTP", "draft step 0 自算、step 1+ 复用同一 buffer")],
  "m-router": [T("W_route", "mlp.gate.weight · [E,H]（checkpoint BF16，运行时 FP32）"), T("b", "e_score_correction_bias · [E] → expert_bias"), T("E", "n_routed_experts = 256"), T("K", "num_experts_per_tok = 8"), T("选与算分离", "topk 用 σ+b，权重用原始 σ 再 renorm")],
  "m-experts": [T("c", "swiglu_limit = 10.0（仅 routed）"), T("g_e/u_e", "expert gate / up 分支"), T("E", "256 个专家打包为 gate_up_proj / down_proj"), T("H_e", "moe_intermediate_size = 2048")],
  "m-shared": [T("W₁/W₃/W₂", "shared_experts.{gate,up,down}_proj"), T("无限幅", "shared expert 不做 clamp"), T("H_e", "intermediate = 2048")],
  "m-sum": [T("s_route", "routed_scaling_factor = 2.827"), T("ŵ_e", "renorm 后的 top-8 权重"), T("E_sh", "shared expert 输出")],
  "d-gateup": [T("W_gate/W_up", "L0 mlp.{gate,up}_proj · [H_dense,H]"), T("H_dense", "intermediate_size = 18432（仅 L0）")],
  "d-split": [T("G/U", "gate / up 两半"), T("H_dense", "18432")],
  "d-act": [T("silu", "SiluAndMul · 无 clamp（与 routed 专家的差异）")],
  "d-down": [T("W_down", "L0 mlp.down_proj · [H,H_dense]")],
  position: [T("q_b", "请求 b 本轮调度的 query 数"), T("p_b,i", "请求 b 第 i 个 token position")],
  attnmeta: [T("q_b", "请求 b 的 query 数"), T("c_b", "请求 b 已有 context 长度"), T("M", "causal / padding mask")],
  slots: [T("p", "token position"), T("B_block", "KV cache block size · runtime"), T("block_table", "逻辑块到物理页映射")],
};

function formulaTerms(node: OpNode) {
  return FORMULA_TERMS_BY_ID[node.id] ?? (node.latex ? [] : FORMULA_TERMS_BY_KIND[node.kind]);
}

const code = (body: string) => body.replace(/^/gm, "").trimStart();

const HC_PRE_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "HYV4HCPreLayer.forward：RMS 统计 + hc_fn 投影 + 双 sigmoid gate", location: "hy_v4/nvidia/hc.py · L97-L139", url: `${HC_URL}#L97-L139`, code: code(`
x_flat = x.flatten(1).float()  # [num_tokens, hc*d]
rsqrt = torch.rsqrt(
    x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
)
mixes = self.hc_fn(x_flat)[0] * rsqrt  # [num_tokens, 2*hc]

pre_raw = mixes[..., :hc]
post_raw = mixes[..., hc : 2 * hc]

pre = (
    torch.sigmoid(
        pre_raw * self.hc_scale[0].float() + self.hc_base[:hc].float()
    )
    + hc_eps
)
post = (
    self.magnitude
    * torch.sigmoid(
        post_raw * self.hc_scale[1].float() + self.hc_base[hc : 2 * hc].float()
    )
    + hc_eps
)

y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
return y.to(x.dtype), post`) },
  { stage: "2 · CALL", title: "HYV4DecoderLayer._forward_ihc：每个子块 pre → norm → sub-block → post", location: "hy_v4/nvidia/model.py · L187-L209", url: `${CODE_URL}#L187-L209`, code: code(`
hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(
    positions=positions,
    hidden_states=hidden_states,
)
hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)`) },
  { stage: "3 · ENTER", title: "HPC 融合版：单 kernel 替代 eager 的 20 个 launch", location: "layers/hpc/hpc_ihc.py · L103-L126", url: `${HPC_IHC_URL}#L103-L126`, code: code(`
"""Computes, in one kernel:
    x_flat = x.flatten(1)
    r      = rsqrt(x_flat.square().mean(-1) + rms_norm_eps)
    mixes  = (x_flat @ w.T) * r
    H_pre  = sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
    H_post = magnitude * sigmoid(
                 mixes[:, hc:] * hc_scale[1] + hc_base[hc:]) + hc_eps
    y      = sum_i H_pre[:, i] * x[:, i, :]
"""`) },
];
const HC_PRE_SYMBOLS: CodeSymbol[] = [
  { symbol: "self.hc_fn", resolvesTo: "ReplicatedLinear(24576→8, fp32)", meaning: "输入是展平的全宽通道（6144 不做 TP 切分），输出 2×hc 个 gate logit。" },
  { symbol: "self.magnitude", resolvesTo: "hc_magnitude = 2.0", meaning: "只乘在 post 半段；pre 半段幅度为 1。" },
  { symbol: "post_gates", resolvesTo: "H_post", meaning: "pre 与 post 一并算出，post 存起来给同子块的 HYV4HCPostLayer 用。" },
  { symbol: "prepare_input", resolvesTo: "[T,6144] → repeat(1,4,1)", meaning: "首层把 embedding 广播成 4 通道；层间直接传 [T,4,6144]。" },
];

const HC_POST_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "HYV4HCPostLayer.forward：post-gate × 子块输出 + 多通道残差", location: "hy_v4/nvidia/hc.py · L163-L186", url: `${HC_URL}#L163-L186`, code: code(`
post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)  # [num_tokens, hc, d]
y = post_gated + residual
return y.to(dtype)`) },
];
const HC_POST_SYMBOLS: CodeSymbol[] = [
  { symbol: "residual", resolvesTo: "pre 层保存的 4 通道输入", meaning: "多通道残差；iHC 下 DecoderLayer 对外返回的 residual 是 None。" },
  { symbol: "hc_post 权重", resolvesTo: "无", meaning: "post 层没有任何参数，checkpoint 实测也只有 hc_pre.*。" },
];

const NORM_SECTIONS: CodeSection[] = [
  { stage: "1 · CALL", title: "HYV4DecoderLayer._forward_ihc：两个标准 RMSNorm 的调用位置", location: "hy_v4/nvidia/model.py · L191,L198", url: `${CODE_URL}#L187-L209`, code: code(`
hidden_states = self.input_layernorm(hidden_states)          # attn 前
...
hidden_states = self.post_attention_layernorm(hidden_states) # mlp 前`) },
  { stage: "2 · INIT", title: "HYV4DecoderLayer.__init__：RMSNorm(6144, eps=1e-5)", location: "hy_v4/nvidia/model.py · L87-L150", url: `${CODE_URL}#L87-L150`, code: code(`
self.input_layernorm = RMSNorm(
    config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(
    config.hidden_size, eps=config.rms_norm_eps)`) },
];

const MLA_INIT_SECTIONS: CodeSection[] = [
  { stage: "1 · INIT", title: "HYV4MLAAttention.__init__：q_a + kv_a 融合成一个 replicated GEMM", location: "hy_v4/nvidia/attention.py · L361-L372", url: `${ATTN_URL}#L361-L372`, code: code(`
# \`\`q_a_proj\`\` and \`\`kv_a_proj_with_mqa\`\` read the same
# \`\`hidden_states\`\` and are both TP-replicated, so they run as one
# GEMM. The checkpoint keeps them separate; \`load_weights\` merges
# them through \`\`stacked_params_mapping\`\`.
self.fused_qkv_a_proj = MergedColumnParallelLinear(
    self.hidden_size,
    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.fused_qkv_a_proj",
    disable_tp=True,
)`) },
];
const MLA_INIT_SYMBOLS: CodeSymbol[] = [
  { symbol: "disable_tp=True", resolvesTo: "TP 全复制", meaning: "低秩下投影不切分：q_a_proj 与 kv_a_proj_with_mqa 合并后每卡全量。" },
  { symbol: "stacked_params_mapping", resolvesTo: "model.py · L621-L624", meaning: "加载时把 checkpoint 的两块矩阵拼进 fused_qkv_a_proj。" },
];

const MLA_FWD_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "HYV4MLAAttention.forward：低秩下投影 → 双 RMSNorm → 上投影 → RoPE 只转 rope 切片", location: "hy_v4/nvidia/attention.py · L664-L691", url: `${ATTN_URL}#L664-L691`, code: code(`
qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
q_c, kv_lora = qkv_lora.split(
    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
    dim=-1,
)
q_c = self.q_a_layernorm(q_c)
q = self.q_b_proj(q_c)[0]

kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_c_normed = self.kv_a_layernorm(kv_c)

q = q.view(-1, self.num_local_heads, self.qk_head_dim)
k_pe = k_pe.unsqueeze(1)
q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
    positions, q[..., self.qk_nope_head_dim :], k_pe
)`) },
];
const MLA_FWD_SYMBOLS: CodeSymbol[] = [
  { symbol: "q[..., 192:]", resolvesTo: "每 head 后 64 维", meaning: "RoPE 只作用于 rope 切片；nope 192 维原样通过。" },
  { symbol: "is_neox_style=False", resolvesTo: "interleaved RoPE", meaning: "PTM/Megatron 布局；indexer 注释明确用 NeoX 会破坏 DSA top-k。" },
  { symbol: "q_c（norm 后）", resolvesTo: "indexer 的输入", meaning: "indexer.wq_b 复用 MLA 的 Q 低秩压缩，不再单独下投影。" },
];

const GATED_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "gated MLA：σ(W_gate·û) 逐元素乘（o_proj 之前）", location: "hy_v4/nvidia/attention.py · L710-L733", url: `${ATTN_URL}#L710-L733`, code: code(`
if self.gated_mla and self.linear_gate is not None:
    ...
    gate_score = self.linear_gate(hidden_states)[0]
    if self.config.gating_type == "headwise":
        gate_score = gate_score.unsqueeze(-1)
        attn_out = attn_out.reshape(
            *attn_out.shape[:-1], -1, self.v_head_dim
        )
        attn_out = attn_out * torch.sigmoid(gate_score)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
    else:
        attn_out = attn_out * torch.sigmoid(gate_score)

out, _ = self.o_proj(attn_out)`) },
  { stage: "2 · ENTER", title: "HPC 融合版约束（投影+sigmoid+乘法单 launch）", location: "layers/hpc/gated_mla.py · L1-L16", url: `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/hpc/gated_mla.py#L1-L16`, code: code(`
Kernel constraints:
  - Requires VLLM_ENABLE_HPC_OPS=1
  - Only sm100 / sm103 (compute capability 100, 103)
  - All three operands must be bfloat16 and contiguous
  - The gate width must be a multiple of 256
  - Only elementwise gating (headwise has no fused form)`) },
];
const GATED_SYMBOLS: CodeSymbol[] = [
  { symbol: "hidden_states", resolvesTo: "input_layernorm 后的 û", meaning: "门控输入是 self_attn 的入口 hidden（norm 后），不是 attention 内部的 q/kv。" },
  { symbol: "gating_type", resolvesTo: "elementwise", meaning: "每 head 每 v 维一个门（64×256=16384）；headwise 是每 head 1 个标量。" },
  { symbol: "16384 = 64×256", resolvesTo: "256 对齐", meaning: "恰好满足融合核的 gate width 256 倍数约束。" },
];

const ABSORB_SECTIONS: CodeSection[] = [
  { stage: "1 · INIT", title: "kv_b 拆 W_UK / W_UV 并预转（process_weights_after_loading）", location: "layers/attention/mla_attention.py · L1173-L1181,L1238-L1246", url: `${MLA_ATTN_URL}#L1238-L1246`, code: code(`
kv_b_proj_weight = kv_b_proj_weight.view(
    self.kv_lora_rank,
    self.num_heads,
    self.qk_nope_head_dim + self.v_head_dim,
)

W_UK, W_UV = kv_b_proj_weight.split(
    [self.qk_nope_head_dim, self.v_head_dim], dim=-1
)
# Convert from (L, N, V) to (N, L, V)
replace_parameter(self, "W_UV", W_UV.transpose(0, 1), prefer_copy=True)
# Convert from (L, N, P) to (N, P, L)
replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0), prefer_copy=True)`) },
  { stage: "2 · FORWARD", title: "decode 吸收：q_nope × W_UK^T → latent query，对 576 维 latent cache 做 MQA", location: "layers/attention/mla_attention.py · L933-L938,L981-L1007", url: `${MLA_ATTN_URL}#L981-L1007`, code: code(`
mqa_q_nope, mqa_q_pe = mqa_q.split(
    [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
)
# Convert from (B, N, P) to (N, B, P)
mqa_q_nope = mqa_q_nope.transpose(0, 1)
...
# Multiply (N, B, P) x (N, P, L) -> (N, B, L)
torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)
mqa_q = (mqa_ql_nope.transpose(0, 1), mqa_q_pe)`) },
  { stage: "3 · CALL", title: "_v_up_proj：latent 输出经 W_UV 上投影回 v 空间", location: "layers/attention/mla_attention.py · L1290-L1322", url: `${MLA_ATTN_URL}#L1290-L1322`, code: code(`
x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
out = out.view(-1, self.num_heads, self.v_head_dim)
# Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
torch.bmm(x, self.W_UV, out=out.transpose(0, 1))`) },
];
const ABSORB_SYMBOLS: CodeSymbol[] = [
  { symbol: "W_UK_T", resolvesTo: "(64,192,512)", meaning: "吸收使 decode 的 Q×K 变成一次 batched GEMM，K 不再逐 head 物化。" },
  { symbol: "W_UV", resolvesTo: "(64,512,256)", meaning: "decode 输出 [B,N,512]（latent）→ [B,N,256]（v 空间）。" },
  { symbol: "prefill", resolvesTo: "flash_mla 稀疏 MHA", meaning: "prefill 由 FlashMLA kernel 在内部完成同样的吸收。" },
];

const SINK_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "sink 折进 softmax 分母（FlashMLA kernel 的 attn_sink 参数）", location: "hy_v4/nvidia/flashmla_sparse.py · L44-L51,L194-L205", url: `${FLASHMLA_URL}#L44-L51`, code: code(`
"""FlashMLA sparse impl that applies HY V4's per-head learnable sink.

The sink enters as the \`\`sinks\`\` impl kwarg of
\`vllm.model_executor.layers.attention.MLAAttention\` and is consumed by
the FlashMLA kernels, which fold it into the softmax denominator:
\`out *= exp(lse) / (exp(lse) + exp(sink))\`.
"""

out, lse = flash_mla_with_kvcache(
    q=q,
    k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
    block_table=kernel_metadata.dummy_block_table,
    head_dim_v=512,
    cache_seqlens=kernel_metadata.cache_lens,
    is_fp8_kvcache=True,
    indices=topk_indices,
    softmax_scale=self.softmax_scale,
    attn_sink=attn_sink,
)`) },
  { stage: "2 · INIT", title: "sink 参数构建：每 head 一个 FP32 标量", location: "hy_v4/nvidia/attention.py · L482-L497", url: `${ATTN_URL}#L482-L497`, code: code(`
# kernels require fp32 sinks; per-head learnable attention sink logit
self.learnable_sink_param = nn.Parameter(
    torch.empty(num_local_heads, dtype=torch.float32), ...)`) },
];
const SINK_SYMBOLS: CodeSymbol[] = [
  { symbol: "sink", resolvesTo: "加性 logit", meaning: "等价于 softmax 分母多加一项 exp(sink_h)：一个不依赖任何 key 的虚拟 token。" },
  { symbol: "attn_sink", resolvesTo: "kernel 参数", meaning: "pad 的 head 用 −inf 填充（无效 sink）。" },
  { symbol: "force sparse MQA", resolvesTo: "attention.py · L635-L656", meaning: "dense MLA prefill 后端不支持 sink，所以 Hy4 的 prefill 也全走稀疏 MQA 路径。" },
];

const IDX_INIT_SECTIONS: CodeSection[] = [
  { stage: "1 · INIT", title: "Indexer.__init__：wq_b + wk/weights_proj 融合 + k_norm(LayerNorm 1e-6)", location: "hy_v4/nvidia/attention.py · L143-L171", url: `${ATTN_URL}#L143-L171`, code: code(`
# No tensor parallelism, just replicated.
self.wq_b = ReplicatedLinear(
    self.q_lora_rank,
    self.head_dim * self.n_head,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.wq_b",
)
# Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
# FP8 wk weights are upcast to BF16 while loading to keep the fusion.
self.wk_weights_proj = MergedColumnParallelLinear(
    hidden_size,
    [self.head_dim, self.n_head],
    bias=False,
    quant_config=None,
    disable_tp=True,
    prefix=f"{prefix}.wk_weights_proj",
)
self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
self.softmax_scale = self.head_dim**-0.5

self.scale_fmt = "ue8m0"
self.quant_block_size = 128`) },
];
const IDX_INIT_SYMBOLS: CodeSymbol[] = [
  { symbol: "wq_b 输入", resolvesTo: "q_a_layernorm 后的 q_c", meaning: "indexer 复用 MLA 的 Q 低秩压缩，forward 里直接传入 q_c。" },
  { symbol: "k_norm", resolvesTo: "LayerNorm(128, eps=1e-6)", meaning: "全模型唯一带 bias 的 norm。" },
  { symbol: "quant_block_size=128", resolvesTo: "恰为一个 head_dim", meaning: "q 逐 (token, head) 分组 FP8 量化。" },
];

const IDX_FWD_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "Indexer.prepare_inputs：q/k 上投影、切 nope|pe、独立 interleaved RoPE", location: "hy_v4/nvidia/attention.py · L219-L245", url: `${ATTN_URL}#L219-L245`, code: code(`
q, _ = self.wq_b(qr)
q = q.view(-1, self.n_head, self.head_dim)
# Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
q_nope, q_pe = torch.split(
    q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
)

kw, _ = self.wk_weights_proj(hidden_states)
k = kw[:, : self.head_dim]
weights = kw[:, self.head_dim :]

k = self.k_norm(k)
k_nope, k_pe = torch.split(
    k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
)

q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))`) },
  { stage: "2 · FORWARD", title: "q FP8 量化（ue8m0）+ per-head 权重折叠", location: "hy_v4/nvidia/attention.py · L247-L263", url: `${ATTN_URL}#L247-L263`, code: code(`
# Only q is quantized here; k quantization is fused with cache insertion.
q = q.view(-1, self.head_dim)
q_fp8, q_scale = per_token_group_quant_fp8(
    q,
    self.quant_block_size,
    column_major_scales=False,
    use_ue8m0=self.scale_fmt is not None,
)
q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
q_scale = q_scale.view(-1, self.n_head, 1)

weights = (
    weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
)`) },
];
const IDX_FWD_SYMBOLS: CodeSymbol[] = [
  { symbol: "weights 折叠", resolvesTo: "w_proj × q_scale × 128^-0.5 × 32^-0.5", meaning: "两个 scale 因子都折进 per-head 权重，kernel 里只剩加权和。" },
  { symbol: "k 量化", resolvesTo: "与 cache 写入融合", meaning: "indexer 自有 FP8 cache：fp8 k(128B) + fp32 scale(4B) = 132 B/token。" },
  { symbol: "score 公式", resolvesTo: "(1/√32)(1/√128)Σ_h w_h⟨q_h,k⟩", meaning: "无 sigmoid/relu，纯加权和；prefill 走 fp8_fp4_mqa_logits。" },
];

const TOPK_SECTIONS: CodeSection[] = [
  { stage: "1 · ENTER", title: "buffer 预填 −1 哨兵，top-k kernel 散射写入", location: "layers/sparse_attn_indexer.py · L426-L432,L500-L518", url: `${INDEXER_OPS_URL}#L500-L518`, code: code(`
# The buffer must be pre-filled with -1 (the "no token" sentinel) before the
# top-k kernels scatter valid indices into it.
if not skip_topk_buffer_clear:
    topk_indices_buffer[: hidden_states.shape[0]] = -1
...
logits = fp8_fp4_mqa_logits(
    (q_slice_cast, q_scale_slice),
    (k_quant_cast, k_scale_cast),
    weights[chunk.token_start : chunk.token_end],
    cu_seqlen_ks,
    cu_seqlen_ke,
    clean_logits=False,
)
ops.top_k_per_row_prefill(
    logits, cu_seqlen_ks, cu_seqlen_ke,
    topk_indices, num_rows,
    logits.stride(0), logits.stride(1), topk_tokens,
)`) },
  { stage: "2 · INIT", title: "buffer 在 HYV4Model 一次性分配，逐层传入（IndexCache）", location: "hy_v4/nvidia/model.py · L226-L235", url: `${CODE_URL}#L226-L235`, code: code(`
self.is_sparse = hasattr(config, "index_topk")
if self.is_sparse:
    self.topk_indices_buffer = torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        config.index_topk,
        dtype=torch.int32,
        device=self.device,
)`) },
];
const TOPK_SYMBOLS: CodeSymbol[] = [
  { symbol: "topk_tokens", resolvesTo: "index_topk = 2048", meaning: "decode 小批走 cooperative_topk（≤64 行），大批走 persistent_topk。" },
  { symbol: "全模型唯一一份", resolvesTo: "[max_num_batched_tokens, 2048]", meaning: "21 个 full 层写入，57 个 shared 层直接读——跨层复用的物理载体。" },
];

const SHARE_SECTIONS: CodeSection[] = [
  { stage: "1 · INIT", title: "shared 层判定：skip_topk → 不构建 indexer 模块", location: "hy_v4/nvidia/attention.py · L313-L331", url: `${ATTN_URL}#L313-L331`, code: code(`
# Only actual sparse layers may share another layer's top-k indices.
self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
    config
)
# The skip pattern only governs backbone layers. MTP/nextn layers
# (layer_id >= num_hidden_layers) always build a full indexer: they
# compute indices at draft step 0 and toggle at runtime.
...
self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)`) },
  { stage: "2 · FORWARD", title: "shared 层 forward：跳过 indexer，MLA 直接按 buffer 行做稀疏 attention", location: "hy_v4/nvidia/attention.py · L756-L765", url: `${ATTN_URL}#L756-L765`, code: code(`
if self.indexer is not None and self.is_sparse and not self.skip_topk:
    self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
out.copy_(
    self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=out.shape,
    )
)`) },
  { stage: "3 · ENTER", title: "MTP 复用同一 buffer：step 0 自算、step 1+ 切换 skip", location: "hy_v4/nvidia/mtp.py · L469-L489", url: `${MTP_URL}#L469-L489`, code: code(`
def set_skip_topk(self, skip: bool) -> None:
    """Toggle the draft indexer for \`\`index_share_for_mtp_iteration\`\`.

    The proposer clears the flag for draft step 0 so the MTP layer builds
    its own top-k indices, then sets it for steps 1+ so they reuse what
    step 0 wrote into the shared buffer instead of re-running the indexer.
    """
    for layer in self.layers.values():
        layer.mtp_block.self_attn.skip_topk = skip`) },
];
const SHARE_SYMBOLS: CodeSymbol[] = [
  { symbol: "compute_skip_topk_layers", resolvesTo: "indexer_types 里 'shared' 的层号", meaning: "config 直接给 78 项逐层声明：full = {0,1,5,9,…,77} 共 21 层。" },
  { symbol: "indexer 权重", resolvesTo: "加载时丢弃", meaning: "shared 层 checkpoint 里的 indexer.* 权重被 is_skip_topk_indexer_weight 过滤。" },
  { symbol: "MTP 层", resolvesTo: "强制 full indexer", meaning: "draft step 0 自算 top-k，step 1+ 复用 buffer（proposer 切换开关）。" },
];

const MOE_INIT_SECTIONS: CodeSection[] = [
  { stage: "1 · INIT", title: "HYV4MoEFused.__init__：fp32 gate + shared experts + FusedMoEFactory", location: "hy_v4/nvidia/moe.py · L122-L170", url: `${MOE_URL}#L122-L170`, code: code(`
self.gate = GateLinear(
    config.hidden_size,
    config.num_experts,
    bias=False,
    out_dtype=torch.float32,
    params_dtype=torch.float32,
    prefix=f"{prefix}.gate",
)
...
self.shared_experts = HYV4FeedForward(
    hidden_size=config.hidden_size,
    intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
    hidden_act=config.hidden_act,
    quant_config=quant_config,
    prefix=f"{prefix}.shared_experts",
    reduce_results=False,
)
...
self.experts = FusedMoEFactory(
    num_experts=self.n_routed_experts,
    top_k=top_k,
    hidden_size=config.hidden_size,
    intermediate_size=intermediate_size,
    renormalize=config.route_norm,
    scoring_func="sigmoid",
    use_grouped_topk=True,
    num_expert_group=1,
    topk_group=1,
    routed_scaling_factor=router_scaling_factor,
    e_score_correction_bias=self.expert_bias,
    shared_experts=self.shared_experts,
    swiglu_limit=moe_swiglu_limit,
)`) },
];
const MOE_INIT_SYMBOLS: CodeSymbol[] = [
  { symbol: "GateLinear", resolvesTo: "fp32 输出", meaning: "gate 权重运行时以 fp32 存储/计算；checkpoint 实存 bf16（字节对账确认）。" },
  { symbol: "num_expert_group=1", resolvesTo: "退化为全局 topk", meaning: "grouped topk 的组数为 1；config 的 n_group/topk_group 字段不被读取。" },
  { symbol: "shared_experts=", resolvesTo: "挂进 FusedMoE", meaning: "shared expert 在 FusedMoE 内部一起算，加法由其统一做。" },
];

const ROUTER_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "路由数学（eager 参考实现）：σ + bias 选、原始 σ 算、renorm", location: "layers/fused_moe/router/fused_topk_bias_router.py · L260-L290", url: `${ROUTER_URL}#L260-L290`, code: code(`
scores = gating_output.sigmoid()
if e_score_correction_bias is not None:
    scores_for_choice = scores.view(
        -1, n_routed_experts
    ) + e_score_correction_bias.unsqueeze(0)
else:
    scores_for_choice = scores.view(-1, n_routed_experts)
topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[
    1
]
topk_weights = scores.gather(1, topk_indices)
if renormalize:
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
topk_weights = topk_weights.to(torch.float32)
if routed_scaling_factor != 1.0:
    topk_weights *= routed_scaling_factor`) },
];
const ROUTER_SYMBOLS: CodeSymbol[] = [
  { symbol: "选与算分离", resolvesTo: "scores_for_choice vs scores", meaning: "topk 用 σ+b 的分数；混合权重用原始 σ 分数 gather 再 renorm。" },
  { symbol: "e_score_correction_bias", resolvesTo: "checkpoint mlp.gate.e_score_correction_bias", meaning: "加载时改名 expert_bias，运行时 fp32。" },
  { symbol: "生产路径", resolvesTo: "vllm_topk_sigmoid 融合 kernel", meaning: "本段是数学等价的参考实现。" },
];

const CLAMP_SECTIONS: CodeSection[] = [
  { stage: "1 · ENTER", title: "SwiGLU 限幅（仅 routed 专家）：gate 只限上界、up 双边", location: "layers/fused_moe/utils.py · L536-L549", url: `${MOE_UTILS_URL}#L536-L549`, code: code(`
d = input.shape[1] // 2
gate = input[:, :d]
up = input[:, d:]

if swiglu_limit > 0:
    gate = torch.clamp(gate, max=swiglu_limit)
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

output.copy_(F.silu(gate) * up)`) },
  { stage: "2 · ENTER", title: "类 docstring：dense 层与 shared 专家不限幅", location: "hy_v4/nvidia/moe.py · L70-L85", url: `${MOE_URL}#L70-L85`, code: code(`
"""When \`\`config.swiglu_limit > 0\`\` the routed experts use a clamped SwiGLU::

    gate = clamp(gate, max=limit)
    up = clamp(up, -limit, limit)
    output = silu(gate) * up

Dense layers and shared experts are NOT clamped. The clamp is forwarded to
\`\`FusedMoEFactory\`\` as \`\`swiglu_limit\`\` so it travels through the quant
config to every expert backend instead of mutating global state.
"""`) },
];
const CLAMP_SYMBOLS: CodeSymbol[] = [
  { symbol: "swiglu_limit", resolvesTo: "10.0", meaning: "≤0 时整段 clamp 跳过；仅作用于 routed experts。" },
];

const DENSE_FFN_SECTIONS: CodeSection[] = [
  { stage: "1 · FORWARD", title: "HYV4FeedForward：L0 dense FFN 与 MoE shared expert 共用，无限幅", location: "hy_v4/nvidia/moe.py · L23-L66", url: `${MOE_URL}#L23-L66`, code: code(`
class HYV4FeedForward(nn.Module):
    """Dense SwiGLU feed-forward block.

    Used both for the dense decoder layers and for the MoE shared experts.
    The routed-expert SwiGLU clamp does not apply here, matching the
    reference implementation.
    """
...
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
    quant_config=quant_config,
    reduce_results=reduce_results,
    prefix=f"{prefix}.down_proj",
)
self.act_fn = SiluAndMul()

def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    out = self.act_fn(gate_up)
    out, _ = self.down_proj(out)
    return out`) },
];
const DENSE_FFN_SYMBOLS: CodeSymbol[] = [
  { symbol: "hidden_act", resolvesTo: "只接受 silu", meaning: "构造时硬校验；L0 中间维 18432，shared expert 中间维 2048。" },
  { symbol: "SiluAndMul", resolvesTo: "无 clamp", meaning: "与 routed expert 的 clamp SwiGLU 是本模型的关键差异点。" },
];

const CODE_BY_ID: Record<string, CodeDetail> = {};
for (const id of ["hc-attnpre", "hc-mlppre"]) CODE_BY_ID[id] = { sections: HC_PRE_SECTIONS, symbols: HC_PRE_SYMBOLS };
for (const id of ["hc-attnpost", "hc-mlppost"]) CODE_BY_ID[id] = { sections: HC_POST_SECTIONS, symbols: HC_POST_SYMBOLS };
for (const id of ["norm-in", "norm-post"]) CODE_BY_ID[id] = { sections: NORM_SECTIONS, symbols: [] };
CODE_BY_ID["a-fused"] = { sections: MLA_INIT_SECTIONS, symbols: MLA_INIT_SYMBOLS };
for (const id of ["a-split", "a-qnorm", "a-qup", "a-kvsplit", "a-kvnorm", "a-rope"]) CODE_BY_ID[id] = { sections: MLA_FWD_SECTIONS, symbols: MLA_FWD_SYMBOLS };
CODE_BY_ID["a-kvb"] = { sections: ABSORB_SECTIONS, symbols: ABSORB_SYMBOLS };
CODE_BY_ID["a-core"] = { sections: SINK_SECTIONS, symbols: SINK_SYMBOLS };
CODE_BY_ID["a-gate"] = { sections: GATED_SECTIONS, symbols: GATED_SYMBOLS };
CODE_BY_ID["a-oproj"] = { sections: MLA_FWD_SECTIONS, symbols: MLA_FWD_SYMBOLS };
CODE_BY_ID["i-qup"] = { sections: IDX_INIT_SECTIONS, symbols: IDX_INIT_SYMBOLS };
CODE_BY_ID["i-kw"] = { sections: IDX_INIT_SECTIONS, symbols: IDX_INIT_SYMBOLS };
for (const id of ["i-knorm", "i-rope", "i-quant", "i-logits"]) CODE_BY_ID[id] = { sections: IDX_FWD_SECTIONS, symbols: IDX_FWD_SYMBOLS };
CODE_BY_ID["i-topk"] = { sections: TOPK_SECTIONS, symbols: TOPK_SYMBOLS };
CODE_BY_ID["i-buffer"] = { sections: SHARE_SECTIONS, symbols: SHARE_SYMBOLS };
CODE_BY_ID["m-router"] = { sections: ROUTER_SECTIONS, symbols: ROUTER_SYMBOLS };
CODE_BY_ID["m-experts"] = { sections: [...MOE_INIT_SECTIONS, ...CLAMP_SECTIONS], symbols: [...MOE_INIT_SYMBOLS, ...CLAMP_SYMBOLS] };
for (const id of ["m-shared", "m-sum"]) CODE_BY_ID[id] = { sections: MOE_INIT_SECTIONS, symbols: MOE_INIT_SYMBOLS };
for (const id of ["d-gateup", "d-split", "d-act", "d-down"]) CODE_BY_ID[id] = { sections: DENSE_FFN_SECTIONS, symbols: DENSE_FFN_SYMBOLS };

const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
  position: [{ kind: "external", label: "num_computed_tokens + query offsets", shape: "[B] + [Nq]", from: "vLLM GPUModelRunner 请求调度状态" }],
  attnmeta: [{ kind: "external", label: "query_start_loc · seq_lens · causal", shape: "[B+1] + [B] + bool", from: "vLLM CommonAttentionMetadata" }],
  slots: [{ kind: "external", label: "positions + block_table", shape: "[Nq] + [B,Nblocks]", from: "runner positions 与 KV cache manager" }],
  "a-fused": [{ kind: "upstream", label: "û · input_layernorm 输出", shape: "[T,6144]", from: "Input RMSNorm 输出" }],
  "a-qnorm": [{ kind: "upstream", label: "q_c", shape: "[T,2048]", from: "Split q_c / kv_lora 输出" }],
  "a-qup": [{ kind: "upstream", label: "q_cⁿ · norm 后 Q 低秩", shape: "[T,2048]", from: "Q Latent RMSNorm 输出（indexer.wq_b 也读它）" }],
  "a-kvnorm": [{ kind: "upstream", label: "kv_c", shape: "[T,512]", from: "Split kv_c / k_pe 输出" }],
  "a-kvb": [{ kind: "upstream", label: "c_n · norm 后 KV latent", shape: "[T,512]", from: "KV Latent RMSNorm 输出" }],
  "a-rope": [{ kind: "upstream", label: "q（rope 切片）· k_pe", shape: "[T,64,64] + [T,1,64]", from: "Q 上投影与 Split kv 输出" }, { kind: "external", label: "positions", shape: "[Nq]", from: "Build Position IDs 输出" }],
  "a-cache": [{ kind: "upstream", label: "c_n + k_peᵣ", shape: "[T,512] + [T,64]", from: "KV Latent RMSNorm 与 Interleaved RoPE 输出" }, { kind: "external", label: "slot_mapping + block_table", shape: "[Nq] + [B,Nblocks]", from: "Resolve KV Slots 输出" }],
  "a-core": [{ kind: "upstream", label: "qᵣ + latent cache", shape: "[T,64,256] + cache pages", from: "Interleaved RoPE 与 Latent KV Cache 输出" }, { kind: "upstream", label: "Top-2048 indices", shape: "[T,2048]", from: "full 层：Indexer Top-k；shared 层：共享 buffer" }],
  "a-gate": [{ kind: "upstream", label: "attn heads", shape: "[T,16384]", from: "Sparse MLA 输出" }, { kind: "external", label: "û（gate 输入，norm 后 hidden）", shape: "[T,6144]", from: "Input RMSNorm 输出旁路" }],
  "a-oproj": [{ kind: "upstream", label: "gated heads", shape: "[T,16384]", from: "σ gate 相乘后的输出" }],
  "i-qup": [{ kind: "upstream", label: "q_cⁿ · norm 后 Q 低秩", shape: "[T,2048]", from: "复用 MLA 的 Q Latent RMSNorm（不再单独下投影）" }],
  "i-kw": [{ kind: "upstream", label: "û", shape: "[T,6144]", from: "Input RMSNorm 输出（indexer key 与 per-head 权重都从它投影）" }],
  "i-rope": [{ kind: "upstream", label: "q_pe / k_pe 切片", shape: "[T,32,64] + [T,64]", from: "Indexer Q/K 上投影与 k_norm 输出" }, { kind: "external", label: "positions", shape: "[Nq]", from: "Build Position IDs 输出" }],
  "i-logits": [{ kind: "upstream", label: "q_fp8 + k cache", shape: "[T,32,128] + fp8 cache", from: "FP8 量化输出与 indexer cache（k 量化在写入时融合）" }],
  "i-topk": [{ kind: "upstream", label: "token scores", shape: "[T,T]", from: "MQA logits 输出（fp32）" }],
  "i-buffer": [{ kind: "external", label: "topk_indices_buffer[t]", shape: "[T,2048] int32", from: "同一次 forward 内最近一个 full 层的 indexer 写入" }],
  "m-router": [{ kind: "upstream", label: "û′ · post-attn norm 输出", shape: "[T,6144]", from: "Post-attn RMSNorm（经 iHC mlp pre 压缩后再 norm）" }],
  "m-experts": [{ kind: "upstream", label: "û′ + expert ids + weights", shape: "[T,6144] + 2×[T,8]", from: "FP32 Router 输出" }],
  "m-shared": [{ kind: "upstream", label: "û′（全部 token）", shape: "[T,6144]", from: "Post-attn RMSNorm 输出；不经过 Top-K" }],
  "m-sum": [{ kind: "upstream", label: "8 routed + 1 shared", shape: "9 × [T,6144]", from: "Routed Experts ×8 与 Shared Expert 输出" }],
  "d-gateup": [{ kind: "upstream", label: "û′", shape: "[T,6144]", from: "Post-attn RMSNorm 输出（仅 L0）" }],
  "hc-attnpre": [{ kind: "upstream", label: "Xₗ · iHC 通道状态", shape: "[T,4,6144]", from: "上一 decoder layer 输出（首层由 embedding 广播）" }],
  "hc-attnpost": [{ kind: "upstream", label: "Yattn + 4 通道残差 + H_post", shape: "[T,6144] + [T,4,6144] + [T,4]", from: "O Projection、iHC pre 与其 gate 输出" }],
  "hc-mlppre": [{ kind: "upstream", label: "U · iHC 通道状态", shape: "[T,4,6144]", from: "iHC attn post 输出" }],
  "hc-mlppost": [{ kind: "upstream", label: "Yffn/Ymoe + 4 通道残差 + H_post", shape: "[T,6144] + [T,4,6144] + [T,4]", from: "MoE / Dense FFN、iHC pre 与其 gate 输出" }],
};

const NEXT_BY_ID: Record<string, string> = {
  position: "Interleaved RoPE（MLA 与 indexer 各一次）", attnmeta: "Sparse MLA kernel（隐式 causal/padding）", slots: "Latent KV Cache · indexer FP8 cache",
  "hc-attnpre": "Input RMSNorm", "norm-in": "Fused 下投影（MLA）与 indexer key 投影", "hc-attnpost": "iHC mlp pre", "hc-mlppre": "Post-attn RMSNorm", "norm-post": "FP32 Router / Dense FFN", "hc-mlppost": "下一 decoder layer / iHC Head",
  "a-fused": "Split q_c / kv_lora", "a-split": "Q Latent RMSNorm · Split kv_c / k_pe", "a-qnorm": "Q 上投影 · Indexer wq_b", "a-qup": "Interleaved RoPE（rope 切片）", "a-kvsplit": "KV Latent RMSNorm · Interleaved RoPE(k_pe)", "a-kvnorm": "kv_b 上投影 / W_UK 吸收 · Latent KV Cache", "a-kvb": "Latent KV Cache（prefill）/ latent MQA（decode）", "a-rope": "Sparse MLA", "a-cache": "Sparse MLA（按 indices gather）", "a-core": "σ gate 相乘", "a-gate": "O Projection", "a-oproj": "iHC attn post",
  "i-qup": "Indexer interleaved RoPE（pe 切片）", "i-kw": "k_norm（LayerNorm 1e-6）", "i-knorm": "Indexer interleaved RoPE", "i-rope": "FP8 量化 + 权重折叠", "i-quant": "MQA logits", "i-logits": "Top-2048 → 共享 buffer", "i-topk": "本层 Sparse MLA + 后继 shared 层复用", "i-buffer": "本层 Sparse MLA",
  "m-router": "Routed Experts ×8", "m-experts": "Weighted Sum", "m-shared": "Weighted Sum", "m-sum": "iHC mlp post",
  "d-gateup": "Split Gate / Up", "d-split": "SiLU · Up", "d-act": "Down Projection", "d-down": "iHC mlp post",
};

function symbolicShape(shape: string) {
  return shape
    .replaceAll("[T,4,6144]", "[T,C,H]")
    .replaceAll("[T,2624]", "[T,R_q+L_k+D_r]")
    .replaceAll("[T,16384]", "[T,N_h·D_qk]")
    .replaceAll("[T,36864]", "[T,2H_dense]")
    .replaceAll("[T,18432]", "[T,H_dense]")
    .replaceAll("[16384,2048]", "[N_h·D_qk,R_q]")
    .replaceAll("[28672,512]", "[N_h·(D_n+D_v),L_k]")
    .replaceAll("[2048,6144]", "[R_q,H]")
    .replaceAll("[576,6144]", "[L_k+D_r,H]")
    .replaceAll("[6144,16384]", "[H,N_h·D_qk]")
    .replaceAll("[16384,6144]", "[N_h·D_qk,H]")
    .replaceAll("[4096,2048]", "[N_idx·D_idx,R_q]")
    .replaceAll("[256,6144]", "[E,H]")
    .replaceAll("[256,4096,6144]", "[E,2H_e,H]")
    .replaceAll("[256,6144,2048]", "[E,H,H_e]")
    .replaceAll("[6144,2048]", "[H,H_e]")
    .replaceAll("[2048,6144]", "[R_q,H]")
    .replaceAll("[18432,6144]", "[H_dense,H]")
    .replaceAll("[6144,18432]", "[H,H_dense]")
    .replaceAll("[8,24576]", "[2C,C·H]")
    .replaceAll("[4,24576]", "[C,C·H]")
    .replaceAll("[120832,6144]", "[V,H]")
    .replaceAll("24576", "C·H")
    .replaceAll("18432", "H_dense")
    .replaceAll("16384", "N_h·D_qk")
    .replaceAll("120832", "V")
    .replaceAll("[T,6144]", "[T,H]")
    .replaceAll("6144", "H");
}

const op = (v: Partial<OpNode> & { id: string; kind: OpKind; title: string; kicker: string }): OpNode => ({
  tone: "projection", summary: "", input: "", inputShape: "", output: "", outputShape: "",
  formula: "", formulaNote: "", runtime: "", source: "", sourceUrl: ATTN_URL, code: "", weights: [],
  latex: LATEX_BY_ID[v.id], codeSections: CODE_BY_ID[v.id]?.sections, codeSymbols: CODE_BY_ID[v.id]?.symbols,
  ...v,
});

function runtimeNodes(): Record<string, OpNode> {
  return {
    position: op({ id: "position", kind: "route", tone: "index", kicker: "vLLM RUNTIME I/O", title: "Build Position IDs", input: "num_computed_tokens + query offsets", inputShape: "[B] + [Nq]", output: "positions", outputShape: "[Nq]", formula: "position(req,i)=num_computed_tokens[req]+i", formulaNote: "positions 由 vLLM runner 在模型 forward 之前构造，MLA RoPE 与 indexer RoPE 各消费一次。", source: "gpu_model_runner.py · _prepare_inputs", sourceUrl: RUNNER_URL, weights: [] }),
    attnmeta: op({ id: "attnmeta", kind: "mask", tone: "attention", kicker: "vLLM RUNTIME I/O", title: "Build Attention Metadata", input: "query_start_loc, seq_lens, causal=True", inputShape: "[B+1] · [B] · bool", output: "implicit causal / padding layout", outputShape: "backend metadata; 非稠密 [S,T]", formula: "valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)", formulaNote: "同一份边界元数据约束 indexer 的 top-k 与稀疏 MLA kernel。", source: "gpu_model_runner.py · CommonAttentionMetadata", sourceUrl: RUNNER_URL, weights: [] }),
    slots: op({ id: "slots", kind: "route", tone: "index", kicker: "vLLM RUNTIME I/O", title: "Resolve KV Slots", input: "positions + block_table", inputShape: "[Nq] + [B,Nblocks]", output: "slot_mapping + block_table", outputShape: "[Nq] + [B,Nblocks]", formula: "slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)", formulaNote: "Hy4 有两套 cache：MLA latent cache（576 元素/token）与 indexer FP8 cache（132 B/token，仅 full 层）。", source: "gpu_model_runner.py · compute_slot_mapping", sourceUrl: RUNNER_URL, weights: [] }),
  };
}

function chainNodes(layer: number): Record<string, OpNode> {
  return {
    "hc-attnpre": op({ id: "hc-attnpre", kind: "linear", tone: "norm", kicker: `L${layer} · iHC PRE · ATTN 子块`, title: "iHC pre-gate · 通道压缩", summary: `把 [T,4,6144] 的 4 条残差通道按 per-token sigmoid gate 加权求和，压缩成子块输入 y；同一个 hc_fn 还顺带算出 post gate（×2.0 幅度）留给出口用。fp32 计算。`, input: "Xₗ · iHC 通道状态", inputShape: "[T,4,6144]", output: "y + H_post", outputShape: "[T,6144] + [T,4]", formulaNote: "iHC 是残差的推广：hc=1 且 gate 恒 1 时退化为 pre-norm 残差流。H_post = 2.0·σ(·)+1e-6。", runtime: "HYV4HCPreLayer.forward（HPC 单 kernel：VLLM_ENABLE_HPC_OPS + sm100/103）", source: "hy_v4/nvidia/hc.py · HYV4HCPreLayer.forward · L97-L139", sourceUrl: `${HC_URL}#L97-L139`, weights: hcWeights(layer, "hc_attn_layer") }),
    "norm-in": op({ id: "norm-in", kind: "norm", tone: "norm", kicker: `L${layer} · PRE-NORM`, title: "RMSNorm · input_layernorm", summary: "对压缩后的 y 做标准 RMSNorm（无 Gemma 的 +1）；输出 û 同时进入 MLA 下投影、indexer key 投影和 gated-MLA 的门。", input: "y", inputShape: "[T,6144]", output: "û", outputShape: "[T,6144]", formulaNote: "ε=1e-5；输出被三个消费者复用（MLA / indexer / gate）。", runtime: "RMSNorm.forward", source: "hy_v4/nvidia/model.py · HYV4DecoderLayer · L191", sourceUrl: `${CODE_URL}#L187-L209`, weights: [wt(layer, "input_layernorm.weight", "[6144]", "BF16", "6,144")] }),
    "hc-attnpost": op({ id: "hc-attnpost", kind: "add", tone: "norm", kicker: "iHC POST · ATTN 子块", title: "iHC post · 多通道残差散射", summary: "把 attention 输出按 H_post 门散射回 4 条通道并加多通道残差：y[n,i,:] = H_post[n,i]·attn_out + x[n,i,:]。无参数。", input: "Yattn + residual + H_post", inputShape: "[T,6144] + [T,4,6144] + [T,4]", output: "U · 更新后通道", outputShape: "[T,4,6144]", formulaNote: "hc_post 无参数（纯逐元素）；iHC 下 DecoderLayer 返回的 residual 是 None——残差内嵌在通道里。", runtime: "HYV4HCPostLayer.forward", source: "hy_v4/nvidia/hc.py · HYV4HCPostLayer.forward · L163-L186", sourceUrl: `${HC_URL}#L163-L186`, weights: [] }),
    "hc-mlppre": op({ id: "hc-mlppre", kind: "linear", tone: "norm", kicker: "iHC PRE · MLP 子块", title: "iHC pre-gate · 通道压缩", summary: "与 attn 子块同构的第二个 iHC pre：压缩 4 通道得到 FFN/MoE 输入 y′，同时产出自己的 post gate。", input: "U · iHC 通道状态", inputShape: "[T,4,6144]", output: "y′ + H_post", outputShape: "[T,6144] + [T,4]", formulaNote: "每个子块（attn/mlp）各有一套独立的 hc_fn/hc_base/hc_scale。", runtime: "HYV4HCPreLayer.forward", source: "hy_v4/nvidia/hc.py · HYV4HCPreLayer.forward · L97-L139", sourceUrl: `${HC_URL}#L97-L139`, weights: hcWeights(layer, "hc_mlp_layer") }),
    "norm-post": op({ id: "norm-post", kind: "norm", tone: "norm", kicker: "PRE-FFN NORM", title: "RMSNorm · post_attention_layernorm", summary: "对 mlp 子块的压缩输入 y′ 做 RMSNorm，输出进入 dense FFN（L0）或 MoE 路由（L1-77）。", input: "y′", inputShape: "[T,6144]", output: "û′", outputShape: "[T,6144]", formulaNote: "ε=1e-5。", runtime: "RMSNorm.forward", source: "hy_v4/nvidia/model.py · HYV4DecoderLayer · L198", sourceUrl: `${CODE_URL}#L187-L209`, weights: [wt(layer, "post_attention_layernorm.weight", "[6144]", "BF16", "6,144")] }),
    "hc-mlppost": op({ id: "hc-mlppost", kind: "add", tone: "norm", kicker: "iHC POST · MLP 子块", title: "iHC post · 多通道残差散射", summary: "把 FFN/MoE 输出按 H_post 散射回 4 通道并加残差，得到 Xₗ₊₁ 传给下一层；无参数。", input: "Yffn/Ymoe + residual + H_post", inputShape: "[T,6144] + [T,4,6144] + [T,4]", output: "Xₗ₊₁", outputShape: "[T,4,6144]", formulaNote: "层间流动的是 [T,4,6144]；PP 传输时 flatten 成 [T,24576]。", runtime: "HYV4HCPostLayer.forward", source: "hy_v4/nvidia/hc.py · HYV4HCPostLayer.forward · L163-L186", sourceUrl: `${HC_URL}#L163-L186`, weights: [] }),
  };
}

function attentionNodes(layer: number, hasIndexer: boolean): Record<string, OpNode> {
  const nodes: Record<string, OpNode> = {
    "a-fused": op({ id: "a-fused", kind: "linear", tone: "projection", kicker: "MLA · FUSED 下投影 · replicated", title: "fused_qkv_a_proj", summary: "checkpoint 里分开的 q_a_proj（6144→2048）与 kv_a_proj_with_mqa（6144→576）读同一份 û，加载时合并成一个 disable_tp 的 GEMM，一次产出 Q 低秩激活与 KV latent。", input: "û", inputShape: "[T,6144]", output: "qkv_lora", outputShape: "[T,2624]", formulaNote: "两块权重都是 TP 全复制（disable_tp=True），合并后每卡仍全量。", runtime: "MergedColumnParallelLinear · disable_tp=True", source: "hy_v4/nvidia/attention.py · L361-L372", sourceUrl: `${ATTN_URL}#L361-L372`, weights: [wt(layer, "self_attn.q_a_proj.weight", "[2048,6144]", "BF16", "12.58M", "fused_qkv_a_proj · q_a 半段", "TP replicated"), wt(layer, "self_attn.kv_a_proj_with_mqa.weight", "[576,6144]", "BF16", "3.54M", "fused_qkv_a_proj · kv_a 半段", "TP replicated")] }),
    "a-split": op({ id: "a-split", kind: "split", tone: "projection", kicker: "MLA · SPLIT", title: "Split q_c / kv_lora", summary: "沿最后一维切出 Q 低秩激活 q_c（2048）与 KV 侧 kv_lora（576）。", input: "qkv_lora", inputShape: "[T,2624]", output: "q_c · kv_lora", outputShape: "2048 · 576", formulaNote: "只切分，无数值计算。", runtime: "Tensor.split", source: "hy_v4/nvidia/attention.py · L669-L682", sourceUrl: `${ATTN_URL}#L664-L691`, weights: [] }),
    "a-qnorm": op({ id: "a-qnorm", kind: "norm", tone: "norm", kicker: "MLA · Q LATENT NORM", title: "q_a_layernorm", summary: "RMSNorm(2048) 作用在 Q 低秩激活上；norm 后的 q_cⁿ 同时被 q_b_proj 与 indexer.wq_b 消费。", input: "q_c", inputShape: "[T,2048]", output: "q_cⁿ", outputShape: "[T,2048]", runtime: "RMSNorm.forward", source: "hy_v4/nvidia/attention.py · L675", sourceUrl: `${ATTN_URL}#L664-L691`, weights: [wt(layer, "self_attn.q_a_layernorm.weight", "[2048]", "BF16", "2,048", undefined, "TP replicated")] }),
    "a-qup": op({ id: "a-qup", kind: "linear", tone: "projection", kicker: "MLA · Q 上投影 · 64×256", title: "q_b_proj", summary: "把 norm 后的 Q 低秩激活上投影成 64 head × 256 维（每 head nope 192 在前 + rope 64 在后）。", input: "q_cⁿ", inputShape: "[T,2048]", output: "q", outputShape: "[T,64,256]", formulaNote: "TP 列切：每卡 64/TP 个 head。", runtime: "ColumnParallelLinear", source: "hy_v4/nvidia/attention.py · L386-L392", sourceUrl: `${ATTN_URL}#L381-L421`, weights: [wt(layer, "self_attn.q_b_proj.weight", "[16384,2048]", "BF16", "33.55M", "TP 列切 [1024,2048]")] }),
    "a-kvsplit": op({ id: "a-kvsplit", kind: "split", tone: "projection", kicker: "MLA · SPLIT", title: "Split kv_c / k_pe", summary: "把 kv_lora 切成共享压缩向量 kv_c（512）与解耦的 per-token rope key k_pe（64）。", input: "kv_lora", inputShape: "[T,576]", output: "kv_c · k_pe", outputShape: "512 · 64", runtime: "Tensor.split", source: "hy_v4/nvidia/attention.py · L683", sourceUrl: `${ATTN_URL}#L664-L691`, weights: [] }),
    "a-kvnorm": op({ id: "a-kvnorm", kind: "norm", tone: "norm", kicker: "MLA · KV LATENT NORM", title: "kv_a_layernorm", summary: "RMSNorm(512) 只作用于 kv_c（k_pe 不 norm）；norm 后的 c_n 进上投影与 latent cache。", input: "kv_c", inputShape: "[T,512]", output: "c_n", outputShape: "[T,512]", runtime: "RMSNorm.forward", source: "hy_v4/nvidia/attention.py · L684", sourceUrl: `${ATTN_URL}#L664-L691`, weights: [wt(layer, "self_attn.kv_a_layernorm.weight", "[512]", "BF16", "512", undefined, "TP replicated")] }),
    "a-kvb": op({ id: "a-kvb", kind: "linear", tone: "projection", kicker: "MLA · kv_b 上投影 / W_UK 吸收", title: "kv_b_proj → W_UK · W_UV", summary: "prefill：c_n 上投影成每 head k_nope(192)+v(256)；decode：加载后按 head 拆成 W_UK/W_UV 并预转置，q_nope 先乘 W_UKᵀ 得 latent query，对 (kv_c+k_pe) 的 576 维 latent cache 直接 MQA，输出再经 W_UV 回 v 空间。", input: "c_n", inputShape: "[T,512]", output: "per-head K/V（prefill）· latent MQA（decode）", outputShape: "[T,64,448] / [B,64,512]", formulaNote: "W_UK_T=permute(1,2,0)→(64,192,512)、W_UV=transpose(0,1)→(64,512,256)；吸收把 decode 的 QK 变成一次 batched GEMM。", runtime: "ColumnParallelLinear / replace_parameter + torch.bmm", source: "layers/attention/mla_attention.py · L1238-L1246", sourceUrl: `${MLA_ATTN_URL}#L1238-L1246`, weights: [wt(layer, "self_attn.kv_b_proj.weight", "[28672,512]", "BF16", "14.68M", "TP 列切 [1792,512]；加载后拆 W_UK/W_UV 预转")] }),
    "a-rope": op({ id: "a-rope", kind: "rope", tone: "attention", kicker: "MLA · INTERLEAVED ROPE · θ=10⁷", title: "Interleaved RoPE（rope 切片）", summary: "只旋转每 head 的后 64 维与 k_pe；interleaved（is_neox_style=False，PTM 布局）——indexer 注释明确说换 NeoX 布局会破坏 DSA top-k。", input: "q[...,192:] + k_pe + positions", inputShape: "[T,64,64] + [T,1,64] + [Nq]", output: "qᵣ · k_peᵣ", outputShape: "same", formulaNote: "nope 192 维原样通过；1M 位置编码。", runtime: "get_rope(is_neox_style=False)", source: "hy_v4/nvidia/attention.py · L416-L421,L687-L691", sourceUrl: `${ATTN_URL}#L416-L421`, weights: [] }),
    "a-cache": op({ id: "a-cache", kind: "cache", tone: "attention", kicker: "MLA · LATENT KV CACHE · 576/token", title: "Latent KV Cache 写入", summary: "把 c_n 与 RoPE 后的 k_peᵣ 一起写入 latent cache：每 token 576 元素（fp8_ds_mla 打包 656 B）；MQA 式单向量，全部 78 层共用同一布局。", input: "c_n + k_peᵣ + slot_mapping", inputShape: "[T,512] + [T,64] + [Nq]", output: "latent KV cache", outputShape: "[pages,576]", formulaNote: "num_kv_heads=1（latent 共享）；indexer 另有一套 FP8 cache（132 B/token，仅 full 层）。", runtime: "MLAAttention kv cache（fp8_ds_mla 布局）", source: "layers/attention/mla_attention.py · L1262-L1288", sourceUrl: `${MLA_ATTN_URL}#L1262-L1288`, weights: [] }),
    "a-core": op({ id: "a-core", kind: "softmax", tone: "attention", kicker: "MLA · SPARSE + LEARNABLE SINK", title: "Sparse MLA · Top-2048 + sink", summary: "只在 indexer 选出的 token 上做 attention；每 head 一个可学习 sink logit κ_h 折进 softmax 分母（虚拟 token）。prefill 走 flash_mla_sparse_fwd（稀疏 MHA），decode 走吸收后的 latent MQA；sink 强制 prefill 也走稀疏路径。", input: "qᵣ + latent cache + ℐ_t", inputShape: "[T,64,256] + cache + [T,2048]", output: "attn heads", outputShape: "[T,16384]", formulaNote: "scale=256^-0.5；out *= exp(lse)/(exp(lse)+exp(sink)) 等价于分母加 e^κ；序列 ≤2048 时 top-k 覆盖全部 token，数值等价 dense。", runtime: "flash_mla_sparse_fwd / flash_mla_with_kvcache(indices)", source: "hy_v4/nvidia/flashmla_sparse.py · L44-L51", sourceUrl: `${FLASHMLA_URL}#L44-L51`, weights: [wt(layer, "self_attn.learnable_sink_param", "[64]", "F32", "64", "kernel attn_sink 参数", "kernels require fp32 sinks；TP 按头切 [4]")] }),
    "a-gate": op({ id: "a-gate", kind: "activation", tone: "attention", kicker: "GATED MLA · ELEMENTWISE", title: "σ gate 相乘", summary: "attention 输出在 o_proj 之前乘 σ(linear_gate(û))：elementwise 模式即每 head 每 v 维一个门（64×256=16384），恰好满足 HPC 融合核的 256 对齐。门输入是 input_layernorm 后的 û。", input: "attn heads + û", inputShape: "[T,16384] + [T,6144]", output: "gated heads", outputShape: "[T,16384]", formulaNote: "headwise 模式每 head 1 个标量门（广播），Hy4 config 为 elementwise。", runtime: "ColumnParallelLinear + sigmoid ⊙（HPC：hpc_gated_mla_gemm 单 launch）", source: "hy_v4/nvidia/attention.py · L710-L733", sourceUrl: `${ATTN_URL}#L710-L733`, weights: [wt(layer, "self_attn.linear_gate.weight", "[16384,6144]", "BF16", "100.66M", "TP 按头列切 [1024,6144]")] }),
    "a-oproj": op({ id: "a-oproj", kind: "linear", tone: "projection", kicker: "MLA · 输出投影", title: "o_proj", summary: "64 head × 256 维拼成 16384 通道，投回模型宽度 6144，交给 iHC post。", input: "gated heads", inputShape: "[T,16384]", output: "Yattn", outputShape: "[T,6144]", runtime: "RowParallelLinear", source: "hy_v4/nvidia/attention.py · L409-L415", sourceUrl: `${ATTN_URL}#L381-L421`, weights: [wt(layer, "self_attn.o_proj.weight", "[6144,16384]", "BF16", "100.66M", "TP 行切 [6144,1024]")] }),
  };
  if (hasIndexer) {
    nodes["i-qup"] = op({ id: "i-qup", kind: "linear", tone: "index", kicker: "INDEXER · wq_b · replicated", title: "indexer q 上投影", summary: "复用 MLA 的 Q 低秩压缩：从 q_a_layernorm 后的 q_cⁿ 上投影出 32 head × 128 维 indexer query（nope 64 前 + rope 64 后）。", input: "q_cⁿ", inputShape: "[T,2048]", output: "q_idx", outputShape: "[T,32,128]", formulaNote: "ReplicatedLinear 不切分；输入不是 û 而是 MLA 的 q_cⁿ。", runtime: "ReplicatedLinear · wq_b", source: "hy_v4/nvidia/attention.py · L150-L157,L220", sourceUrl: `${ATTN_URL}#L143-L171`, weights: [wt(layer, "self_attn.indexer.wq_b.weight", "[4096,2048]", "BF16", "8.39M", "TP replicated")] });
    nodes["i-kw"] = op({ id: "i-kw", kind: "linear", tone: "index", kicker: "INDEXER · 融合 wk + weights_proj", title: "indexer key + per-head 权重", summary: "wk（6144→128）与 weights_proj（6144→32）融合成单 GEMM wk_weights_proj：一次产出 indexer key 与 32 个 per-head 标量权重。", input: "û", inputShape: "[T,6144]", output: "k · w", outputShape: "[T,128] + [T,32]", formulaNote: "disable_tp=True；FP8 checkpoint 的 wk 反量化成 BF16 后再融合。", runtime: "MergedColumnParallelLinear · wk_weights_proj", source: "hy_v4/nvidia/attention.py · L159-L171,L232-L234", sourceUrl: `${ATTN_URL}#L143-L171`, weights: [wt(layer, "self_attn.indexer.wk.weight", "[128,6144]", "BF16", "0.79M", "wk_weights_proj · k 半段", "TP replicated"), wt(layer, "self_attn.indexer.weights_proj.weight", "[32,6144]", "BF16", "0.20M", "wk_weights_proj · weights 半段", "TP replicated")] });
    nodes["i-knorm"] = op({ id: "i-knorm", kind: "norm", tone: "index", kicker: "INDEXER · LAYERNORM 1e-6", title: "k_norm（带 bias）", summary: "LayerNorm(128, eps=1e-6) 归一化 indexer key——全模型唯一带 bias 的 norm。", input: "k", inputShape: "[T,128]", output: "k̃", outputShape: "[T,128]", formulaNote: "eps 硬编码 1e-6（区别于主干 RMSNorm 的 1e-5）。", runtime: "nn.LayerNorm", source: "hy_v4/nvidia/attention.py · L167,L236", sourceUrl: `${ATTN_URL}#L143-L171`, weights: [wt(layer, "self_attn.indexer.k_norm.weight", "[128]", "BF16", "128", undefined, "checkpoint dtype 推断 BF16（11KB 级字节不可区分）"), wt(layer, "self_attn.indexer.k_norm.bias", "[128]", "BF16", "128", undefined, "全模型唯一带 bias 的 norm")] });
    nodes["i-rope"] = op({ id: "i-rope", kind: "rope", tone: "index", kicker: "INDEXER · INTERLEAVED ROPE", title: "indexer RoPE（pe 切片）", summary: "q_pe 与 k_pe 过独立的 interleaved RoPE（θ=10⁷），再拼回 nope 前 / pe 后的物理布局。", input: "q_pe + k_pe + positions", inputShape: "[T,32,64] + [T,64] + [Nq]", output: "qᵣ_idx · kᵣ", outputShape: "same", formulaNote: "128 维内部切分：nope 64 + rope 64。", runtime: "indexer_rope_emb", source: "hy_v4/nvidia/attention.py · L243-L259", sourceUrl: `${ATTN_URL}#L219-L245`, weights: [] });
    nodes["i-quant"] = op({ id: "i-quant", kind: "activation", tone: "index", kicker: "INDEXER · FP8 ue8m0", title: "FP8 量化 + 权重折叠", summary: "q 逐 (token, head) 分组 FP8 量化（group=128 恰好一个 head，ue8m0 scale）；per-head 权重折叠 w_h = w_proj·s_q·128^-0.5·32^-0.5，k 侧量化与 cache 写入融合。", input: "qᵣ_idx + w", inputShape: "[T,32,128] + [T,32]", output: "q_fp8 + w_h + fp8 k cache", outputShape: "[T,32,128] + [T,32] + 132B/token", formulaNote: "两个 scale 因子都折进 weights，kernel 里只剩加权和。", runtime: "per_token_group_quant_fp8（k 侧融合进 cache insert）", source: "hy_v4/nvidia/attention.py · L247-L263", sourceUrl: `${ATTN_URL}#L247-L263`, weights: [] });
    nodes["i-logits"] = op({ id: "i-logits", kind: "matmul", tone: "index", kicker: "INDEXER · MQA LOGITS", title: "加权 head 内积", summary: "score(t,s) = (1/√32)(1/√128)·Σ_h w_h(t)·⟨q_h(t), k(s)⟩——无激活函数的纯加权和；prefill 走 fp8_fp4_mqa_logits（DeepGEMM，fp8×fp8→fp32），decode 走 paged 版。", input: "q_fp8 + k cache + w_h", inputShape: "[T,32,128] + cache + [T,32]", output: "token scores", outputShape: "[T,T] fp32", formulaNote: "indexer cache 每 token 132 字节（fp8 k 128B + fp32 scale 4B），仅 21 个 full 层持有。", runtime: "fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits", source: "layers/sparse_attn_indexer.py · L500-L518", sourceUrl: `${INDEXER_OPS_URL}#L500-L518`, weights: [] });
    nodes["i-topk"] = op({ id: "i-topk", kind: "route", tone: "index", kicker: "INDEXER · TOP-2048", title: "Top-2048 → 共享 buffer", summary: "每个 query 行取 top index_topk=2048 写入全模型唯一的 topk_indices_buffer（[max_num_batched_tokens,2048] int32，预填 −1 哨兵）；后续 shared 层直接按行读。", input: "token scores", inputShape: "[T,T]", output: "ℐ_t", outputShape: "[T,2048]", formulaNote: "decode 小批（≤64 行）走 cooperative_topk，大批走 persistent_topk；2048 恰在白名单内。", runtime: "top_k_per_row_prefill / cooperative_topk / persistent_topk", source: "layers/sparse_attn_indexer.py · L500-L518,L617-L666", sourceUrl: `${INDEXER_OPS_URL}#L426-L432`, weights: [] });
  } else {
    nodes["i-buffer"] = op({ id: "i-buffer", kind: "cache", tone: "index", kicker: "SHARED INDEXER · 读共享 buffer", title: "读共享 Top-2048", summary: "shared 层不构建 indexer 模块（其 checkpoint indexer 权重加载时被丢弃），MLA 直接读同一次 forward 内最近一个 full 层写进 topk_indices_buffer 的行。", input: "topk_indices_buffer[t]", inputShape: "[T,2048] int32", output: "ℐ_t", outputShape: "[T,2048]", formulaNote: "full 层位置 = {0,1,5,9,…,77}（1+4k 规律，21 层）；每个 full 层后面跟 ≤3 个 shared 层。MTP 层 draft step 0 自算、step 1+ 复用。", runtime: "skip_topk=True → 直接 mla_attn(按 buffer 行)", source: "hy_v4/nvidia/attention.py · L756-L765", sourceUrl: `${ATTN_URL}#L756-L765`, weights: [] });
  }
  return nodes;
}

function moeNodes(layer: number): Record<string, OpNode> {
  return {
    "m-router": op({ id: "m-router", kind: "route", tone: "moe", kicker: "FP32 ROUTER · 256 → TOP-8", title: "Sigmoid 路由 + bias 修正", summary: "GateLinear 以 fp32 输出 256 个 logits；选专家用 σ(r)+bias 的分数，混合权重用原始 σ 分数再 renorm。n_group=topk_group=1，grouped topk 退化为全局 topk。", input: "û′", inputShape: "[T,6144]", output: "expert ids + weights", outputShape: "2 × [T,8]", formulaNote: "routed_scaling_factor=2.827 在权重上乘；checkpoint 里 gate.weight 是 BF16（字节对账排除 fp32），运行时以 fp32 加载。", runtime: "GateLinear + vllm_topk_sigmoid（fused）", source: "hy_v4/nvidia/moe.py · L122-L129", sourceUrl: `${ROUTER_URL}#L260-L290`, weights: [wt(layer, "mlp.gate.weight", "[256,6144]", "BF16", "1.57M", "GateLinear · fp32 运行时", "checkpoint 实存 BF16；路由器不切分"), wt(layer, "mlp.gate.e_score_correction_bias", "[256]", "BF16", "256", "expert_bias", "运行时 fp32；checkpoint dtype 待验证")] }),
    "m-experts": op({ id: "m-experts", kind: "activation", tone: "moe", kicker: "ROUTED LANE · 256 EXPERTS", title: "每 token 执行 8 个 routed experts", summary: "每 token 走 top-8 个专家，SwiGLU 带 clamp ±10（gate 只限上界）；全部 256 个专家打包成两张 fused 张量。以下是 expert 维度上的真实键。", input: "8 token groups", inputShape: "8 × [tokensₑ,6144]", output: "8 expert outputs", outputShape: "8 × [tokensₑ,6144]", formulaNote: "单专家 37.75M；256 个 routed experts 合计 9.66B/层，是 770B 总参的绝对大头（77 层 + MTP 共 753.77B）。", runtime: "FusedMoE · experts.gate_up_proj / down_proj", source: "hy_v4/nvidia/moe.py · L148-L170", sourceUrl: `${MOE_URL}#L70-L85`, weights: [wt(layer, "mlp.experts.gate_up_proj", "[256,4096,6144]", "BF16", "6.44B", "fused 全专家 gate+up", "纯 TP16：[256,256,6144]；EP16：[16,4096,6144]"), wt(layer, "mlp.experts.down_proj", "[256,6144,2048]", "BF16", "3.22B", "fused 全专家 down", "纯 TP16：[256,6144,128]；EP16：[16,6144,2048]")] }),
    "m-shared": op({ id: "m-shared", kind: "activation", tone: "moe", kicker: "SHARED LANE · ALWAYS ON", title: "1 个 shared expert（无限幅）", summary: "所有 token 都执行，不参与 Top-K 竞争；结构与 routed 专家相同但 SwiGLU 不 clamp，挂在 FusedMoE 内部一起算。", input: "all tokens", inputShape: "[T,6144]", output: "shared output", outputShape: "[T,6144]", formulaNote: "37.75M 参数/层，始终属于激活路径；HYV4FeedForward 与 L0 dense FFN 同一个类。", runtime: "HYV4FeedForward · shared_experts（reduce_results=False）", source: "hy_v4/nvidia/moe.py · L100-L112", sourceUrl: `${MOE_URL}#L23-L66`, weights: [wt(layer, "mlp.shared_experts.gate_proj.weight", "[2048,6144]", "BF16", "12.58M", "shared gate_up · gate"), wt(layer, "mlp.shared_experts.up_proj.weight", "[2048,6144]", "BF16", "12.58M", "shared gate_up · up"), wt(layer, "mlp.shared_experts.down_proj.weight", "[6144,2048]", "BF16", "12.58M")] }),
    "m-sum": op({ id: "m-sum", kind: "add", tone: "output", kicker: "MOE OUTPUT", title: "Weighted combine", summary: "8 个 routed 输出按 renorm 权重求和并乘 2.827，再与 shared expert 输出相加；实现按 token 分派/合并，不物化 9 份完整张量。", input: "8 routed + shared", inputShape: "9 × [T,6144]", output: "Ymoe", outputShape: "[T,6144]", runtime: "FusedMoE output combine", source: "hy_v4/nvidia/moe.py · L148-L170", sourceUrl: MOE_URL, weights: [] }),
  };
}

function denseFfnNodes(layer: number): Record<string, OpNode> {
  return {
    "d-gateup": op({ id: "d-gateup", kind: "linear", tone: "moe", kicker: "DENSE FFN · L0 · H_dense=18432", title: "Gate + Up Projection", summary: "仅 L0 使用 dense FFN：MergedColumnParallelLinear 一次 GEMM 产出 packed gate/up（各 18432 宽）。", input: "û′", inputShape: "[T,6144]", output: "packed gate_up (TP-local)", outputShape: "[T,36864/TP]", formulaNote: "TP 列切：每卡各 18432/TP。", runtime: "MergedColumnParallelLinear · gate_up_proj", source: "hy_v4/nvidia/moe.py · L45-L52", sourceUrl: `${MOE_URL}#L23-L66`, weights: [wt(layer, "mlp.gate_proj.weight", "[18432,6144]", "BF16", "113.25M", "gate_up_proj · gate"), wt(layer, "mlp.up_proj.weight", "[18432,6144]", "BF16", "113.25M", "gate_up_proj · up")] }),
    "d-split": op({ id: "d-split", kind: "split", tone: "moe", kicker: "DENSE FFN · TP-LOCAL SPLIT", title: "Split Gate / Up", summary: "把 packed gate_up 沿最后一维等分为 G 与 U；只切 view，不改数值。", input: "packed gate_up (TP-local)", inputShape: "[T,36864/TP]", output: "G · U", outputShape: "2 × [T,18432/TP]", runtime: "SiluAndMul · fused input slicing", source: "hy_v4/nvidia/moe.py · L62-L64", sourceUrl: `${MOE_URL}#L23-L66`, weights: [] }),
    "d-act": op({ id: "d-act", kind: "activation", tone: "moe", kicker: "DENSE FFN · 无限幅", title: "SiluAndMul", summary: "silu(G)⊙U——没有 clamp！dense FFN 与 shared expert 都不做限幅，这是与 routed 专家（clamp ±10）的关键差异。", input: "G, U", inputShape: "2 × [T,18432/TP]", output: "Z", outputShape: "[T,18432/TP]", formulaNote: "swiglu_limit=10.0 只作用于 routed experts。", runtime: "SiluAndMul.forward", source: "hy_v4/nvidia/moe.py · L56,L62-L64", sourceUrl: `${MOE_URL}#L23-L66`, weights: [] }),
    "d-down": op({ id: "d-down", kind: "linear", tone: "moe", kicker: "DENSE FFN · ROW PARALLEL", title: "Down Projection", summary: "把激活后的 18432/TP 投回 6144 并归并各 rank 部分和。", input: "activated (TP-local)", inputShape: "[T,18432/TP]", output: "Yffn", outputShape: "[T,6144]", runtime: "RowParallelLinear · down_proj", source: "hy_v4/nvidia/moe.py · L52-L59", sourceUrl: `${MOE_URL}#L23-L66`, weights: [wt(layer, "mlp.down_proj.weight", "[6144,18432]", "BF16", "113.25M")] }),
  };
}

function graphForLayerKind(kind: LayerKind): Record<string, OpNode> {
  const layer = DISPLAY_LAYER[kind];
  return {
    ...runtimeNodes(),
    ...chainNodes(layer),
    ...attentionNodes(layer, kind !== "shared"),
    ...(kind === "l0" ? denseFfnNodes(layer) : moeNodes(layer)),
  };
}

function bindingsFor(node: OpNode): IoBinding[] {
  const dataInputs = INPUT_OVERRIDES[node.id] ?? [{ kind: node.kind === "io" ? "external" : "upstream", label: node.input, shape: node.inputShape, from: node.kind === "io" ? "模型调用方 / runtime" : "图中紧邻的上游模块输出" }];
  const weightInputs = node.weights.map(weight => ({ kind: "weight" as const, label: weight.key, shape: `${weight.dtype} · ${weight.shape}`, from: weight.runtime ? `checkpoint → ${weight.runtime}` : `checkpoint · ${weight.shard}`, note: weight.params ? `${weight.params} parameters` : undefined }));
  return [...dataInputs, ...weightInputs];
}

const CONFIG_GROUPS: readonly ConfigGroup[] = [
  { title: "模型总体", rows: [["architectures", "HYV4ForCausalLM"], ["model_type", "hy_v4"], ["dtype / torch_dtype", "bfloat16"], ["hidden_size", "6144"], ["num_hidden_layers", "78（backbone，不含 MTP）"], ["vocab_size", "120832"], ["max_position_embeddings", "1048576（1M）"], ["rms_norm_eps", "1e−5"], ["hidden_act", "silu（dense/shared FFN 硬校验）"], ["intermediate_size", "18432（仅 L0 dense FFN）"], ["initializer_range", "0.006"], ["tie_word_embeddings", "false"], ["bos / eos / pad_token_id", "120000 / 120025 / 120002"], ["transformers_version", "5.16.2"], ["use_cache", "true"], ["attention_bias / attention_dropout", "false / 0.0"]] },
  { title: "MLA 注意力", rows: [["num_attention_heads", "64"], ["q_lora_rank", "2048"], ["kv_lora_rank", "512"], ["qk_nope_head_dim", "192"], ["qk_rope_head_dim", "64"], ["qk_head_dim", "256（__post_init__ 算出 = 192+64）"], ["v_head_dim", "256"], ["head_dim", "config 写 64；代码覆写为 qk_rope_head_dim 供 rotary 计算"], ["num_key_value_heads", "8（MLA 下无意义：实现里 kv heads=1，字段未被 hy_v4 代码读取）"], ["gated_mla", "true"], ["gating_type", "elementwise（每 head 每维一门；另一合法值 headwise）"], ["learnable_sink", "true"], ["learnable_sink_init", "0.0（推理代码不读，仅训练侧含义）"], ["rope_parameters.rope_theta", "10000000"], ["rope_parameters.rope_type", "default（interleaved：is_neox_style=False）"]] },
  { title: "DSA 稀疏索引", rows: [["use_dsa", "true（声明性元数据；判定 sparse 实际用 index_topk + layer_types）"], ["layer_types", "78 × deepseek_sparse_attention（无 full_attention 层）"], ["index_topk", "2048"], ["index_head_dim", "128（nope 64 + rope 64）"], ["index_n_heads", "32"], ["indexer_types", "21 × full（L0,1,5,9,…,77，1+4k 规律）+ 57 × shared"], ["indexer cache", "132 B/token/full 层（fp8 k + fp32 scale）"], ["量化硬编码", "block=128 + ue8m0 scale"]] },
  { title: "iHC 残差通道", rows: [["enable_ihc", "true"], ["hc_mult", "4（README 称 Residual Streams = 4）"], ["hc_magnitude", "2.0（H_post = 2.0·σ(·)+eps）"], ["hc_eps", "1e−6"], ["hc 权重", "hc_fn [8,24576] / hc_base [8] / hc_scale [2]，全 FP32"], ["hc_head", "hc_head_fn [4,24576]（顶层 4→1 合流，无 post 半段）"]] },
  { title: "MoE", rows: [["mlp_layer_types", "[dense] + 77 × [sparse]"], ["n_routed_experts", "256"], ["n_shared_experts", "1"], ["num_experts_per_tok", "8"], ["moe_intermediate_size", "2048"], ["norm_topk_prob", "true"], ["routed_scaling_factor", "2.827"], ["scoring_func", "sigmoid（代码硬编码）"], ["n_group / topk_group", "1 / 1（代码硬编码=1；config 字段不被读取，grouped topk 退化为全局）"], ["swiglu_limit", "10.0（仅 routed 专家；dense/shared 不限幅）"]] },
  { title: "MTP / 输出头", rows: [["num_nextn_predict_layers", "1（MTP 层，10B 总参 / 0.7B 激活）"], ["mtp_loss_factor", "0.1（训练期 MTP loss 权重，推理不用）"], ["官方推荐 spec tokens", "3（draft 3 步、target 一次验证 4 个）"], ["MTP 结构", "enorm + hnorm + eh_proj(12288→6144) + 1 层 decoder（无 iHC，DSA+MoE）+ final_layernorm；共享 embed/lm_head"], ["enable_lm_head_fp32", "true（head_dtype=float32 → LogitsProcessor torch.mm(out_dtype=fp32) 累积；权重本体仍 bf16）"], ["bitwise_backward_align", "false（vLLM 全仓无使用，训练框架遗留）"]] },
];

const CONFIG_SYMBOLS: Record<string, string> = {
  "模型总体:hidden_size": "H", "模型总体:num_hidden_layers": "L", "模型总体:vocab_size": "V", "模型总体:max_position_embeddings": "S_max", "模型总体:rms_norm_eps": "ε", "模型总体:intermediate_size": "H_dense",
  "MLA 注意力:num_attention_heads": "Nₕ", "MLA 注意力:q_lora_rank": "R_q", "MLA 注意力:kv_lora_rank": "L_k", "MLA 注意力:qk_nope_head_dim": "D_n", "MLA 注意力:qk_rope_head_dim": "D_r", "MLA 注意力:qk_head_dim": "D_qk", "MLA 注意力:v_head_dim": "D_v", "MLA 注意力:rope_parameters.rope_theta": "θ_base", "MLA 注意力:gating_type": "elementwise",
  "DSA 稀疏索引:index_topk": "K_idx", "DSA 稀疏索引:index_head_dim": "D_idx", "DSA 稀疏索引:index_n_heads": "N_idx",
  "iHC 残差通道:hc_mult": "C", "iHC 残差通道:hc_magnitude": "M", "iHC 残差通道:hc_eps": "ε_hc",
  "MoE:n_routed_experts": "E", "MoE:num_experts_per_tok": "K", "MoE:moe_intermediate_size": "H_e", "MoE:routed_scaling_factor": "s_route", "MoE:swiglu_limit": "c",
  "MTP / 输出头:num_nextn_predict_layers": "L_mtp",
};

function stageOverviewFor(kind: LayerKind, stage: Exclude<ExpandedStage, null>): StageOverview {
  const full = kind !== "shared";
  if (stage === "attention") return {
    kicker: full ? "GATED DSA · FULL INDEXER 层" : "GATED DSA · SHARED INDEXER 层",
    title: full ? "MLA + lightning indexer + learnable sink" : "MLA + 共享 Top-2048 + learnable sink",
    summary: full ? "低秩 MLA 注意力；FP8 lightning indexer 先选 Top-2048，主 attention 只在选中 token 上计算，sink 折进 softmax 分母，输出再过 σ gate。" : "与 full 层同一套 MLA 主干，但 indexer 已由最近一个 full 层算完，本层只读共享 buffer。",
    flow: full
      ? "û → fused 下投影 → q_c/kv latent → 双 RMSNorm → 上投影 + interleaved RoPE ‖ indexer: wq_b/wk → k_norm → FP8 → logits → Top-2048 → sparse MLA(+sink) → ×σ(gate) → o_proj"
      : "û → fused 下投影 → q_c/kv latent → 双 RMSNorm → 上投影 + interleaved RoPE ‖ 读 topk_indices_buffer → sparse MLA(+sink) → ×σ(gate) → o_proj",
    formula: full
      ? "q = W_qb·RMSNorm(q_c)，(kv_c,k_pe) = Split(W_kva·û)\no = Σ_{s∈ℐ_t} e^{qk/√256}v_s / (Σ e^{qk/√256} + e^{κ_h})\nℐ_t = TopK_{2048}[(1/√32)(1/√128)Σ_h w_h⟨q_h,k⟩]\no' = o ⊙ σ(û·W_g)"
      : "q = W_qb·RMSNorm(q_c)，(kv_c,k_pe) = Split(W_kva·û)\no = Σ_{s∈ℐ_t} e^{qk/√256}v_s / (Σ e^{qk/√256} + e^{κ_h})\nℐ_t ← topk_indices_buffer[t]（full 层写入）\no' = o ⊙ σ(û·W_g)",
    notes: full
      ? ["indexer 的 query 输入是 q_a_layernorm 后的 q_c（复用 MLA 压缩），key 输入是 û。", "sink 强制 prefill 也走稀疏 MQA 路径（dense 后端不支持 sink）。"]
      : ["shared 层不构建 indexer 模块，checkpoint 的 indexer 权重加载时被丢弃。", "full 层位置 = {0,1,5,9,…,77}；每个 full 层后面跟 ≤3 个 shared 层。"],
    parameters: [["Nₕ", "64", "num_attention_heads"], ["D_qk", "256", "nope 192 + rope 64"], ["L_k", "512", "kv_lora_rank（cache 576/token）"], ["N_idx", "32", "index_n_heads"], ["K_idx", "2048", "index_topk"]],
  };
  return kind === "l0" ? {
    kicker: "DENSE FFN · L0", title: "Dense SwiGLU（无限幅）", summary: "第 0 层用 dense FFN 而非 MoE；SwiGLU 不带 clamp。",
    flow: "û′ → Gate+Up Projection(2×18432) → Split → silu⊙up → Down Projection → Yffn",
    formula: "G = û′W_gateᵀ，U = û′W_upᵀ\nZ = silu(G) ⊙ U\nYffn = Z·W_downᵀ",
    notes: ["L0 同时是 full indexer 层（layer_types 全部 78 层都是 DSA）。", "与 shared expert 共用 HYV4FeedForward 类，均无限幅。"],
    parameters: [["H", "6144", "hidden_size"], ["H_dense", "18432", "intermediate_size"], ["clamp", "无", "swiglu_limit 仅 routed"]],
  } : {
    kicker: "SPARSE FFN · L1–L77", title: "Top-8 MoE + Shared Expert", summary: "每 token 进入 8 个路由专家（SwiGLU clamp ±10、renorm、×2.827），同时经过 1 个共享专家（无限幅）。",
    flow: "û′ → Router Top-8 ↘ Routed Experts · Shared Expert ↗ Weighted Sum → Ymoe",
    formula: "𝓔 = TopK₈(σ(û′W_routeᵀ)+b)\nŵₑ = σ(rₑ)/Σ_{j∈𝓔}σ(rⱼ)\nYmoe = 2.827·Σ_{e∈𝓔}ŵₑEₑ(û′) + E_shared(û′)",
    notes: ["选专家用 σ+bias，混合权重用原始 σ 分数再 renorm（选与算分离）。", "shared expert 挂在 FusedMoE 内部一起算，加法由其统一做。"],
    parameters: [["E", "256", "n_routed_experts"], ["K", "8", "num_experts_per_tok"], ["H_e", "2048", "moe_intermediate_size"], ["c", "10.0", "swiglu_limit（仅 routed）"], ["s_route", "2.827", "routed_scaling_factor"]],
  };
}

function Hy4Overview() {
  return <div className="model-overview">
    <div className="model-step">Token IDs</div><Arrow/>
    <div className="model-step">Embedding <code>[T,4,6144]</code></div><Arrow/>
    <div className="overview-stack"><b>Decoder ×78 + MTP ×1</b><span><i className="dense"/>L0 ×1 · dense FFN + full indexer</span><span><i className="sparse"/>full ×21（L1,5,…,77）· shared ×57 · MoE</span></div><Arrow/>
    <div className="model-step">iHC Head 4→1 <code>[T,6144]</code></div><Arrow/>
    <div className="model-step">Final RMSNorm</div><Arrow/>
    <div className="model-step">LM Head · fp32 <code>[T,120832]</code></div>
  </div>;
}

function Hy4Navigator({ value, onChange }: NavigatorProps) {
  const kind: LayerKind = value === "l0" ? "l0" : value === "shared" ? "shared" : "full";
  return <div className="layer-nav layer-type-nav hy4-nav"><div className="layer-nav-head"><b>{kind === "l0" ? "Full Indexer + Dense FFN" : kind === "full" ? "Full Indexer + MoE" : "Shared Indexer + MoE"}</b></div><div className="layer-type-options">
    <button className={kind === "l0" ? "active dense" : "dense"} onClick={() => onChange("l0")}><span>L0 ×1</span><b>full indexer + Dense SwiGLU FFN</b><small>唯一 dense FFN 层；iHC + MLA 同构</small></button>
    <button className={kind === "full" ? "active sparse" : "sparse"} onClick={() => onChange("full")}><span>L1–L77 ×21</span><b>full indexer + 256-expert MoE</b><small>indexer_types=full：L1,5,9,…,77（1+4k）</small></button>
    <button className={kind === "shared" ? "active sparse" : "sparse"} onClick={() => onChange("shared")}><span>L2–L76 ×57</span><b>shared indexer + 256-expert MoE</b><small>读最近 full 层写入的 Top-2048 buffer</small></button>
  </div></div>;
}

/* eslint-disable react-hooks/static-components -- local N/IW/A aliases keep the dependency diagrams legible */
function Hy4AttentionZoom({ kind, g, active, onHover, onLeave, onSelect, onClose }: { kind: LayerKind; g: Record<string, OpNode>; active: string; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; onClose: () => void }) {
  const p = { active: false, onHover, onLeave, onSelect };
  const N = ({ id, graphId }: { id: string; graphId?: string }) => <Op node={g[id]} {...p} active={active === g[id].id} graphId={graphId} />;
  const full = kind !== "shared";
  const edges: GraphEdge[] = [
    {from:"attn-uh",to:"attn-fused", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wfused",to:"attn-fused", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-fused",to:"attn-lora" },
    {from:"attn-lora",to:"attn-split" },
    {from:"attn-split",to:"attn-qc" },
    {from:"attn-split",to:"attn-kvl" },
    {from:"attn-qc",to:"attn-qanorm", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wqanorm",to:"attn-qanorm", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-qanorm",to:"attn-qcn" },
    {from:"attn-qcn",to:"attn-qb", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wqb",to:"attn-qb", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-qb",to:"attn-q" },
    {from:"attn-q",to:"attn-rope" },
    {from:"attn-posq",to:"attn-rope", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-kpe",to:"attn-rope", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-rope",to:"attn-qr" },
    {from:"attn-rope",to:"attn-kper" },
    {from:"attn-kvl",to:"attn-kvsplit" },
    {from:"attn-kvsplit",to:"attn-kvc" },
    {from:"attn-kvsplit",to:"attn-kpe" },
    {from:"attn-kvc",to:"attn-kvnorm", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wkvnorm",to:"attn-kvnorm", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-kvnorm",to:"attn-kvcn" },
    {from:"attn-kvcn",to:"attn-kvb", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wkvb",to:"attn-kvb", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-kvb",to:"attn-kvbout" },
    {from:"attn-kvbout",to:"attn-cache" },
    {from:"attn-kper",to:"attn-cache" },
    {from:"attn-cachemeta",to:"attn-cache", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-cache",to:"attn-latent" },
    ...(full ? [
      {from:"attn-qcn",to:"attn-idxq" },
      {from:"attn-idxwq",to:"attn-idxq", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
      {from:"attn-idxq",to:"attn-idxqt" },
      {from:"attn-idxh",to:"attn-idxk" },
      {from:"attn-idxwk",to:"attn-idxk", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
      {from:"attn-idxk",to:"attn-idxkt" },
      {from:"attn-idxkt",to:"attn-idxknorm" },
      {from:"attn-idxwkn",to:"attn-idxknorm", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
      {from:"attn-idxqt",to:"attn-idxrope" },
      {from:"attn-idxknorm",to:"attn-idxrope" },
      {from:"attn-posi",to:"attn-idxrope", fromPort: "left" as EdgePort, toPort: "right" as EdgePort },
      {from:"attn-idxrope",to:"attn-idxquant" },
      {from:"attn-idxquant",to:"attn-idxlogits" },
      {from:"attn-idxlogits",to:"attn-idxtopk" },
      {from:"attn-idxtopk",to:"attn-idxids" },
      {from:"attn-idxids",to:"attn-core" },
    ] : [
      {from:"attn-sharedbuf",to:"attn-core" },
    ]),
    {from:"attn-qr",to:"attn-core" },
    {from:"attn-latent",to:"attn-core" },
    {from:"attn-wsink",to:"attn-core", toPort: "top" as EdgePort },
    {from:"attn-core",to:"attn-heads", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-heads",to:"attn-gate", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-hg",to:"attn-gate", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wgate",to:"attn-gate", toPort: "top" as EdgePort },
    {from:"attn-gate",to:"attn-gated", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-gated",to:"attn-oproj", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
    {from:"attn-wop",to:"attn-oproj", toPort: "top" as EdgePort },
    {from:"attn-oproj",to:"attn-y", fromPort: "right" as EdgePort, toPort: "left" as EdgePort },
  ];
  return <section className="stage-zoom lesson-zoom"><header><span>{full ? "GATED DSA · MLA + LIGHTNING INDEXER + SINK" : "GATED DSA · MLA + 共享 TOP-2048 + SINK"}</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="hy4-attn-graph" edges={edges}>
    <div className="compact-chain"><Tensor name="û" shape="[T,H]" graphId="attn-uh"/><Tensor name="q_a_proj ‖ kv_a_proj_with_mqa" shape="[R_q,H] ‖ [L_k+D_r,H]" role="weight" graphId="attn-wfused"/><N id="a-fused" graphId="attn-fused"/><Tensor name="qkv_lora" shape="[T,R_q+L_k+D_r]" graphId="attn-lora"/><N id="a-split" graphId="attn-split"/></div>
    <div className="qkv-lanes">
      <section><header>Q LATENT PATH · q_c → 64×256</header><div className="hy4-lane-row"><Tensor name="q_c" shape="[T,R_q]" graphId="attn-qc"/><Tensor name="q_a_layernorm.weight" shape="[R_q]" role="weight" graphId="attn-wqanorm"/><N id="a-qnorm" graphId="attn-qanorm"/><Tensor name="q_cⁿ（indexer 也读它）" shape="[T,R_q]" graphId="attn-qcn"/><Tensor name="q_b_proj.weight" shape="[N_h·D_qk,R_q]" role="weight" graphId="attn-wqb"/><N id="a-qup" graphId="attn-qb"/><Tensor name="q · nope‖rope" shape="[T,N_h,256]" graphId="attn-q"/><Tensor name="q rope 切片 · positions" shape="[T,N_h,64]+[Nq]" graphId="attn-posq"/><Tensor name="k_pe" shape="[T,1,64]" graphId="attn-kpe"/><N id="a-rope" graphId="attn-rope"/><Tensor name="qᵣ" shape="[T,N_h,256]" graphId="attn-qr"/></div></section>
      <section><header>KV LATENT PATH · 576/token</header><div className="hy4-lane-row"><Tensor name="kv_lora" shape="[T,L_k+D_r]" graphId="attn-kvl"/><N id="a-kvsplit" graphId="attn-kvsplit"/><Tensor name="kv_c" shape="[T,L_k]" graphId="attn-kvc"/><Tensor name="kv_a_layernorm.weight" shape="[L_k]" role="weight" graphId="attn-wkvnorm"/><N id="a-kvnorm" graphId="attn-kvnorm"/><Tensor name="c_n" shape="[T,L_k]" graphId="attn-kvcn"/><Tensor name="kv_b_proj.weight" shape="[N_h·(D_n+D_v),L_k]" role="weight" graphId="attn-wkvb"/><N id="a-kvb" graphId="attn-kvb"/><Tensor name="k_nope‖v · 每 head 448" shape="[T,N_h,448]" graphId="attn-kvbout"/><Tensor name="k_peᵣ" shape="[T,1,64]" graphId="attn-kper"/><Tensor name="slot_mapping · block_table" shape="runtime" role="side" graphId="attn-cachemeta"/><N id="a-cache" graphId="attn-cache"/><Tensor name="latent KV cache" shape="[pages,576]" graphId="attn-latent"/></div></section>
    </div>
    {full ? <div className="hy4-index-chain"><header>LIGHTNING INDEXER · FP8 · 32 HEADS · Top-2048</header><div className="hy4-index-row"><Tensor name="wq_b.weight" shape="[N_idx·D_idx,R_q]" role="weight" graphId="attn-idxwq"/><N id="i-qup" graphId="attn-idxq"/><Tensor name="q_idx · nope‖pe" shape="[T,N_idx,128]" graphId="attn-idxqt"/><Tensor name="û（indexer key/权重输入）" shape="[T,H]" role="side" graphId="attn-idxh"/><Tensor name="wk ‖ weights_proj" shape="[D_idx,H] ‖ [N_idx,H]" role="weight" graphId="attn-idxwk"/><N id="i-kw" graphId="attn-idxk"/><Tensor name="k · w" shape="[T,128]+[T,32]" graphId="attn-idxkt"/><N id="i-knorm" graphId="attn-idxknorm"/><Tensor name="k_norm.weight/bias" shape="[128]" role="weight" graphId="attn-idxwkn"/><N id="i-rope" graphId="attn-idxrope"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posi"/><N id="i-quant" graphId="attn-idxquant"/><N id="i-logits" graphId="attn-idxlogits"/><N id="i-topk" graphId="attn-idxtopk"/><Tensor name="ℐ_t → topk_indices_buffer" shape="[T,K_idx]" graphId="attn-idxids"/></div></div>
      : <div className="hy4-index-chain hy4-shared"><header>SHARED INDEXER · 读共享 buffer（skip_topk）</header><div className="hy4-index-row"><Tensor name="topk_indices_buffer · 最近 full 层写入" shape="[T,K_idx] int32" role="side" graphId="attn-sharedbuf"/></div></div>}
    <div className="hy4-out-pipeline"><div className="hy4-out-weights"><Tensor name="learnable_sink_param" shape="[N_h] FP32" role="weight" graphId="attn-wsink"/><Tensor name="linear_gate.weight" shape="[N_h·D_v,H]" role="weight" graphId="attn-wgate"/><Tensor name="o_proj.weight" shape="[H,N_h·D_v]" role="weight" graphId="attn-wop"/></div><div className="hy4-out-row"><N id="a-core" graphId="attn-core"/><Tensor name="attn heads" shape="[T,N_h·D_v]" graphId="attn-heads"/><Tensor name="û（gate 输入）" shape="[T,H]" role="side" graphId="attn-hg"/><N id="a-gate" graphId="attn-gate"/><Tensor name="gated heads" shape="[T,N_h·D_v]" graphId="attn-gated"/><N id="a-oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[T,H]" graphId="attn-y"/></div></div>
  </GraphSurface></section>;
}
/* eslint-enable react-hooks/static-components */

function Hy4MoeZoom({ g, active, onHover, onLeave, onSelect, onClose }: { g: Record<string, OpNode>; active: string; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; onClose: () => void }) {
  const p = { active: false, onHover, onLeave, onSelect };
  const N = ({ id, graphId }: { id: string; graphId?: string }) => <Op node={g[id]} {...p} active={active === g[id].id} graphId={graphId} />;
  const edges: GraphEdge[] = [
    {from:"moe-u",to:"moe-router" }, {from:"moe-wrouter",to:"moe-router", fromPort: "right", toPort: "left" }, {from:"moe-router",to:"moe-ids" }, {from:"moe-router",to:"moe-rweights" }, {from:"moe-u",to:"moe-experts", toPort: "top-right" }, {from:"moe-ids",to:"moe-experts", toPort: "top-left" }, {from:"moe-rweights",to:"moe-experts" }, {from:"moe-wexperts",to:"moe-experts", fromPort: "right", toPort: "left" }, {from:"moe-experts",to:"moe-routed" }, {from:"moe-u",to:"moe-shared", fromPort: "bottom", toPort: "top" }, {from:"moe-wshared",to:"moe-shared", fromPort: "left", toPort: "right" }, {from:"moe-shared",to:"moe-shared-out" }, {from:"moe-routed",to:"moe-sum" }, {from:"moe-shared-out",to:"moe-sum" }, {from:"moe-sum",to:"moe-y" },
  ];
  return <section className="stage-zoom lesson-zoom"><header><span>TOP-8 MOE + SHARED EXPERT · L1–L77</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="moe-node-graph" edges={edges}>
    <Tensor name="û′" shape="[T,H]" graphId="moe-u"/><Tensor name="gate.weight (BF16→FP32) · correction bias" shape="[E,H] · [E]" role="weight" graphId="moe-wrouter"/><N id="m-router" graphId="moe-router"/><Tensor name="expert ids" shape="[T,K]" graphId="moe-ids"/><Tensor name="router weights" shape="[T,K]" graphId="moe-rweights"/><Tensor name="experts.gate_up_proj / down_proj（256 专家打包）" shape="[E,2H_e,H] · [E,H,H_e]" role="weight" graphId="moe-wexperts"/><N id="m-experts" graphId="moe-experts"/><Tensor name="weighted routed output" shape="[T,H]" graphId="moe-routed"/><N id="m-shared" graphId="moe-shared"/><Tensor name="shared expert weights ×3（无限幅）" shape="gate / up / down" role="weight" graphId="moe-wshared"/><Tensor name="shared output" shape="[T,H]" graphId="moe-shared-out"/><N id="m-sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[T,H]" graphId="moe-y"/>
  </GraphSurface></section>;
}

function Hy4DenseZoom({ g, active, onHover, onLeave, onSelect, onClose }: { g: Record<string, OpNode>; active: string; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; onClose: () => void }) {
  const p = { active: false, onHover, onLeave, onSelect };
  const N = ({ id, graphId }: { id: string; graphId?: string }) => <Op node={g[id]} {...p} active={active === g[id].id} graphId={graphId} />;
  const edges: GraphEdge[] = [
    {from:"mlp-uhat",to:"mlp-gateup", toPort: "top" }, {from:"mlp-wgate",to:"mlp-gateup", toPort: "top-left" }, {from:"mlp-wup",to:"mlp-gateup", toPort: "top-right" }, {from:"mlp-gateup",to:"mlp-packed" }, {from:"mlp-packed",to:"mlp-split" }, {from:"mlp-split",to:"mlp-gate" }, {from:"mlp-split",to:"mlp-up" }, {from:"mlp-gate",to:"mlp-gate-act" }, {from:"mlp-up",to:"mlp-up-act" }, {from:"mlp-gate-act",to:"mlp-mul" }, {from:"mlp-up-act",to:"mlp-mul" }, {from:"mlp-mul",to:"mlp-activated" }, {from:"mlp-activated",to:"mlp-down" }, {from:"mlp-wdown",to:"mlp-down", fromPort: "left", toPort: "right" }, {from:"mlp-down",to:"mlp-y" },
  ];
  return <section className="stage-zoom lesson-zoom"><header><span>DENSE SWIGLU FFN · L0（无限幅）</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="mlp-node-graph" edges={edges}>
    <Tensor name="û′" shape="[T,H]" graphId="mlp-uhat"/><Tensor name="mlp.gate_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wgate"/><N id="d-gateup" graphId="mlp-gateup"/><Tensor name="mlp.up_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wup"/><Tensor name="packed gate_up⁽ʳ⁾" shape="[T,2H_dense/TP]" graphId="mlp-packed"/><N id="d-split" graphId="mlp-split"/><Tensor name="G · gate" shape="[T,H_dense/TP]" graphId="mlp-gate"/><Tensor name="U · up" shape="[T,H_dense/TP]" graphId="mlp-up"/><button type="button" className="mini-math activation-step" data-graph-id="mlp-gate-act" aria-pressed={active === g["d-act"].id} onPointerDown={() => onSelect(g["d-act"])} onClick={event => { if (event.detail === 0) onSelect(g["d-act"]) }} onMouseEnter={() => onHover(g["d-act"])} onMouseLeave={onLeave}>silu(G⁽ʳ⁾)（无限幅）</button><button type="button" className="mini-math activation-step" data-graph-id="mlp-up-act" aria-pressed={active === g["d-act"].id} onPointerDown={() => onSelect(g["d-act"])} onClick={event => { if (event.detail === 0) onSelect(g["d-act"]) }} onMouseEnter={() => onHover(g["d-act"])} onMouseLeave={onLeave}>U⁽ʳ⁾（不 clamp）</button><button className="multiply-circle" data-graph-id="mlp-mul" aria-pressed={active === g["d-act"].id} onPointerDown={() => onSelect(g["d-act"])} onClick={event => { if (event.detail === 0) onSelect(g["d-act"]) }} onMouseEnter={() => onHover(g["d-act"])} onMouseLeave={onLeave}>×</button><Tensor name="Z⁽ʳ⁾" shape="[T,H_dense/TP]" graphId="mlp-activated"/><N id="d-down" graphId="mlp-down"/><Tensor name="mlp.down_proj.weight⁽ʳ⁾" shape="[H,H_dense/TP]" role="weight" graphId="mlp-wdown"/><Tensor name="Yffn" shape="[T,H]" graphId="mlp-y"/>
  </GraphSurface></section>;
}

function Hy4DecoderDiagram({ layerType, graph, activeId, expanded, onExpand, onHover, onLeave, onSelect }: WorkbenchProps) {
  const kind: LayerKind = layerType === "l0" ? "l0" : layerType === "shared" ? "shared" : "full";
  const g = graph;
  const active = activeId;
  const p = { active: false, onHover, onLeave, onSelect };
  const IW = ({ id, inputName, inputShape, inputGraphId, graphId, weightGraphId }: { id: string; inputName: string; inputShape: string; inputGraphId: string; graphId: string; weightGraphId: string }) => <InputWeightedOp node={g[id]} {...p} active={active === g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId} />;
  const A = ({ id, graphId }: { id: string; graphId: string }) => <AddCircle node={g[id]} {...p} active={active === g[id].id} graphId={graphId} />;
  const edges: GraphEdge[] = [
    {from:"main-x",to:"main-hcpre1", toPort: "top-left" }, {from:"main-whcpre1",to:"main-hcpre1", toPort: "top-right" }, {from:"main-hcpre1",to:"main-y" }, {from:"main-y",to:"main-norm", toPort: "top-left" }, {from:"main-win",to:"main-norm", toPort: "top-right" }, {from:"main-norm",to:"main-attn" }, {from:"main-attn",to:"main-hcpost1" }, {from:"main-x",to:"main-hcpost1", fromPort: "left", toPort: "left", route: "side-left" }, {from:"main-hcpost1",to:"main-u" }, {from:"main-u",to:"main-hcpre2", toPort: "top-left" }, {from:"main-whcpre2",to:"main-hcpre2", toPort: "top-right" }, {from:"main-hcpre2",to:"main-y2" }, {from:"main-y2",to:"main-post", toPort: "top-left" }, {from:"main-wpost",to:"main-post", toPort: "top-right" }, {from:"main-post",to:"main-ffn" }, {from:"main-ffn",to:"main-hcpost2" }, {from:"main-u",to:"main-hcpost2", fromPort: "left", toPort: "left", route: "side-left" }, {from:"main-hcpost2",to:"main-out" },
  ];
  const attentionLabel = kind === "shared" ? "Gated DSA · 共享 Top-2048 + sink" : "Gated DSA · lightning indexer + sink";
  const ffnLabel = kind === "l0" ? "Dense SwiGLU（18432 · 无限幅）" : "256-expert MoE + shared expert";
  return <div className={`decoder-workbench ${expanded ? "has-zoom" : ""}`}>{!expanded && <div className="hy4-collapsed"><GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <div className="hy4-pair-row"><IW id="hc-attnpre" inputName="Xₗ · iHC 通道状态" inputShape="[T,C,H]" inputGraphId="main-x" graphId="main-hcpre1" weightGraphId="main-whcpre1"/><IW id="norm-in" inputName="y" inputShape="[T,H]" inputGraphId="main-y" graphId="main-norm" weightGraphId="main-win"/></div><button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={() => onExpand(expanded === "attention" ? null : "attention")}><small>点击展开</small><b>{attentionLabel}</b></button><A id="hc-attnpost" graphId="main-hcpost1"/><Tensor name="U · 更新后通道" shape="[T,C,H]" graphId="main-u"/><div className="hy4-pair-row"><IW id="hc-mlppre" inputName="U · iHC 通道状态" inputShape="[T,C,H]" inputGraphId="main-u" graphId="main-hcpre2" weightGraphId="main-whcpre2"/><IW id="norm-post" inputName="y′" inputShape="[T,H]" inputGraphId="main-y2" graphId="main-post" weightGraphId="main-wpost"/></div><button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={() => onExpand(expanded === "ffn" ? null : "ffn")}><small>点击展开</small><b>{ffnLabel}</b></button><A id="hc-mlppost" graphId="main-hcpost2"/><Tensor name="Xₗ₊₁ · iHC 通道" shape="[T,C,H]" graphId="main-out"/>
  </GraphSurface><RuntimeIORail N={({ id }) => <Op node={g[id]} {...p} active={active === g[id].id}/>}/></div>}{expanded === "attention" && <Hy4AttentionZoom kind={kind} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={() => onExpand(null)}/>}{expanded === "ffn" && (kind === "l0"
    ? <Hy4DenseZoom g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={() => onExpand(null)}/>
    : <Hy4MoeZoom g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={() => onExpand(null)}/>)}</div>;
}

export const hy4Module: ModelModule = {
  id: "hunyuan-hy4",
  name: "Hunyuan Hy4",
  facts: { total: "770B", active: "49B", context: "1M", checkpoint: "1.56 TB" },
  links: { codeUrl: CODE_URL, codeLabel: `vLLM @ ${VLLM_COMMIT.slice(0, 7)}`, weightsUrl: WEIGHTS_URL, weightsLabel: "ModelScope · 131 shards" },
  vllmCommit: VLLM_COMMIT,
  defaultLayerType: "full",
  configGroups: CONFIG_GROUPS,
  configSymbols: CONFIG_SYMBOLS,
  graphFor: (layerType: string) => graphForLayerKind(layerType === "l0" ? "l0" : layerType === "shared" ? "shared" : "full"),
  inputBindingsFor: bindingsFor,
  nextFor: (nodeId: string) => NEXT_BY_ID[nodeId],
  symbolicShape,
  stageOverviewFor: (layerType: string, stage: Exclude<ExpandedStage, null>) => stageOverviewFor(layerType === "l0" ? "l0" : layerType === "shared" ? "shared" : "full", stage),
  formulaTermsFor: formulaTerms,
  formulaNoteFor: (node: OpNode) => node.formulaNote ?? FORMULA_NOTE_DEFAULT[node.kind] ?? "",
  Overview: Hy4Overview,
  Navigator: Hy4Navigator,
  canvasHeading: (layerType: string) => ({
    kicker: "DECODER LAYER · 按结构类型展示",
    title: layerType === "l0"
      ? "iHC + Gated DSA（full indexer）+ Dense SwiGLU · L0 独有"
      : layerType === "shared"
        ? "iHC + Gated DSA（读共享 Top-2048）+ 256-expert MoE · 其余 57 层同构"
        : "iHC + Gated DSA（lightning indexer）+ 256-expert MoE · L1,5,…,77 同构（21 层）",
  }),
  Workbench: Hy4DecoderDiagram,
};
