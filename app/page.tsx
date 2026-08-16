"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import katex from "katex";
import { denseNodes, layerShard, sparseNodes, visionNodes, type Node, type Weight } from "./model-data";

type Tab = "io" | "formula" | "code" | "weights";
type OpKind = "io" | "norm" | "linear" | "split" | "rope" | "matmul" | "scale" | "mask" | "softmax" | "activation" | "route" | "cache" | "add";
type CodeSection = { stage: string; title: string; location: string; code: string; url?: string };
type CodeSymbol = { symbol: string; resolvesTo: string; meaning: string };
type CodeDetail = { sections: CodeSection[]; symbols: CodeSymbol[] };
type OpNode = Node & { kind: OpKind; latex?: string; codeSections?: CodeSection[]; codeSymbols?: CodeSymbol[] };

const VLLM_COMMIT = "edd4c8176cfd98ece8a29beda574378c42971967";
const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3";
const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
const ACTIVATION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/activation.py`;

const MODEL_REGISTRY = [
  { id: "minimax-m3", name: "MiniMax-M3", enabled: true },
  { id: "kimi-k3", name: "Kimi K3 · 待添加", enabled: false },
  { id: "deepseek-v4", name: "DeepSeek V4 · 待添加", enabled: false },
  { id: "step-3.7", name: "Step 3.7 · 待添加", enabled: false },
];

const LATEX_BY_ID: Record<string,string> = {
  "d-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "s-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "d-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "s-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "d-slots":String.raw`\begin{aligned}\ell&=\left\lfloor p/\mathrm{block\_size}\right\rfloor,\quad o=p\bmod \mathrm{block\_size}\\b_{\mathrm{phys}}&=\mathrm{block\_table}[r,\ell]\\\mathrm{slot}(r,p)&=b_{\mathrm{phys}}\cdot\mathrm{block\_size}+o\end{aligned}`,
  "s-slots":String.raw`\begin{aligned}\ell&=\left\lfloor p/\mathrm{block\_size}\right\rfloor,\quad o=p\bmod \mathrm{block\_size}\\b_{\mathrm{phys}}&=\mathrm{block\_table}[r,\ell]\\\mathrm{slot}(r,p)&=b_{\mathrm{phys}}\cdot\mathrm{block\_size}+o\end{aligned}`,
  "d-norm":String.raw`\begin{aligned}\operatorname{rms}(x)&=\sqrt{\frac1{6144}\sum_{j=1}^{6144}x_j^2+\varepsilon}\\\hat x_j&=\frac{x_j}{\operatorname{rms}(x)}(1+\gamma_j),\qquad \varepsilon=10^{-6}\end{aligned}`,
  "d-qkv":String.raw`\begin{aligned}Z&=\hat X\,[W_Q^\top\mid W_K^\top\mid W_V^\top]\\Z&\in\mathbb R^{B\times S\times(8192+512+512)}\end{aligned}`,
  "d-split":String.raw`\begin{aligned}Q&=Z_{:,:,0:8192}\in\mathbb R^{B\times64\times S\times128}\\K&=Z_{:,:,8192:8704}\in\mathbb R^{B\times4\times S\times128}\\V&=Z_{:,:,8704:9216}\in\mathbb R^{B\times4\times S\times128}\end{aligned}`,
  "d-qnorm":String.raw`\tilde Q_{b,h,s,:}=\frac{Q_{b,h,s,:}}{\sqrt{\frac1{128}\lVert Q_{b,h,s,:}\rVert_2^2+\varepsilon}}\odot(1+\gamma_Q)`,
  "d-knorm":String.raw`\tilde K_{b,g,s,:}=\frac{K_{b,g,s,:}}{\sqrt{\frac1{128}\lVert K_{b,g,s,:}\rVert_2^2+\varepsilon}}\odot(1+\gamma_K)`,
  "d-ropeq":String.raw`\begin{aligned}\theta_{p,j}&=p\,\theta_{\mathrm{base}}^{-2j/d_r},\quad d_r=64\\\binom{Q^r_{2j}}{Q^r_{2j+1}}&=\begin{bmatrix}\cos\theta_{p,j}&-\sin\theta_{p,j}\\\sin\theta_{p,j}&\cos\theta_{p,j}\end{bmatrix}\binom{\tilde Q_{2j}}{\tilde Q_{2j+1}}\\Q^r_{d_r:128}&=\tilde Q_{d_r:128}\end{aligned}`,
  "d-ropek":String.raw`\begin{aligned}\theta_{p,j}&=p\,\theta_{\mathrm{base}}^{-2j/d_r},\quad d_r=64\\\binom{K^r_{2j}}{K^r_{2j+1}}&=\begin{bmatrix}\cos\theta_{p,j}&-\sin\theta_{p,j}\\\sin\theta_{p,j}&\cos\theta_{p,j}\end{bmatrix}\binom{\tilde K_{2j}}{\tilde K_{2j+1}}\\K^r_{d_r:128}&=\tilde K_{d_r:128}\end{aligned}`,
  "d-cache":String.raw`\begin{aligned}\mathcal K[\mathrm{slot}(r,p)]&\leftarrow K^r_{r,p}\\\mathcal V[\mathrm{slot}(r,p)]&\leftarrow V_{r,p}\\K_{\le p},V_{\le p}&\leftarrow\operatorname{gather}(\mathcal K,\mathcal V,\mathrm{block\_table}_r)\end{aligned}`,
  "d-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{128}Q^r_{b,h,i,m}\,K^r_{b,\lfloor h/16\rfloor,j,m}`,
  "d-scale":String.raw`\bar A_{b,h,i,j}=\frac{A_{b,h,i,j}}{\sqrt{128}}`,
  "d-mask":String.raw`\tilde A_{b,h,i,j}=\bar A_{b,h,i,j}+M_{b,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\le c_b+i\\-\infty,&j>c_b+i\end{cases}`,
  "d-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-m_{b,h,i})}{\sum_{t=0}^{T-1}\exp(\tilde A_{b,h,i,t}-m_{b,h,i})},\quad m_{b,h,i}=\max_t\tilde A_{b,h,i,t}`,
  "d-pv":String.raw`O_{b,h,i,m}=\sum_{j=0}^{T-1}P_{b,h,i,j}\,V_{b,\lfloor h/16\rfloor,j,m}`,
  "d-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{Concat}_{h=1}^{64}(O_h)W_O^\top\in\mathbb R^{B\times S\times6144}`,
  "d-add1":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "d-postnorm":String.raw`\hat U=\operatorname{RMSNorm}(U)=\frac{U}{\sqrt{\operatorname{mean}(U^2)+\varepsilon}}\odot(1+\gamma_{\mathrm{post}})`,
  "d-gateup":String.raw`\begin{aligned}G&=\hat U W_{\mathrm{gate}}^\top\\R&=\hat U W_{\mathrm{up}}^\top,\qquad G,R\in\mathbb R^{B\times S\times12288}\end{aligned}`,
  "d-swiglu":String.raw`\begin{aligned}\bar G&=\min(G,7),\qquad \bar U=\operatorname{clip}(U,-7,7)\\H&=\bar G\odot\sigma(1.702\,\bar G)\odot(\bar U+1)\end{aligned}`,
  "d-down":String.raw`Y_{\mathrm{ffn}}=HW_{\mathrm{down}}^\top\in\mathbb R^{B\times S\times6144}`,
  "d-add2":String.raw`X_{l+1}=U+Y_{\mathrm{ffn}}`,
  "s-norm":String.raw`\hat X=\frac{X}{\sqrt{\operatorname{mean}(X^2)+\varepsilon}}\odot(1+\gamma),\qquad\varepsilon=10^{-6}`,
  "s-packed":String.raw`Z=\hat X[W_Q^\top\mid W_K^\top\mid W_V^\top\mid W_{Q_i}^\top\mid W_{K_i}^\top]\in\mathbb R^{B\times S\times9856}`,
  "s-split":String.raw`Z\longrightarrow(Q_{8192},K_{512},V_{512},Q^{\mathrm{idx}}_{512},K^{\mathrm{idx}}_{128})`,
  "s-idxnorm":String.raw`\tilde Q^{\mathrm{idx}}=\operatorname{RMSNorm}(Q^{\mathrm{idx}}),\qquad\tilde K^{\mathrm{idx}}=\operatorname{RMSNorm}(K^{\mathrm{idx}})`,
  "s-idxscore":String.raw`S^{(r)}_{b,i,j}=\frac{\langle\tilde Q^{\mathrm{idx}}_{b,r,i,:},\tilde K^{\mathrm{idx}}_{b,0,j,:}\rangle}{\sqrt{128}}+M_{b,i,j}`,
  "s-blockmax":String.raw`B^{(r)}_{b,i,u}=\max_{j\in[128u,128(u+1))}S^{(r)}_{b,i,j}`,
  "s-topk":String.raw`\begin{aligned}\hat B_u&=B_u+10^{29}\mathbf1[u\in\mathcal L_i]+10^{30}\mathbf1[u\in\mathcal I]\\\mathcal S_{b,r,i}&=\operatorname{TopK}_{16}(\hat B)\end{aligned}`,
  "s-mainnorm":String.raw`\tilde Q=\operatorname{RMSNorm}(Q),\qquad\tilde K=\operatorname{RMSNorm}(K)`,
  "s-rope":String.raw`\begin{aligned}(Q^r_{:d_r},K^r_{:d_r})&=\operatorname{RoPE}(\tilde Q_{:d_r},\tilde K_{:d_r};\mathbf p),\quad d_r=64\\(Q^r_{d_r:},K^r_{d_r:})&=(\tilde Q_{d_r:},\tilde K_{d_r:})\end{aligned}`,
  "s-cache":String.raw`\mathcal K[\mathrm{slot}(r,p)]\leftarrow K^r_{r,p},\qquad\mathcal V[\mathrm{slot}(r,p)]\leftarrow V_{r,p}`,
  "s-select":String.raw`\begin{aligned}\mathcal P_{b,r,i}&=\{\mathrm{block\_table}[b,u]\mid u\in\mathcal S_{b,r,i}\}\\(K_{\mathcal S},V_{\mathcal S})&=\operatorname{gather}(\mathcal K,\mathcal V;\mathcal P_{b,r,i})\end{aligned}`,
  "s-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{128}Q^r_{b,h,i,m}(K_{\mathcal S})_{b,\lfloor h/16\rfloor,j,m},\quad j\in\mathcal S_{b,\lfloor h/16\rfloor,i}`,
  "s-scale":String.raw`\bar A_{b,h,i,j}=A_{b,h,i,j}/\sqrt{128}`,
  "s-mask":String.raw`\tilde A_{b,h,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\in\mathcal S_i\ \land\ j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}`,
  "s-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-\max_t\tilde A_{b,h,i,t})}{\sum_{t\in\mathcal S_i}\exp(\tilde A_{b,h,i,t}-\max_u\tilde A_{b,h,i,u})}`,
  "s-pv":String.raw`O_{b,h,i,m}=\sum_{j\in\mathcal S_i}P_{b,h,i,j}(V_{\mathcal S})_{b,\lfloor h/16\rfloor,j,m}`,
  "s-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{Concat}_{h=1}^{64}(O_h)W_O^\top`,
  "s-addattn":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "s-router":String.raw`\begin{aligned}r&=UW_{\mathrm{router}}^\top\in\mathbb R^{B\times S\times128}\\s&=\sigma(r),\qquad\mathcal E=\operatorname{TopK}_4(s+b)\\\hat w_e&=2\,\frac{s_e}{\sum_{j\in\mathcal E}s_j},\quad e\in\mathcal E\end{aligned}`,
  "s-experts":String.raw`\begin{aligned}g_e&=W_{1,e}u,\quad v_e=W_{3,e}u\\\bar g_e&=\min(g_e,7),\quad\bar v_e=\operatorname{clip}(v_e,-7,7)\\E_e(u)&=W_{2,e}[\bar g_e\odot\sigma(1.702\bar g_e)\odot(\bar v_e+1)]\end{aligned}`,
  "s-shared":String.raw`E_{\mathrm{shared}}(u)=W_{2,s}\operatorname{SwiGLUOAI}(W_{1,s}u,W_{3,s}u)`,
  "s-sum":String.raw`Y_{\mathrm{moe}}=\sum_{e\in\mathcal E}\hat w_eE_e(U)+E_{\mathrm{shared}}(U)`,
  "s-addout":String.raw`X_{l+1}=U+Y_{\mathrm{moe}}`,
};

const MLP_SECTIONS: CodeSection[] = [
  {stage:"1 · DEFINE",title:"MiniMaxM3MLP.__init__：成员真实类型",location:"nvidia/model.py · L136–163",url:`${CODE_URL}#L136-L163`,code:`self.gate_up_proj = MergedColumnParallelLinear(
    config.hidden_size,
    [intermediate_size] * 2,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    config.hidden_size,
    bias=False,
    quant_config=quant_config,
    reduce_results=reduce_results,
    prefix=f"{prefix}.down_proj",
)
self.act_fn = SiluAndMulWithClamp(
    swiglu_limit=config.swiglu_limit,  # 7.0
    alpha=config.swiglu_alpha,        # 1.702
    beta=config.swiglu_beta,          # 1.0
)`},
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
  {stage:"4 · KERNEL",title:"CUDA 路径：同一语义的自定义算子",location:"activation.py · L219–224",url:`${ACTIVATION_URL}#L219-L224`,code:`def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    self.op(out, x, self.swiglu_limit, self.alpha, self.beta)
    return out

# self.op = torch.ops._C.silu_and_mul_with_clamp`},
];

const MLP_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.gate_up_proj",resolvesTo:"MergedColumnParallelLinear",meaning:"一次并行 GEMM 产生 packed [gate | up]，随后沿最后一维平分。"},
  {symbol:"self.act_fn",resolvesTo:"SiluAndMulWithClamp",meaning:"不是未说明的黑盒 SiLU；内部完成 split、clamp、sigmoid 与逐元素乘法。"},
  {symbol:"self.down_proj",resolvesTo:"RowParallelLinear",meaning:"把激活后的中间维投回 hidden_size，并按配置归并 TP 结果。"},
  {symbol:"swiglu_limit / alpha / beta",resolvesTo:"7.0 / 1.702 / 1.0",meaning:"来自 MiniMax-M3 config，并直接传入激活算子。"},
];

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

const MOE_SECTIONS: CodeSection[] = [
  {stage:"1 · ROUTE",title:"MiniMaxM3MoE.forward：router logits",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)`},
  {stage:"2 · SHARED",title:"共享专家复用 MiniMaxM3MLP",location:"nvidia/model.py · MiniMaxM3MoE.forward",url:CODE_URL,code:`shared_hidden_states = self.shared_experts(hidden_states)
final_hidden_states = final_hidden_states + shared_hidden_states
return final_hidden_states.view(num_tokens, hidden_dim)`},
  {stage:"3 · CONFIG",title:"FusedMoEFactory：routed expert 配置",location:"nvidia/model.py · MiniMaxM3MoE.__init__",url:CODE_URL,code:`FusedMoEFactory(
    num_experts=128,
    top_k=4,
    hidden_size=6144,
    intermediate_size=3072,
    activation="swigluoai_uninterleave",
    routed_scaling_factor=2.0,
)`},
];

const MOE_SYMBOLS: CodeSymbol[] = [
  {symbol:"self.gate",resolvesTo:"GateLinear",meaning:"输出 128 个 FP32 router logits；Top-4 路由由 fused MoE 消费。"},
  {symbol:"self.experts",resolvesTo:"FusedMoE",meaning:"把 w1/w3 打包为 w13，并对每个 token 执行 4 个 routed experts。"},
  {symbol:"self.shared_experts",resolvesTo:"MiniMaxM3MLP",meaning:"所有 token 都执行，内部的 self.act_fn 同样是 SiluAndMulWithClamp。"},
  {symbol:"activation",resolvesTo:"swigluoai_uninterleave",meaning:"routed-expert fused kernel 中与 dense/shared 分支等价的 SwiGLU-OAI 语义。"},
];

const CODE_BY_ID: Record<string, CodeDetail> = {};
for(const id of ["d-gateup","d-swiglu","d-down","s-shared"]) CODE_BY_ID[id]={sections:MLP_SECTIONS,symbols:MLP_SYMBOLS};
for(const id of ["d-qkv","d-split","d-qnorm","d-knorm","d-ropeq","d-ropek","d-cache","d-qk","d-scale","d-mask","d-softmax","d-pv","d-oproj","s-packed","s-split","s-mainnorm","s-rope","s-cache","s-select","s-qk","s-scale","s-mask","s-softmax","s-pv","s-oproj"]) CODE_BY_ID[id]={sections:ATTENTION_SECTIONS,symbols:ATTENTION_SYMBOLS};
for(const id of ["s-router","s-experts","s-shared","s-sum"]) CODE_BY_ID[id]={sections:id==="s-shared"?[...MOE_SECTIONS,...MLP_SECTIONS]:MOE_SECTIONS,symbols:id==="s-shared"?[...MOE_SYMBOLS,...MLP_SYMBOLS]:MOE_SYMBOLS};

const cloneOp = (base: Node, values: Partial<OpNode> & { id: string; kind: OpKind; title: string }): OpNode => {
  const detail=CODE_BY_ID[values.id];
  return { ...base, ...values, latex:values.latex??LATEX_BY_ID[values.id], codeSections:values.codeSections??detail?.sections, codeSymbols:values.codeSymbols??detail?.symbols };
};
const pinSource = (url: string) => url.replace("/blob/main/", `/blob/${VLLM_COMMIT}/`);

function denseGraph(layer: number): Record<string, OpNode> {
  const [norm, qkv, attn, out, mlp] = denseNodes(layer);
  const shard = layerShard(layer);
  const postNorm: Weight = { key: `language_model.model.layers.${layer}.post_attention_layernorm.weight`, shape: "[6144]", dtype: "BF16", shard, params: "6,144" };
  return {
    input: cloneOp(norm,{id:"d-input",kind:"io",title:"Hidden states",kicker:`L${layer} INPUT`,input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[],formula:"residual ← Xₗ; working ← Xₗ"}),
    position: cloneOp(attn,{id:"d-position",kind:"route",title:"Build Position IDs",kicker:"vLLM RUNTIME I/O",input:"num_computed_tokens + query offsets",inputShape:"[B] + [Nq]",output:"positions",outputShape:"[Nq]",formula:"position(req,i)=num_computed_tokens[req]+i",formulaNote:"positions 不是模型权重，也不是在 Attention 内凭空产生；由 vLLM runner 根据每个请求已计算 token 数和本轮 query 偏移生成。",source:"gpu_model_runner.py · _prepare_inputs",sourceUrl:RUNNER_URL,weights:[]}),
    attnmeta: cloneOp(attn,{id:"d-attnmeta",kind:"mask",title:"Build Attention Metadata",kicker:"vLLM RUNTIME I/O",input:"query_start_loc, seq_lens, causal=True",inputShape:"[B+1] · [B] · bool",output:"implicit causal / padding layout",outputShape:"backend metadata; 非稠密 [S,T]",formula:"valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)",formulaNote:"优化推理中通常不会真的构造 [S,T] mask；causal、query_start_loc 与 seq_lens 被后端内核直接消费。",source:"gpu_model_runner.py · CommonAttentionMetadata",sourceUrl:RUNNER_URL,weights:[]}),
    slots: cloneOp(attn,{id:"d-slots",kind:"route",title:"Resolve KV Slots",kicker:"vLLM RUNTIME I/O",input:"positions + block_table",inputShape:"[Nq] + [B,Nblocks]",output:"slot_mapping + block_table",outputShape:"[Nq] + [B,Nblocks]",formula:"slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)",formulaNote:"slot_mapping 决定新 K/V 写到哪个物理槽；block_table 决定 Attention 从哪些物理 pages 读取。",source:"gpu_model_runner.py · compute_slot_mapping",sourceUrl:RUNNER_URL,weights:[]}),
    norm: cloneOp(norm,{id:"d-norm",kind:"norm",title:"RMSNorm"}),
    qkv: cloneOp(qkv,{id:"d-qkv",kind:"linear",title:"QKV Projection"}),
    split: cloneOp(qkv,{id:"d-split",kind:"split",title:"Split Q / K / V",input:"packed qkv",inputShape:"[B,S,9216]",output:"Q · K · V",outputShape:"8192 · 512 · 512",formula:"split(qkv,[8192,512,512],dim=-1)",formulaNote:"checkpoint 中三块矩阵分离；vLLM 运行时一次 GEMM 后切分。",weights:[]}),
    qnorm: cloneOp(attn,{id:"d-qnorm",kind:"norm",title:"Q RMSNorm",input:"Q",inputShape:"[B,64,S,128]",output:"Q̃",outputShape:"[B,64,S,128]",formula:"Q̃=Q/√(mean(Q²)+ε)⊙(1+γq)",weights:attn.weights.filter(w=>w.key.includes("q_norm"))}),
    knorm: cloneOp(attn,{id:"d-knorm",kind:"norm",title:"K RMSNorm",input:"K",inputShape:"[B,4,T,128]",output:"K̃",outputShape:"[B,4,T,128]",formula:"K̃=K/√(mean(K²)+ε)⊙(1+γk)",weights:attn.weights.filter(w=>w.key.includes("k_norm"))}),
    ropeq: cloneOp(attn,{id:"d-ropeq",kind:"rope",title:"Partial RoPE (Q)",input:"Q̃ + positions",inputShape:"[B,64,S,128] + [S]",output:"Qᵣ",outputShape:"[B,64,S,128]",formula:"Qᵣ[:64]=RoPE(Q̃[:64],pos); Qᵣ[64:]=Q̃[64:]",weights:[]}),
    ropek: cloneOp(attn,{id:"d-ropek",kind:"rope",title:"Partial RoPE (K)",input:"K̃ + positions",inputShape:"[B,4,T,128] + [T]",output:"Kᵣ",outputShape:"[B,4,T,128]",formula:"Kᵣ[:64]=RoPE(K̃[:64],pos); Kᵣ[64:]=K̃[64:]",weights:[]}),
    cache: cloneOp(attn,{id:"d-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"[T,4,128] ×2",output:"visible K,V",outputShape:"[B,4,T,128] ×2",formula:"slot = block_table[seq, logical_block] + offset",formulaNote:"Dense 层读取完整可见历史；block table 决定物理 page。",weights:[]}),
    qk: cloneOp(attn,{id:"d-qk",kind:"matmul",title:"Q × Kᵀ",input:"Qᵣ,Kᵣ",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"scores",outputShape:"[B,64,S,T]",formula:"A=QᵣKᵣᵀ",weights:[]}),
    scale: cloneOp(attn,{id:"d-scale",kind:"scale",title:"Scale 1/√128",input:"A",inputShape:"[B,64,S,T]",output:"scaled scores",outputShape:"[B,64,S,T]",formula:"A←A/√128",weights:[]}),
    mask: cloneOp(attn,{id:"d-mask",kind:"mask",title:"Apply Causal / Pad Bounds",input:"scores + attention metadata",inputShape:"[B,64,S,T] + runtime metadata",output:"masked scores",outputShape:"[B,64,S,T]",formula:"Aᵢⱼ←valid(i,j) ? Aᵢⱼ : −∞",formulaNote:"图中把 mask 画成逻辑算子；vLLM 后端实际以 causal、seq_lens 和 query_start_loc 实现，不物化完整 mask 矩阵。",weights:[]}),
    softmax: cloneOp(attn,{id:"d-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64,S,T]",output:"attention prob",outputShape:"[B,64,S,T]",formula:"P=softmax(A,dim=-1)",weights:[]}),
    pv: cloneOp(attn,{id:"d-pv",kind:"matmul",title:"P × V",input:"P,V",inputShape:"[B,64,S,T] · [B,4,T,128]",output:"heads",outputShape:"[B,S,8192]",formula:"Oₕ=PₕV⌊h/16⌋",weights:[]}),
    oproj: cloneOp(out,{id:"d-oproj",kind:"linear",title:"O Projection"}),
    add1: cloneOp(out,{id:"d-add1",kind:"add",title:"+ Attention Residual",input:"Yattn + residual",inputShape:"2 × [B,S,6144]",output:"U",outputShape:"[B,S,6144]",formula:"U=residual+Yattn",weights:[]}),
    postnorm: cloneOp(norm,{id:"d-postnorm",kind:"norm",title:"Post-attn RMSNorm",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",weights:[postNorm]}),
    gateup: cloneOp(mlp,{id:"d-gateup",kind:"linear",title:"Gate + Up Projection",output:"gate · up",outputShape:"2 × [B,S,12288]",weights:mlp.weights.filter(w=>!w.key.includes("down_proj"))}),
    swiglu: cloneOp(mlp,{id:"d-swiglu",kind:"activation",title:"SwiGLU-OAI",input:"gate,up",inputShape:"2 × [B,S,12288]",output:"activated",outputShape:"[B,S,12288]",weights:[]}),
    down: cloneOp(mlp,{id:"d-down",kind:"linear",title:"Down Projection",input:"activated",inputShape:"[B,S,12288]",output:"Yffn",outputShape:"[B,S,6144]",weights:mlp.weights.filter(w=>w.key.includes("down_proj"))}),
    add2: cloneOp(mlp,{id:"d-add2",kind:"add",title:"+ MLP Residual",input:"Yffn + U",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+MLP(RMSNorm(U))",weights:[]}),
  };
}

function sparseGraph(layer: number): Record<string, OpNode> {
  const [packed,indexer,topk,attn,router,experts,shared,combine]=sparseNodes(layer);
  const shard=layerShard(layer);
  const inputNorm: Weight={key:`language_model.model.layers.${layer}.input_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  return {
    input:cloneOp(packed,{id:"s-input",kind:"io",title:"Hidden states",input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[]}),
    position:cloneOp(attn,{id:"s-position",kind:"route",title:"Build Position IDs",kicker:"vLLM RUNTIME I/O",input:"num_computed_tokens + query offsets",inputShape:"[B] + [Nq]",output:"positions",outputShape:"[Nq]",formula:"position(req,i)=num_computed_tokens[req]+i",formulaNote:"positions 由 vLLM runner 在模型 forward 之前构造，再传给 MiniMax-M3 的 fused QKNorm + RoPE kernel。",source:"gpu_model_runner.py · _prepare_inputs",sourceUrl:RUNNER_URL,weights:[]}),
    attnmeta:cloneOp(attn,{id:"s-attnmeta",kind:"mask",title:"Build Attention Metadata",kicker:"vLLM RUNTIME I/O",input:"query_start_loc, seq_lens, causal=True",inputShape:"[B+1] · [B] · bool",output:"implicit causal / padding layout",outputShape:"backend metadata; 非稠密 [S,T]",formula:"valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)",formulaNote:"同一份边界元数据同时约束 indexer 的 block selection 和 main sparse attention。",source:"gpu_model_runner.py · CommonAttentionMetadata",sourceUrl:RUNNER_URL,weights:[]}),
    slots:cloneOp(attn,{id:"s-slots",kind:"route",title:"Resolve KV Slots",kicker:"vLLM RUNTIME I/O",input:"positions + block_table",inputShape:"[Nq] + [B,Nblocks]",output:"slot_mapping + block_table",outputShape:"[Nq] + [B,Nblocks]",formula:"slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)",formulaNote:"slot_mapping 用于 K/V 写入；block_table 把 indexer 选出的逻辑 block id 翻译为物理 page。",source:"gpu_model_runner.py · compute_slot_mapping",sourceUrl:RUNNER_URL,weights:[]}),
    norm:cloneOp(packed,{id:"s-norm",kind:"norm",title:"RMSNorm",input:"Xₗ",inputShape:"[B,S,6144]",output:"X̂",outputShape:"[B,S,6144]",weights:[inputNorm]}),
    packed:cloneOp(packed,{id:"s-packed",kind:"linear",title:"QKV + Index Projection"}),
    split:cloneOp(packed,{id:"s-split",kind:"split",title:"Split 5 outputs",input:"packed projection",inputShape:"[B,S,9856]",output:"Q/K/V · Qidx/Kidx",outputShape:"8192/512/512 · 512/128",formula:"split(x,[8192,512,512,512,128],dim=-1)",weights:[]}),
    idxnorm:cloneOp(indexer,{id:"s-idxnorm",kind:"norm",title:"Index Q/K Norm",input:"Qidx,Kidx",inputShape:"[B,S,4,128] · [B,T,1,128]",output:"Q̃idx,K̃idx",outputShape:"same",weights:indexer.weights}),
    idxscore:cloneOp(indexer,{id:"s-idxscore",kind:"matmul",title:"Index Q × Kᵀ",input:"Q̃idx,K̃idx",inputShape:"[B,4,S,128] · [B,1,T,128]",output:"token scores",outputShape:"[B,4,S,T]",weights:[]}),
    blockmax:cloneOp(indexer,{id:"s-blockmax",kind:"route",title:"Block Max (128 tokens)",input:"causal token scores",inputShape:"[B,4,S,T]",output:"block scores",outputShape:"[B,4,S,⌈T/128⌉]",weights:[]}),
    topk:cloneOp(topk,{id:"s-topk",kind:"route",title:"Top-16 Blocks",input:"block scores + local priority",inputShape:"[B,4,S,Nblocks]",output:"logical block ids",outputShape:"[B,S,4,16]"}),
    mainnorm:cloneOp(attn,{id:"s-mainnorm",kind:"norm",title:"Main Q/K Norm",input:"Q,K",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"Q̃,K̃",outputShape:"same",weights:attn.weights.filter(w=>w.key.includes("_norm"))}),
    rope:cloneOp(attn,{id:"s-rope",kind:"rope",title:"Partial RoPE",input:"Q̃,K̃ + positions",inputShape:"Q/K + [S]",output:"Qᵣ,Kᵣ",outputShape:"Q/K unchanged",weights:[]}),
    cache:cloneOp(attn,{id:"s-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"KV pages + [B,Nblocks]",output:"paged K,V",outputShape:"[Npages,128,4,128] ×2",formula:"physical_page=block_table[logical_block]",weights:[]}),
    select:cloneOp(attn,{id:"s-select",kind:"route",title:"Select KV Pages",input:"paged K,V + Top-16 block ids",inputShape:"KV pages + [B,S,4,16]",output:"selected K,V",outputShape:"≤2048 KV tokens / group",formula:"physical_page=block_table[logical_top16]",weights:[]}),
    qk:cloneOp(attn,{id:"s-qk",kind:"matmul",title:"Q × selected Kᵀ",input:"Qᵣ, selected K",inputShape:"[B,64,S,128] · ≤16×128",output:"sparse scores",outputShape:"[B,64,S,≤2048]",weights:[]}),
    scale:cloneOp(attn,{id:"s-scale",kind:"scale",title:"Scale 1/√128",input:"scores",inputShape:"[B,64,S,≤2048]",output:"scaled scores",outputShape:"same",weights:[]}),
    mask:cloneOp(attn,{id:"s-mask",kind:"mask",title:"Apply Causal / Pad Bounds",input:"scores + attention metadata",inputShape:"sparse scores + runtime metadata",output:"masked scores",outputShape:"same",formula:"Aᵢⱼ←valid_sparse(i,j) ? Aᵢⱼ : −∞",formulaNote:"Top-16 只决定候选 KV blocks；causal/padding 边界仍会在最终 Attention kernel 内再次约束可见 token。",weights:[]}),
    softmax:cloneOp(attn,{id:"s-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64,S,≤2048]",output:"probabilities",outputShape:"same",weights:[]}),
    pv:cloneOp(attn,{id:"s-pv",kind:"matmul",title:"P × selected V",input:"P, selected V",inputShape:"probabilities · KV pages",output:"heads",outputShape:"[B,S,8192]",weights:[]}),
    oproj:cloneOp(attn,{id:"s-oproj",kind:"linear",title:"O Projection",input:"heads",inputShape:"[B,S,8192]",output:"Yattn",outputShape:"[B,S,6144]",weights:attn.weights.filter(w=>w.key.includes("o_proj"))}),
    addattn:cloneOp(combine,{id:"s-addattn",kind:"add",title:"+ Attention Residual",input:"Yattn + residual",inputShape:"2 × [B,S,6144]",output:"U",outputShape:"[B,S,6144]",weights:[]}),
    router:cloneOp(router,{id:"s-router",kind:"route",title:"FP32 Router → Top-4"}),
    experts:cloneOp(experts,{id:"s-experts",kind:"activation",title:"Routed Experts ×4"}),
    shared:cloneOp(shared,{id:"s-shared",kind:"activation",title:"Shared Expert ×1"}),
    sum:cloneOp(combine,{id:"s-sum",kind:"add",title:"Weighted Sum",input:"4 routed + shared",inputShape:"5 × [B,S,6144]",output:"Ymoe",outputShape:"[B,S,6144]",weights:[]}),
    addout:cloneOp(combine,{id:"s-addout",kind:"add",title:"+ Decoder Residual"}),
  };
}

function Op({node,active,onHover,onLeave,onSelect}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  return <button className={`op-node op-${node.kind} ${active?"active":""}`} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onClick={()=>onSelect(node)}><small>OP · {node.kind}</small><b>{node.title}</b></button>;
}

type TensorRole = "input" | "tensor" | "output" | "side";

function Tensor({name,shape,role="tensor"}:{name:string;shape:string;role?:TensorRole}){
  const label={input:"INPUT",tensor:"TENSOR",output:"OUTPUT",side:"SIDE INPUT"}[role];
  return <div className={`tensor-node tensor-${role}`}><small>{label}</small><b>{name}</b><code>{shape}</code></div>;
}

const Arrow=({label}:{label?:string})=><span className="op-arrow"><i/>{label&&<small>{label}</small>}</span>;

function RuntimeIORail({N}:{N:({id}:{id:string})=>ReactNode}){
  return <section className="runtime-io"><header><b>ATTENTION RUNTIME I/O</b><span>这些输入由 vLLM runner 生成并传入模型；mask 在内核中按边界隐式执行</span></header><div className="runtime-io-grid">
    <div className="io-lane"><Tensor name="num_computed_tokens · query offsets" shape="[B] + [Nq]" role="input"/><Arrow/><N id="position"/><Arrow/><Tensor name="positions → RoPE" shape="[Nq]"/></div>
    <div className="io-lane"><Tensor name="query_start_loc · seq_lens · causal" shape="[B+1] · [B] · True" role="input"/><Arrow/><N id="attnmeta"/><Arrow/><Tensor name="causal / padding layout → Attention" shape="implicit · 非稠密 [S,T]"/></div>
    <div className="io-lane"><Tensor name="positions · block_table" shape="[Nq] + [B,Nblocks]" role="input"/><Arrow/><N id="slots"/><Arrow/><Tensor name="slot_mapping · block_table → KV Cache" shape="[Nq] + [B,Nblocks]"/></div>
  </div></section>;
}

/* eslint-disable react-hooks/static-components -- local alias only shortens a large, stateless operator graph */
function DenseDiagram({g,active,onHover,onLeave,onSelect}:{g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect}; const N=({id}:{id:string})=><Op node={g[id]} {...p} active={active===g[id].id}/>;
  return <div className="operator-diagram dense-diagram">
    <RuntimeIORail N={N}/>
    <div className="flow-row"><Tensor name="Xₗ · hidden_states" shape="[B,S,6144]" role="input"/><Arrow/><N id="norm"/><Arrow/><Tensor name="X̂" shape="[B,S,6144]"/><Arrow/><N id="qkv"/><Arrow/><Tensor name="packed_qkv" shape="[B,S,9216]"/><Arrow/><N id="split"/><Arrow/><Tensor name="Q · K · V" shape="8192 · 512 · 512"/></div>
    <div className="branch-box qkv-branches">
      <section><header>Q BRANCH · positions 来自上方 I/O</header><div className="mini-flow"><Tensor name="Q · positions" shape="[B,64,S,128] + [Nq]"/><Arrow/><N id="qnorm"/><Arrow/><Tensor name="Q̃ · positions" shape="same + [Nq]"/><Arrow/><N id="ropeq"/><Arrow/><Tensor name="Qᵣ" shape="[B,64,S,128]"/></div></section>
      <section><header>K BRANCH · positions 来自上方 I/O</header><div className="mini-flow"><Tensor name="K · positions" shape="[B,4,S,128] + [Nq]"/><Arrow/><N id="knorm"/><Arrow/><Tensor name="K̃ · positions" shape="same + [Nq]"/><Arrow/><N id="ropek"/><Arrow/><Tensor name="Kᵣ" shape="[B,4,S,128]"/></div></section>
      <section><header>KV MEMORY · metadata 来自上方 I/O</header><div className="mini-flow"><Tensor name="Kᵣ · V · slot_mapping" shape="KV + [Nq]"/><Arrow/><N id="cache"/><Arrow/><Tensor name="paged K · V · block_table" shape="KV pages + [B,Nblocks]"/></div></section>
    </div>
    <div className="flow-row attention-row"><Tensor name="Qᵣ · paged K · block_table" shape="Q [B,64,S,128] · paged K"/><Arrow/><N id="qk"/><Arrow/><Tensor name="A" shape="[B,64,S,T]"/><Arrow/><N id="scale"/><Arrow/><Tensor name="scaled A · causal/pad layout" shape="scores + runtime metadata"/><Arrow/><N id="mask"/><Arrow/><Tensor name="A masked" shape="[B,64,S,T]"/><Arrow/><N id="softmax"/><Arrow/><Tensor name="P" shape="[B,64,S,T]"/></div>
    <div className="flow-row"><Tensor name="P · Vcache" shape="P [B,64,S,T] · V [B,4,T,128]"/><Arrow/><N id="pv"/><Arrow/><Tensor name="heads" shape="[B,S,8192]"/><Arrow/><N id="oproj"/><Arrow/><Tensor name="Yattn · Xₗ" shape="2 × [B,S,6144]"/><Arrow/><N id="add1"/><Arrow/><Tensor name="U" shape="[B,S,6144]"/></div>
    <div className="flow-row"><Tensor name="U" shape="[B,S,6144]"/><Arrow/><N id="postnorm"/><Arrow/><Tensor name="Û" shape="[B,S,6144]"/><Arrow/><N id="gateup"/><Arrow/><Tensor name="gate · up" shape="2 × [B,S,12288]"/><Arrow/><N id="swiglu"/><Arrow/><Tensor name="activated" shape="[B,S,12288]"/><Arrow/><N id="down"/><Arrow/><Tensor name="Yffn · U" shape="2 × [B,S,6144]"/><Arrow/><N id="add2"/><Arrow/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,6144]" role="output"/></div>
  </div>;
}

function SparseDiagram({g,active,onHover,onLeave,onSelect}:{g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect}; const N=({id}:{id:string})=><Op node={g[id]} {...p} active={active===g[id].id}/>;
  return <div className="operator-diagram sparse-diagram">
    <RuntimeIORail N={N}/>
    <div className="flow-row"><Tensor name="Xₗ · hidden_states" shape="[B,S,6144]" role="input"/><Arrow/><N id="norm"/><Arrow/><Tensor name="X̂" shape="[B,S,6144]"/><Arrow/><N id="packed"/><Arrow/><Tensor name="packed_5" shape="[B,S,9856]"/><Arrow/><N id="split"/><Arrow/><Tensor name="Q · K · V · Qidx · Kidx" shape="8192 · 512 · 512 · 512 · 128"/></div>
    <div className="dual-path">
      <section><header>INDEX PATH · causal/pad layout 来自上方 I/O</header><div className="mini-flow"><Tensor name="Qidx · Kidx" shape="[B,4,S,128] · [B,1,T,128]"/><Arrow/><N id="idxnorm"/><Arrow/><Tensor name="Q̃idx · K̃idx" shape="same"/><Arrow/><N id="idxscore"/><Arrow/><Tensor name="token scores · causal bounds" shape="[B,4,S,T] + metadata"/><Arrow/><N id="blockmax"/><Arrow/><Tensor name="block scores · local/init priority" shape="[B,4,S,⌈T/128⌉]"/><Arrow/><N id="topk"/><Arrow/><Tensor name="Top-16 block ids" shape="[B,S,4,16]"/></div></section>
      <section><header>MAIN PATH · positions 来自上方 I/O</header><div className="mini-flow"><Tensor name="Q · K" shape="[B,64,S,128] · [B,4,S,128]"/><Arrow/><N id="mainnorm"/><Arrow/><Tensor name="Q̃ · K̃ · positions" shape="same + [Nq]"/><Arrow/><N id="rope"/><Arrow/><Tensor name="Qᵣ · Kᵣ" shape="same"/></div></section>
    </div>
    <div className="flow-row"><Tensor name="Kᵣ · V · slot_mapping" shape="KV + [Nq]"/><Arrow/><N id="cache"/><Arrow/><Tensor name="paged K · V · block_table · Top-16 ids" shape="KV pages + runtime metadata"/><Arrow/><N id="select"/><Arrow/><Tensor name="selected K · V" shape="≤2048 tokens / group"/></div>
    <div className="flow-row attention-row"><Tensor name="Qᵣ · selected K" shape="Q · Kselected"/><Arrow/><N id="qk"/><Arrow/><Tensor name="sparse scores" shape="[B,64,S,≤2048]"/><Arrow/><N id="scale"/><Arrow/><Tensor name="scaled scores · causal/pad layout" shape="scores + runtime metadata"/><Arrow/><N id="mask"/><Arrow/><Tensor name="masked scores" shape="same"/><Arrow/><N id="softmax"/><Arrow/><Tensor name="P" shape="same"/></div>
    <div className="flow-row"><Tensor name="P · selected V" shape="probabilities · Vselected"/><Arrow/><N id="pv"/><Arrow/><Tensor name="heads" shape="[B,S,8192]"/><Arrow/><N id="oproj"/><Arrow/><Tensor name="Yattn · Xₗ" shape="2 × [B,S,6144]"/><Arrow/><N id="addattn"/><Arrow/><Tensor name="U" shape="[B,S,6144]"/></div>
    <div className="flow-row moe-path"><Tensor name="U" shape="[B,S,6144]"/><Arrow/><N id="router"/><Arrow/><Tensor name="expert ids · weights" shape="Top-4 / token"/><Arrow/><div className="parallel-ops"><N id="experts"/><N id="shared"/></div><Arrow/><Tensor name="4 routed · 1 shared" shape="5 × [B,S,6144]"/><Arrow/><N id="sum"/><Arrow/><Tensor name="Ymoe · U" shape="2 × [B,S,6144]"/><Arrow/><N id="addout"/><Arrow/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,6144]" role="output"/></div>
  </div>;
}
/* eslint-enable react-hooks/static-components */

function LayerNavigator({layer,onChange}:{layer:number;onChange:(n:number)=>void}){
  const ticksRef=useRef<HTMLDivElement>(null);
  useEffect(()=>{ticksRef.current?.querySelector(".active")?.scrollIntoView({behavior:"smooth",block:"nearest",inline:"center"})},[layer]);
  return <div className="layer-nav"><div className="layer-nav-head"><div><span>DECODER LAYER</span><b>L{layer}</b><small>{layer<3?"Dense GQA + Dense MLP":"MSA + Top-4 MoE"}</small></div><div className="layer-type-legend"><span><i className="dense"/>Dense · L0–2</span><span><i className="sparse"/>MSA+MoE · L3–59</span></div></div><div className="layer-ticks" ref={ticksRef}>{Array.from({length:60},(_,i)=><button key={i} className={`${i<3?"dense":"sparse"} ${i===layer?"active":""}`} onClick={()=>onChange(i)} title={`L${i} · ${i<3?"Dense":"MSA+MoE"}`}>{i}</button>)}</div><div className="layer-slider"><span>L0</span><input type="range" min="0" max="59" value={layer} onChange={e=>onChange(Number(e.target.value))}/><span>L59</span></div></div>;
}

function LatexFormula({node}:{node:OpNode}){
  if(!node.latex)return <code className="formula-fallback">{node.formula}</code>;
  const html=katex.renderToString(node.latex,{displayMode:true,throwOnError:false,strict:"ignore",output:"htmlAndMathml"});
  return <div className="latex-render" aria-label={`${node.title} 完整计算公式`} dangerouslySetInnerHTML={{__html:html}}/>;
}

function CodeView({node}:{node:OpNode}){
  const sections=node.codeSections??[{stage:"SOURCE",title:"当前模块摘录",location:node.source,code:node.code,url:pinSource(node.sourceUrl)}];
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {VLLM_COMMIT.slice(0,7)}</span><b>{node.source}</b><i>↗</i></a>
    {!!node.codeSymbols?.length&&<section className="code-symbols"><header><span>OBJECT RESOLUTION</span><b>对象解析</b></header>{node.codeSymbols.map(item=><article key={`${node.id}-${item.symbol}`}><code>{item.symbol}</code><i>→</i><b>{item.resolvesTo}</b><p>{item.meaning}</p></article>)}</section>}
    <section className="code-call-chain"><header><span>CALL CHAIN</span><b>从父模块追到实际运算</b></header>{sections.map((section,index)=><article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><span>{section.stage}</span><b>{section.title}</b><small>{section.location}</small></div>{section.url&&<a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article>)}</section>
  </div>;
}

function DetailPanel({node,tab,setTab}:{node:OpNode;tab:Tab;setTab:(t:Tab)=>void}){
  const tabs:[Tab,string][]=[["io","I/O"],["formula","公式"],["code","代码"],["weights","权重"]];
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div><i className={`kind-dot op-${node.kind}`}/><p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div className="detail-content">
    {tab==="io"&&<div className="shape-view"><article><span>INPUT</span><b>{node.input}</b><code>{node.inputShape}</code></article><i>→</i><article><span>OUTPUT</span><b>{node.output}</b><code>{node.outputShape}</code></article></div>}
    {tab==="formula"&&<div className="formula-view"><span>LATEX · FULL COMPUTE</span><LatexFormula node={node}/><div className="formula-implementation"><b>实现摘要</b><code>{node.formula}</code></div><p>{node.formulaNote}</p></div>}
    {tab==="code"&&<CodeView node={node}/>}
    {tab==="weights"&&<WeightView weights={node.weights}/>}</div><footer>vLLM @ {VLLM_COMMIT.slice(0,7)} · official safetensors</footer></aside>;
}

function WeightView({weights}:{weights:Weight[]}){return weights.length?<div className="weight-view">{weights.map(w=><article key={w.key}><code>{w.key}</code><div><b>{w.dtype}</b><span>{w.shape}</span>{w.params&&<em>{w.params}</em>}</div><small>{w.shard}</small>{w.runtime&&<small>→ {w.runtime}</small>}</article>)}</div>:<div className="empty-weight"><b>无可训练权重</b><p>这是 shape、mask、缓存、路由选择或逐元素计算。</p></div>}

function HelpModal({onClose}:{onClose:()=>void}){
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="参数、符号与映射说明"><header><div><span>REFERENCE</span><h2>参数、符号与运行时映射</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><div className="help-grid"><section><h3>Shape 符号</h3><table><tbody><tr><th>B</th><td>batch size</td><th>S</th><td>当前 query token 数</td></tr><tr><th>Nq</th><td>本轮所有 query tokens</td><th>T</th><td>含历史 cache 的 KV 长度</td></tr><tr><th>d</th><td>head dim = 128</td><th>V</th><td>vocab = 200064</td></tr><tr><th>E</th><td>routed experts = 128</td><th>K</th><td>Top-K experts = 4</td></tr></tbody></table></section><section><h3>节点与底色</h3><div className="node-rule"><i className="input-swatch"/><b>输入</b><span>→</span><i className="operator-swatch"/><b>计算算子</b><span>→</span><i className="tensor-swatch"/><b>中间张量</b><span>→</span><i className="output-swatch"/><b>输出</b></div><p className="node-rule-note">Attention 的 mask 在 vLLM 中不是显式 [S,T] 张量：runner 生成 query_start_loc、seq_lens、causal=True、block_table 与 slot_mapping，后端内核直接据此限制可见 token。</p><div className="color-legend">{[["norm","Norm"],["linear","Linear"],["matmul","MatMul"],["rope","RoPE"],["scale","Scale"],["mask","Mask / Bounds"],["activation","Activation"],["route","Runtime / Routing"],["cache","Cache"],["add","Residual Add"]].map(([kind,label])=><span key={kind}><i className={`op-${kind}`}/>{label}</span>)}</div></section><section className="mapping-section"><h3>Checkpoint → vLLM runtime</h3><div className="mapping-table"><div><code>q_proj · k_proj · v_proj</code><span>→</span><b>QKVParallelLinear.qkv_proj</b></div><div><code>q/k/v + index_q/index_k</code><span>→</span><b>MinimaxM3QKV…WithIndexer</b></div><div><code>gate_proj · up_proj</code><span>→</span><b>gate_up_proj</b></div><div><code>experts.*.w1 · w3 · w2</code><span>→</span><b>FusedMoE w13 · w2</b></div></div></section></div></section></div>;
}

export default function Home(){
  const [layer,setLayer]=useState(3); const [tab,setTab]=useState<Tab>("io"); const [dark,setDark]=useState(false); const [help,setHelp]=useState(false);
  const graph=layer<3?denseGraph(layer):sparseGraph(layer); const [pinned,setPinned]=useState<OpNode>(sparseGraph(3).packed); const [hovered,setHovered]=useState<OpNode|null>(null); const active=hovered??pinned;
  const changeLayer=(next:number)=>{const g=next<3?denseGraph(next):sparseGraph(next);setLayer(next);setPinned(next<3?g.qkv:g.packed);setHovered(null)};
  const vision=visionNodes;
  return <main className={`atlas-app ${dark?"dark":""}`}><header className="app-header">
    <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value="minimax-m3" onChange={()=>undefined}>{MODEL_REGISTRY.map(m=><option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}</option>)}</select></label>
    <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>模型结构概览</b><small>MiniMax-M3</small></div></div>
    <nav className="resource-links"><a href={CODE_URL} target="_blank" rel="noreferrer"><b>CODE ↗</b><small>vLLM @ {VLLM_COMMIT.slice(0,7)}</small></a><a href={WEIGHTS_URL} target="_blank" rel="noreferrer"><b>WEIGHTS ↗</b><small>Hugging Face · 59 shards</small></a></nav>
    <div className="model-facts"><span><b>428B</b><small>模型总参数量</small></span><span><b>23B</b><small>每 token 激活参数</small></span><span><b>1M</b><small>最大上下文 token</small></span><span><b>869 GB</b><small>BF16 checkpoint</small></span></div>
    <button className="help-button" onClick={()=>setHelp(true)} aria-label="查看参数和符号说明">?</button><button className="theme-button" onClick={()=>setDark(v=>!v)} aria-label="切换明暗主题">{dark?"☀":"☾"}</button>
  </header><div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><button onMouseEnter={()=>setHovered(cloneOp(vision[0],{id:"overview-input",kind:"io",title:"Text / Vision Inputs"}))} onMouseLeave={()=>setHovered(null)}>Text / Vision Inputs</button><Arrow/><button onMouseEnter={()=>setHovered(cloneOp(vision[4],{id:"overview-fusion",kind:"linear",title:"Embedding Fusion"}))} onMouseLeave={()=>setHovered(null)}>Embedding Fusion <code>[B,S,6144]</code></button><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>Dense ×3</span><span><i className="sparse"/>MSA+MoE ×57</span></div><Arrow/><button onMouseEnter={()=>setHovered((layer<3?denseGraph(layer).add2:sparseGraph(layer).addout))} onMouseLeave={()=>setHovered(null)}>Final Norm → LM Head <code>[B,S,200064]</code></button></div>
    <LayerNavigator layer={layer} onChange={changeLayer}/>
    <section className="layer-canvas"><header><div><span>SELECTED LAYER DETAIL</span><h1>L{layer} · {layer<3?"Dense GQA + Dense MLP":"MiniMax Sparse Attention + MoE"}</h1></div><div className="node-legend"><span><i className="input-swatch"/>INPUT</span><span><i className="tensor-swatch"/>TENSOR</span><span><i className="operator-swatch"/>OPERATOR</span><span><i className="output-swatch"/>OUTPUT</span><code>{layerShard(layer)}</code></div></header>{layer<3?<DenseDiagram g={graph} active={active.id} onHover={setHovered} onLeave={()=>setHovered(null)} onSelect={setPinned}/>:<SparseDiagram g={graph} active={active.id} onHover={setHovered} onLeave={()=>setHovered(null)} onSelect={setPinned}/>}</section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab}/></div>{help&&<HelpModal onClose={()=>setHelp(false)}/>}</main>;
}
