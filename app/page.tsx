"use client";

import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import katex from "katex";
import { routeGraphEdge } from "./graph-routing";
import { nextDetailState, type DetailEvent, type DetailState } from "./detail-selection";
import { denseNodes, layerShard, sparseNodes, type Node, type Weight } from "./model-data";

type Tab = "io" | "formula" | "code";
type OpKind = "io" | "norm" | "linear" | "split" | "rope" | "matmul" | "scale" | "mask" | "softmax" | "activation" | "route" | "cache" | "add";
type BindingKind = "upstream" | "external" | "weight";
type IoBinding = { kind: BindingKind; label: string; shape: string; from: string; note?: string };
type CodeSection = { stage: string; title: string; location: string; code: string; url?: string };
type CodeSymbol = { symbol: string; resolvesTo: string; meaning: string };
type CodeDetail = { sections: CodeSection[]; symbols: CodeSymbol[] };
type OpNode = Node & { kind: OpKind; latex?: string; codeSections?: CodeSection[]; codeSymbols?: CodeSymbol[] };
type LayerType = "dense" | "sparse";
type ExpandedStage = "attention" | "ffn" | null;
type EdgePort = "top" | "top-left" | "top-right" | "right" | "bottom" | "left";
type GraphEdge = { from: string; to: string; fromPort?: EdgePort; toPort?: EdgePort; route?: "direct" | "side-left" | "side-right" };

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

const CONFIG_GROUPS = [
  {title:"顶层多模态配置",rows:[
    ["architectures","MiniMaxM3SparseForConditionalGeneration"],["auto_map.AutoConfig","configuration_minimax_m3_vl.MiniMaxM3VLConfig"],["model_type","minimax_m3_vl"],["torch_dtype","bfloat16"],["transformers_version","4.52.4"],["image_seq_length","576"],["image_token_index","200025"],["video_token_index","200026"],["multimodal_projector_bias","true"],["num_reward_heads","0"],["process_image_mode","dynamic_res"],["projector_hidden_act","gelu"],["projector_hidden_size","6144"],["vision_feature_layer","−1"],["vision_feature_select_strategy","full"],["image_grid_pinpoints","336…2016（步长 336）的 6×6 全组合"],
  ]},
  {title:"text_config",rows:[
    ["architectures","MiniMaxM3SparseForCausalLM"],["hidden_size","6144"],["intermediate_size","3072"],["dense_intermediate_size","12288"],["shared_intermediate_size","3072"],["num_hidden_layers","60"],["num_attention_heads","64"],["num_key_value_heads","4"],["head_dim","128"],["vocab_size","200064"],["max_position_embeddings","1048576"],["rms_norm_eps","1e−6"],["use_gemma_norm","true"],["attention_output_gate","false"],["rope_theta","5000000"],["rotary_dim","64"],["partial_rotary_factor","0.5"],["hidden_act","swigluoai"],["use_qk_norm","true"],["qk_norm_type","per_head"],["tie_word_embeddings","false"],["num_local_experts","128"],["num_experts_per_tok","4"],["n_shared_experts","1"],["scoring_func","sigmoid"],["use_routing_bias","true"],["moe_layer_freq","L0–2: 0 · L3–59: 1"],["num_mtp_modules","7"],["num_nextn_predict_layers","1"],["swiglu_alpha","1.702"],["swiglu_limit","7.0"],["routed_scaling_factor","2.0"],
  ]},
  {title:"text_config.sparse_attention_config",rows:[
    ["use_sparse_attention","true"],["sparse_index_dim","128"],["sparse_num_index_heads","4"],["sparse_topk_blocks","16"],["sparse_block_size","128"],["sparse_disable_index_value","L0–2: 0 · L3–59: 1"],["sparse_score_type","max"],["sparse_init_block","0"],["sparse_local_block","1"],["sparse_attention_freq","L0–2: 0 · L3–59: 1"],
  ]},
  {title:"vision_config",rows:[
    ["model_type","clip_vision_model"],["hidden_size","1280"],["num_attention_heads","16"],["num_hidden_layers","32"],["intermediate_size","5120"],["patch_size","14"],["image_size","2016"],["projection_dim","6144"],["position_embedding_type","rope"],["rope_mode","3d"],["rope_theta","10000.0"],["attention_dropout","0.0"],["hidden_act","gelu"],["initializer_factor","1.0"],["initializer_range","0.02"],["layer_norm_eps","1e−5"],["num_channels","3"],["vocab_size","32000"],["vision_segment_max_frames","4"],
  ]},
  {title:"图像 token 压缩（顶层与 vision_config 内相同）",rows:[
    ["image_token_compression_method","patch_merge"],["spatial_merge_size","2"],["temporal_patch_size","2"],
  ]},
] as const;

const SIMPLE_FORMULA: Partial<Record<OpKind,string>> = {
  norm:String.raw`y=\operatorname{Norm}(x)`,linear:String.raw`y=xW^{\mathsf T}`,split:String.raw`(a,b,\ldots)=\operatorname{Split}(x)`,rope:String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`,matmul:String.raw`y=a\,b^{\mathsf T}`,scale:String.raw`y=x/\sqrt{d_h}`,mask:String.raw`y=x+\mathrm{mask}`,softmax:String.raw`p=\operatorname{softmax}(x)`,activation:String.raw`y=g\,\sigma(1.702g)\,(u+1)`,route:String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`,cache:String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(K,V)`,add:String.raw`y=x+f(x)`,io:String.raw`y=x`,
};

const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"把每个 token 的向量缩放到稳定范围；shape 不变。",linear:"W 是当前模块绑定的权重；最后一维由 W 的输出维决定。",split:"只切分最后一维，不做数值计算，也没有权重。",rope:"position 决定旋转角度；这里只旋转每个 head 的前 64 维。",matmul:"沿共同的 head_dim 相乘并求和。",scale:"dₕ=128；缩放避免 score 随维度增大。",mask:"不可见位置加 −∞，softmax 后概率变为 0。",softmax:"把每行 score 转为和为 1 的概率。",activation:"g 是 gate，u 是 up；实际实现还包含 limit=7 的截断。",route:"只选择去哪里计算；Top-K 本身不生成 expert 输出。",cache:"slot 与 block table 由 runtime 提供，权重不参与。",add:"残差支路与计算支路逐元素相加，shape 必须一致。",io:"这是数据入口或运行时元数据，不执行可训练计算。",
};

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

const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
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
  "d-qk":[{kind:"upstream",label:"Qᵣ",shape:"[B,64,S,128]",from:"Partial RoPE (Q) 输出"},{kind:"upstream",label:"visible K",shape:"[B,4,T,128]",from:"Paged KV Cache 输出"}],
  "d-mask":[{kind:"upstream",label:"scaled scores",shape:"[B,64,S,T]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "d-pv":[{kind:"upstream",label:"attention probability P",shape:"[B,64,S,T]",from:"Softmax 输出"},{kind:"upstream",label:"visible V",shape:"[B,4,T,128]",from:"Paged KV Cache 输出"}],
  "s-rope":[{kind:"upstream",label:"Q̃ · K̃",shape:"Q/K unchanged",from:"Main Q/K Norm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "s-cache":[{kind:"upstream",label:"Kᵣ · V",shape:"KV pages",from:"Partial RoPE 与 Split 5 outputs"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "s-topk":[{kind:"upstream",label:"block scores",shape:"[B,4,S,Nblocks]",from:"Block Max 输出"},{kind:"external",label:"local / init priority",shape:"logical block flags",from:"Indexer 配置：local_blocks=1, init_blocks=0"}],
  "s-select":[{kind:"upstream",label:"logical block ids",shape:"[B,S,4,16]",from:"Top-16 Blocks 输出"},{kind:"upstream",label:"paged K · V",shape:"KV pages",from:"Paged KV Cache 输出"},{kind:"external",label:"block_table",shape:"[B,Nblocks]",from:"KV cache manager"}],
  "s-qk":[{kind:"upstream",label:"Qᵣ",shape:"[B,64,S,128]",from:"Partial RoPE 输出"},{kind:"upstream",label:"selected K",shape:"≤16 pages/group",from:"Select KV Pages 输出"}],
  "s-mask":[{kind:"upstream",label:"scaled selected scores",shape:"[B,64,S,Ksel]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "s-pv":[{kind:"upstream",label:"selected attention P",shape:"[B,64,S,Ksel]",from:"Softmax 输出"},{kind:"upstream",label:"selected V",shape:"≤16 pages/group",from:"Select KV Pages 输出"}],
  "s-router":[{kind:"upstream",label:"post-attn normalized hidden Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出"}],
  "s-experts":[{kind:"upstream",label:"normalized hidden + router logits",shape:"[B,S,6144] + [B,S,128]",from:"Post-attn RMSNorm 与 FP32 Router 输出"}],
  "s-shared":[{kind:"upstream",label:"all normalized tokens Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出；不经过 Top-K"}],
  "s-sum":[{kind:"upstream",label:"4 routed outputs",shape:"4 × [B,S,6144]",from:"Routed Experts ×4 输出"},{kind:"upstream",label:"shared output",shape:"[B,S,6144]",from:"Shared Expert ×1 输出"}],
};

const NEXT_BY_ID: Record<string,string> = {
  "d-input":"Gemma RMSNorm","d-position":"Partial RoPE (Q/K)","d-attnmeta":"Apply Causal / Pad Bounds","d-slots":"Paged KV Cache","d-norm":"QKV Projection","d-qkv":"Split Q / K / V","d-split":"Q RMSNorm · K RMSNorm · Paged KV Cache","d-qnorm":"Partial RoPE (Q)","d-knorm":"Partial RoPE (K)","d-ropeq":"Q × Kᵀ","d-ropek":"Paged KV Cache","d-cache":"Q × Kᵀ · P × V","d-qk":"Scale 1/√128","d-scale":"Apply Causal / Pad Bounds","d-mask":"Softmax","d-softmax":"P × V","d-pv":"O Projection","d-oproj":"Attention Residual","d-add1":"Post-attn Gemma RMSNorm","d-postnorm":"Gate + Up Projection","d-gateup":"SwiGLU-OAI","d-swiglu":"Down Projection","d-down":"MLP Residual","d-add2":"下一 decoder layer / Final Norm",
  "s-input":"Gemma RMSNorm","s-position":"Partial RoPE","s-attnmeta":"Indexer 与 Sparse Attention mask","s-slots":"Paged KV Cache","s-norm":"QKV + Index Projection","s-packed":"Split 5 outputs","s-split":"Index Q/K Norm · Main Q/K Norm · Paged KV Cache","s-idxnorm":"Index Q × Kᵀ","s-idxscore":"Block Max","s-blockmax":"Top-16 Blocks","s-topk":"Select KV Pages","s-mainnorm":"Partial RoPE","s-rope":"Paged KV Cache · Q × selected Kᵀ","s-cache":"Select KV Pages","s-select":"Q × selected Kᵀ · P × selected V","s-qk":"Scale 1/√128","s-scale":"Apply Causal / Pad Bounds","s-mask":"Softmax","s-softmax":"P × selected V","s-pv":"O Projection","s-oproj":"Attention Residual","s-addattn":"Post-attn Gemma RMSNorm","s-postnorm":"FP32 Router · Routed Experts · Shared Expert","s-router":"Routed Experts ×4","s-experts":"Weighted Sum","s-shared":"Weighted Sum","s-sum":"MoE Residual","s-addout":"下一 decoder layer / Final Norm",
};

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
    norm: cloneOp(norm,{id:"d-norm",kind:"norm",title:"Gemma RMSNorm"}),
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
    postnorm: cloneOp(norm,{id:"d-postnorm",kind:"norm",title:"Post-attn Gemma RMSNorm",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",weights:[postNorm]}),
    gateup: cloneOp(mlp,{id:"d-gateup",kind:"linear",title:"Gate + Up Projection",output:"gate · up",outputShape:"2 × [B,S,12288]",weights:mlp.weights.filter(w=>!w.key.includes("down_proj"))}),
    swiglu: cloneOp(mlp,{id:"d-swiglu",kind:"activation",title:"SwiGLU-OAI",input:"gate,up",inputShape:"2 × [B,S,12288]",output:"activated",outputShape:"[B,S,12288]",weights:[]}),
    down: cloneOp(mlp,{id:"d-down",kind:"linear",title:"Down Projection",input:"activated",inputShape:"[B,S,12288]",output:"Yffn",outputShape:"[B,S,6144]",weights:mlp.weights.filter(w=>w.key.includes("down_proj"))}),
    add2: cloneOp(mlp,{id:"d-add2",kind:"add",title:"+ MLP Residual",input:"Yffn + U",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+MLP(RMSNorm(U))",weights:[]}),
  };
}

function sparseGraph(layer: number): Record<string, OpNode> {
  const [packed,indexer,topk,attn,router,experts,shared,combine]=sparseNodes(layer);
  const normBase=denseNodes(layer)[0];
  const shard=layerShard(layer);
  const inputNorm: Weight={key:`language_model.model.layers.${layer}.input_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  const postNorm: Weight={key:`language_model.model.layers.${layer}.post_attention_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  return {
    input:cloneOp(packed,{id:"s-input",kind:"io",title:"Hidden states",input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[]}),
    position:cloneOp(attn,{id:"s-position",kind:"route",title:"Build Position IDs",kicker:"vLLM RUNTIME I/O",input:"num_computed_tokens + query offsets",inputShape:"[B] + [Nq]",output:"positions",outputShape:"[Nq]",formula:"position(req,i)=num_computed_tokens[req]+i",formulaNote:"positions 由 vLLM runner 在模型 forward 之前构造，再传给 MiniMax-M3 的 fused QKNorm + RoPE kernel。",source:"gpu_model_runner.py · _prepare_inputs",sourceUrl:RUNNER_URL,weights:[]}),
    attnmeta:cloneOp(attn,{id:"s-attnmeta",kind:"mask",title:"Build Attention Metadata",kicker:"vLLM RUNTIME I/O",input:"query_start_loc, seq_lens, causal=True",inputShape:"[B+1] · [B] · bool",output:"implicit causal / padding layout",outputShape:"backend metadata; 非稠密 [S,T]",formula:"valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)",formulaNote:"同一份边界元数据同时约束 indexer 的 block selection 和 main sparse attention。",source:"gpu_model_runner.py · CommonAttentionMetadata",sourceUrl:RUNNER_URL,weights:[]}),
    slots:cloneOp(attn,{id:"s-slots",kind:"route",title:"Resolve KV Slots",kicker:"vLLM RUNTIME I/O",input:"positions + block_table",inputShape:"[Nq] + [B,Nblocks]",output:"slot_mapping + block_table",outputShape:"[Nq] + [B,Nblocks]",formula:"slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)",formulaNote:"slot_mapping 用于 K/V 写入；block_table 把 indexer 选出的逻辑 block id 翻译为物理 page。",source:"gpu_model_runner.py · compute_slot_mapping",sourceUrl:RUNNER_URL,weights:[]}),
    norm:cloneOp(normBase,{id:"s-norm",kind:"norm",title:"Gemma RMSNorm",input:"Xₗ",inputShape:"[B,S,6144]",output:"X̂",outputShape:"[B,S,6144]",weights:[inputNorm]}),
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
    postnorm:cloneOp(normBase,{id:"s-postnorm",kind:"norm",title:"Post-attn Gemma RMSNorm",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",weights:[postNorm]}),
    router:cloneOp(router,{id:"s-router",kind:"route",title:"FP32 Router → Top-4",input:"Û",inputShape:"[B,S,6144]"}),
    experts:cloneOp(experts,{id:"s-experts",kind:"activation",title:"Routed Experts ×4",input:"Û + expert ids + weights",inputShape:"[B,S,6144] + 2×[B,S,4]"}),
    shared:cloneOp(shared,{id:"s-shared",kind:"activation",title:"Shared Expert ×1",input:"Û",inputShape:"[B,S,6144]"}),
    sum:cloneOp(combine,{id:"s-sum",kind:"add",title:"Weighted Sum",input:"4 routed + shared",inputShape:"5 × [B,S,6144]",output:"Ymoe",outputShape:"[B,S,6144]",weights:[]}),
    addout:cloneOp(combine,{id:"s-addout",kind:"add",title:"+ Decoder Residual"}),
  };
}

function GraphSurface({edges,className,children}:{edges:GraphEdge[];className:string;children:ReactNode}){
  const rootRef=useRef<HTMLDivElement>(null);
  const markerId=`graph-arrow-${useId().replace(/:/g,"")}`;
  const serializedEdges=JSON.stringify(edges);
  const edgeKey=edges.map(edge=>`${edge.from}:${edge.fromPort??"bottom"}>${edge.to}:${edge.toPort??"top"}:${edge.route??"direct"}`).join("|");
  const [paths,setPaths]=useState<string[]>([]);
  useLayoutEffect(()=>{
    const root=rootRef.current;
    if(!root)return;
    let frame=0;
    const point=(rect:DOMRect,port:EdgePort,rootRect:DOMRect)=>{
      const x=rect.left-rootRect.left; const y=rect.top-rootRect.top;
      if(port==="top")return [x+rect.width/2,y];
      if(port==="top-left")return [x+rect.width*.34,y];
      if(port==="top-right")return [x+rect.width*.66,y];
      if(port==="right")return [x+rect.width,y+rect.height/2];
      if(port==="left")return [x,y+rect.height/2];
      return [x+rect.width/2,y+rect.height];
    };
    const measure=()=>{
      const rootRect=root.getBoundingClientRect();
      const currentEdges=JSON.parse(serializedEdges) as GraphEdge[];
      const nodeRects=[...root.querySelectorAll<HTMLElement>("[data-graph-id]")].map(node=>node.getBoundingClientRect());
      const obstacleBounds={
        left:Math.min(...nodeRects.map(rect=>rect.left-rootRect.left)),
        right:Math.max(...nodeRects.map(rect=>rect.right-rootRect.left)),
      };
      const next=currentEdges.flatMap(edge=>{
        const source=root.querySelector<HTMLElement>(`[data-graph-id="${edge.from}"]`);
        const target=root.querySelector<HTMLElement>(`[data-graph-id="${edge.to}"]`);
        if(!source||!target)return [];
        const fromPort=edge.fromPort??"bottom"; const toPort=edge.toPort??"top";
        const [sx,sy]=point(source.getBoundingClientRect(),fromPort,rootRect);
        const [tx,ty]=point(target.getBoundingClientRect(),toPort,rootRect);
        const direction=edge.route??(fromPort==="right"||fromPort==="left"||toPort==="right"||toPort==="left"?"horizontal":"vertical");
        return [routeGraphEdge({source:{x:sx,y:sy},target:{x:tx,y:ty},direction,obstacleBounds,clearance:24}).path];
      });
      setPaths(next);
    };
    const observer=new ResizeObserver(()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(measure)});
    observer.observe(root);
    root.querySelectorAll<HTMLElement>("[data-graph-id]").forEach(node=>observer.observe(node));
    frame=requestAnimationFrame(measure);
    return()=>{cancelAnimationFrame(frame);observer.disconnect()};
  },[serializedEdges]);
  return <div ref={rootRef} className={`graph-surface ${className}`}>{children}<svg className="graph-connectors" aria-hidden="true"><defs><marker id={markerId} markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 8 4 L 0 8 Z"/></marker></defs>{paths.map((path,index)=><path key={`${edgeKey}-connector-${index}`} d={path} markerEnd={`url(#${markerId})`}/>)}</svg></div>;
}

function Op({node,active,onHover,onLeave,onSelect,graphId}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;graphId?:string}){
  return <button data-graph-id={graphId} className={`op-node op-${node.kind} ${active?"active":""}`} aria-pressed={active} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event=>{if(event.detail===0)onSelect(node)}}><small>OP · {node.kind}</small><b>{node.title}</b></button>;
}

type TensorRole = "input" | "tensor" | "output" | "side" | "weight";

function Tensor({name,shape,role="tensor",graphId}:{name:string;shape:string;role?:TensorRole;graphId?:string}){
  const label={input:"TENSOR",tensor:"TENSOR",output:"TENSOR",side:"EXTERNAL",weight:"WEIGHT"}[role];
  return <div data-graph-id={graphId} className={`tensor-node tensor-${role}`}><small>{label}</small><b>{name}</b><code>{shape}</code></div>;
}

const Arrow=({label}:{label?:string})=><span className="op-arrow"><i/>{label&&<small>{label}</small>}</span>;

function checkpointWeightName(weight?:Weight){
  return weight?.key.replace(/^language_model\.model\.layers\.\d+\./,"")??"weight";
}

function InputWeightedOp({node,active,onHover,onLeave,onSelect,inputName,inputShape,weightIndex=0,inputGraphId,graphId,weightGraphId,className=""}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;inputName:string;inputShape:string;weightIndex?:number;inputGraphId:string;graphId:string;weightGraphId:string;className?:string}){
  const weight=node.weights[weightIndex];
  const symbolicWeightShape=weight?.shape.replaceAll("6144","H")??"[H]";
  return <div className={`input-weighted-op ${className}`}><div className="co-input-row"><Tensor name={inputName} shape={inputShape} graphId={inputGraphId}/><Tensor name={checkpointWeightName(weight)} shape={symbolicWeightShape} role="weight" graphId={weightGraphId}/></div><Op node={node} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} graphId={graphId}/></div>;
}

function AddCircle({node,active,onHover,onLeave,onSelect,graphId}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;graphId?:string}){
  return <button data-graph-id={graphId} className={`add-circle ${active?"active":""}`} aria-label={node.title} aria-pressed={active} title={node.title} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event=>{if(event.detail===0)onSelect(node)}}>+</button>;
}

function RuntimeIORail({N}:{N:({id}:{id:string})=>ReactNode}){
  return <section className="runtime-io"><header><b>ATTENTION RUNTIME I/O</b><span>这些输入由 vLLM runner 生成并传入模型；mask 在内核中按边界隐式执行</span></header><div className="runtime-io-grid">
    <div className="io-lane"><Tensor name="num_computed_tokens · query offsets" shape="[B] + [Nq]" role="input"/><Arrow/><N id="position"/><Arrow/><Tensor name="positions → RoPE" shape="[Nq]"/></div>
    <div className="io-lane"><Tensor name="query_start_loc · seq_lens · causal" shape="[B+1] · [B] · True" role="input"/><Arrow/><N id="attnmeta"/><Arrow/><Tensor name="causal / padding layout → Attention" shape="implicit · 非稠密 [S,T]"/></div>
    <div className="io-lane"><Tensor name="positions · block_table" shape="[Nq] + [B,Nblocks]" role="input"/><Arrow/><N id="slots"/><Arrow/><Tensor name="slot_mapping · block_table → KV Cache" shape="[Nq] + [B,Nblocks]"/></div>
  </div></section>;
}

/* eslint-disable react-hooks/static-components -- local alias only shortens a large, stateless operator graph */
// Legacy full graph kept as a source-level reference while progressive disclosure is active.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

/* eslint-disable react-hooks/static-components -- local N aliases keep the dependency diagrams legible */
function StageZoom({type,stage,g,active,onHover,onLeave,onSelect,onClose}:{type:LayerType;stage:Exclude<ExpandedStage,null>;g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;onClose:()=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const N=({id,graphId}:{id:string;graphId?:string})=><Op node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const IW=({id,inputName,inputShape,weightIndex,inputGraphId,graphId,weightGraphId,className}:{id:string;inputName:string;inputShape:string;weightIndex?:number;inputGraphId:string;graphId:string;weightGraphId:string;className?:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} weightIndex={weightIndex} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId} className={className}/>;
  if(stage==="ffn"&&type==="dense"){
    const edges:GraphEdge[]=[
      {from:"mlp-u",to:"mlp-post",toPort:"top-left"},{from:"mlp-wpost",to:"mlp-post",toPort:"top-right"},{from:"mlp-post",to:"mlp-uhat"},{from:"mlp-uhat",to:"mlp-gateup"},{from:"mlp-wgate",to:"mlp-gateup",fromPort:"left",toPort:"right"},{from:"mlp-wup",to:"mlp-gateup",fromPort:"left",toPort:"right"},{from:"mlp-gateup",to:"mlp-gate"},{from:"mlp-gateup",to:"mlp-up"},{from:"mlp-gate",to:"mlp-gate-act"},{from:"mlp-up",to:"mlp-up-act"},{from:"mlp-gate-act",to:"mlp-mul"},{from:"mlp-up-act",to:"mlp-mul"},{from:"mlp-mul",to:"mlp-activated"},{from:"mlp-activated",to:"mlp-down"},{from:"mlp-wdown",to:"mlp-down",fromPort:"left",toPort:"right"},{from:"mlp-down",to:"mlp-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><div><span>GQA + MLP · L0–2</span><b>SwiGLU Graph：每条边都连接具体张量、权重与算子</b></div><button onClick={onClose}>收起 ×</button></header><div className="lesson-layout"><GraphSurface className="mlp-node-graph" edges={edges}>
      <IW id="postnorm" inputName="U" inputShape="[B,S,H]" inputGraphId="mlp-u" graphId="mlp-post" weightGraphId="mlp-wpost" className="mlp-postnorm-unit"/><Tensor name="Û" shape="[B,S,H]" graphId="mlp-uhat"/><N id="gateup" graphId="mlp-gateup"/><Tensor name="mlp.gate_proj.weight" shape="[H_dense,H]" role="weight" graphId="mlp-wgate"/><Tensor name="mlp.up_proj.weight" shape="[H_dense,H]" role="weight" graphId="mlp-wup"/><Tensor name="gate" shape="[B,S,H_dense]" graphId="mlp-gate"/><Tensor name="up" shape="[B,S,H_dense]" graphId="mlp-up"/><div className="mini-math" data-graph-id="mlp-gate-act">clamp → SiLU(1.702·gate)</div><div className="mini-math" data-graph-id="mlp-up-act">clamp → up + 1</div><button className="multiply-circle" data-graph-id="mlp-mul" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}>×</button><Tensor name="activated" shape="[B,S,H_dense]" graphId="mlp-activated"/><N id="down" graphId="mlp-down"/><Tensor name="mlp.down_proj.weight" shape="[H,H_dense]" role="weight" graphId="mlp-wdown"/><Tensor name="Yffn" shape="[B,S,H]" graphId="mlp-y"/>
    </GraphSurface><aside className="lesson-notes"><span>作用</span><h3>逐 token 扩维、门控，再投回 hidden size</h3><p>MLP 不混合 token。Gate 与 Up 从同一个 fused projection 并行分叉，随后在 × 节点汇合。</p><span>简化公式</span><code>MLP(x)=Wdown[SiLU(gate) ⊙ (up+1)]</code><p>每条箭头都从上游张量或权重的边界出发，并进入实际消费它的算子。</p></aside></div></section>;
  }
  if(stage==="ffn"){
    const edges:GraphEdge[]=[
      {from:"moe-u",to:"moe-router"},{from:"moe-wrouter",to:"moe-router",fromPort:"right",toPort:"left"},{from:"moe-router",to:"moe-ids"},{from:"moe-router",to:"moe-rweights"},{from:"moe-u",to:"moe-experts",route:"side-left",fromPort:"left",toPort:"left"},{from:"moe-ids",to:"moe-experts"},{from:"moe-rweights",to:"moe-experts"},{from:"moe-wexperts",to:"moe-experts",fromPort:"right",toPort:"left"},{from:"moe-experts",to:"moe-routed"},{from:"moe-u",to:"moe-shared",fromPort:"bottom",toPort:"top"},{from:"moe-wshared",to:"moe-shared",fromPort:"left",toPort:"right"},{from:"moe-shared",to:"moe-shared-out"},{from:"moe-routed",to:"moe-sum"},{from:"moe-shared-out",to:"moe-sum"},{from:"moe-sum",to:"moe-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><div><span>TOP-4 MOE · L3–59</span><b>同一个 Û 同时进入 Router、Routed Experts 与 Shared Expert</b></div><button onClick={onClose}>收起 ×</button></header><GraphSurface className="moe-node-graph" edges={edges}><Tensor name="Û" shape="[B,S,H]" graphId="moe-u"/><Tensor name="block_sparse_moe.gate.weight · e_score_correction_bias" shape="[E,H] · [E]" role="weight" graphId="moe-wrouter"/><N id="router" graphId="moe-router"/><Tensor name="expert ids" shape="[B,S,K]" graphId="moe-ids"/><Tensor name="router weights" shape="[B,S,K]" graphId="moe-rweights"/><Tensor name="block_sparse_moe.experts.*.{w1,w3,w2}.weight" shape="E × expert weights" role="weight" graphId="moe-wexperts"/><N id="experts" graphId="moe-experts"/><Tensor name="weighted routed output" shape="[B,S,H]" graphId="moe-routed"/><N id="shared" graphId="moe-shared"/><Tensor name="block_sparse_moe.shared_experts.{gate_proj,up_proj,down_proj}.weight" shape="shared MLP weights" role="weight" graphId="moe-wshared"/><Tensor name="shared output" shape="[B,S,H]" graphId="moe-shared-out"/><N id="sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[B,S,H]" graphId="moe-y"/></GraphSurface></section>;
  }
  const dense=type==="dense";
  const ids=dense?{project:"qkv",split:"split",qnorm:"qnorm",knorm:"knorm",ropeq:"ropeq",ropek:"ropek"}:{project:"packed",split:"split",qnorm:"mainnorm",knorm:"mainnorm",ropeq:"rope",ropek:"rope"};
  const keyId=dense?"attn-paged-k":"attn-selected-k"; const valueId=dense?"attn-paged-v":"attn-selected-v";
  const edges:GraphEdge[]=[
    {from:"attn-x",to:"attn-project",fromPort:"right",toPort:"left"},{from:"attn-project",to:"attn-packed",fromPort:"right",toPort:"left"},{from:"attn-packed",to:"attn-split",fromPort:"right",toPort:"left"},{from:"attn-split",to:"attn-q"},{from:"attn-split",to:"attn-k"},{from:"attn-split",to:"attn-v"},{from:"attn-q",to:"attn-qnorm",toPort:"top-left"},{from:"attn-wq",to:"attn-qnorm",toPort:"top-right"},{from:"attn-qnorm",to:"attn-qt"},{from:"attn-qt",to:"attn-qrope"},{from:"attn-posq",to:"attn-qrope",fromPort:"left",toPort:"right"},{from:"attn-qrope",to:"attn-qr"},{from:"attn-k",to:"attn-knorm",toPort:"top-left"},{from:"attn-wk",to:"attn-knorm",toPort:"top-right"},{from:"attn-knorm",to:"attn-kt"},{from:"attn-kt",to:"attn-krope"},{from:"attn-posk",to:"attn-krope",fromPort:"left",toPort:"right"},{from:"attn-krope",to:"attn-kr"},{from:"attn-kr",to:"attn-cache"},{from:"attn-v",to:"attn-cache"},{from:"attn-cache-meta",to:"attn-cache",fromPort:"left",toPort:"right"},{from:"attn-cache",to:"attn-paged-k"},{from:"attn-cache",to:"attn-paged-v"},{from:"attn-qr",to:"attn-qk"},{from:keyId,to:"attn-qk"},{from:"attn-qk",to:"attn-scale",fromPort:"right",toPort:"left"},{from:"attn-scale",to:"attn-scaled",fromPort:"right",toPort:"left"},{from:"attn-scaled",to:"attn-mask",fromPort:"right",toPort:"left"},{from:"attn-bounds",to:"attn-mask"},{from:"attn-mask",to:"attn-softmax",fromPort:"right",toPort:"left"},{from:"attn-softmax",to:"attn-p",fromPort:"right",toPort:"left"},{from:"attn-p",to:"attn-pv",fromPort:"right",toPort:"left"},{from:valueId,to:"attn-pv"},{from:"attn-pv",to:"attn-heads",fromPort:"right",toPort:"left"},{from:"attn-heads",to:"attn-oproj",fromPort:"right",toPort:"left"},{from:"attn-oproj",to:"attn-y",fromPort:"right",toPort:"left"},
    ...(!dense?[{from:"attn-split",to:"attn-qidx"},{from:"attn-split",to:"attn-kidx"},{from:"attn-qidx",to:"attn-idxnorm"},{from:"attn-kidx",to:"attn-idxnorm"},{from:"attn-idxnorm",to:"attn-idxscore",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-idxbounds",to:"attn-idxscore"},{from:"attn-idxscore",to:"attn-blockmax",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-blockmax",to:"attn-topk",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-topk",to:"attn-topids",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-topids",to:"attn-select"},{from:"attn-paged-k",to:"attn-select"},{from:"attn-paged-v",to:"attn-select"},{from:"attn-select",to:"attn-selected-k",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-select",to:"attn-selected-v",fromPort:"right" as EdgePort,toPort:"left" as EdgePort}] : []),
  ];
  return <section className="stage-zoom lesson-zoom attention-lesson"><header><div><span>{dense?"GQA + ROPE · L0–2":"MSA · L3–59"}</span><b>{dense?"Q / K / V 从 Split 节点分叉，再在 Attention 算子汇合":"Indexer 与主 Attention 通过明确的 KV page 边连接"}</b></div><button onClick={onClose}>收起 ×</button></header><div className="lesson-layout"><GraphSurface className="attention-flowchart connected-attention-graph" edges={edges}>
    <div className="compact-chain"><Tensor name="X̂" shape="[B,S,H]" graphId="attn-x"/><N id={ids.project} graphId="attn-project"/><Tensor name="packed" shape={dense?"[B,S,9216]":"[B,S,9856]"} graphId="attn-packed"/><N id={ids.split} graphId="attn-split"/></div>
    {!dense&&<div className="index-ribbon"><div className="multi-source"><Tensor name="Qidx" shape="[B,S,4,128]" graphId="attn-qidx"/><Tensor name="Kidx" shape="[B,T,1,128]" graphId="attn-kidx"/></div><N id="idxnorm" graphId="attn-idxnorm"/><N id="idxscore" graphId="attn-idxscore"/><Tensor name="causal bounds" shape="runtime" role="side" graphId="attn-idxbounds"/><N id="blockmax" graphId="attn-blockmax"/><N id="topk" graphId="attn-topk"/><Tensor name="Top-16 block ids" shape="[B,S,4,16]" graphId="attn-topids"/></div>}
    <div className="qkv-lanes"><section><header>Q PATH</header><IW id={ids.qnorm} inputName="Q" inputShape="[B,Nₕ,S,Dₕ]" inputGraphId="attn-q" graphId="attn-qnorm" weightGraphId="attn-wq"/><div className="two-source"><Tensor name="Q̃" shape="same" graphId="attn-qt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posq"/></div><N id={ids.ropeq} graphId="attn-qrope"/><Tensor name="Qᵣ" shape="[B,Nₕ,S,Dₕ]" graphId="attn-qr"/></section><section><header>K PATH</header><IW id={ids.knorm} inputName="K" inputShape="[B,Nₖᵥ,S,Dₕ]" weightIndex={dense?0:1} inputGraphId="attn-k" graphId="attn-knorm" weightGraphId="attn-wk"/><div className="two-source"><Tensor name="K̃" shape="same" graphId="attn-kt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posk"/></div><N id={ids.ropek} graphId="attn-krope"/><Tensor name="Kᵣ" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-kr"/></section><section><header>V + KV CACHE</header><Tensor name="V" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-v"/><Tensor name="slot_mapping · block_table" shape="runtime" role="side" graphId="attn-cache-meta"/><N id="cache" graphId="attn-cache"/><div className="two-source"><Tensor name="paged K" shape="KV pages" graphId="attn-paged-k"/><Tensor name="paged V" shape="KV pages" graphId="attn-paged-v"/></div></section></div>
    {!dense&&<div className="selection-chain"><N id="select" graphId="attn-select"/><div className="two-source"><Tensor name="selected K" shape="≤2048 tokens/group" graphId="attn-selected-k"/><Tensor name="selected V" shape="≤2048 tokens/group" graphId="attn-selected-v"/></div></div>}
    <div className="score-pipeline"><N id="qk" graphId="attn-qk"/><N id="scale" graphId="attn-scale"/><Tensor name="scaled scores" shape="[B,Nₕ,S,T]" graphId="attn-scaled"/><Tensor name="causal / pad bounds" shape="runtime metadata" role="side" graphId="attn-bounds"/><N id="mask" graphId="attn-mask"/><N id="softmax" graphId="attn-softmax"/><Tensor name="P" shape="[B,Nₕ,S,T]" graphId="attn-p"/></div>
    <div className="context-pipeline"><N id="pv" graphId="attn-pv"/><Tensor name="heads" shape="[B,S,Nₕ·Dₕ]" graphId="attn-heads"/><N id="oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[B,S,H]" graphId="attn-y"/></div>
  </GraphSurface><aside className="lesson-notes"><span>作用</span><h3>用 Q 找到相关 K，再按概率汇总 V</h3><p>图中的每条边都绑定源节点和目标节点；分叉来自 Split，汇合进入 MatMul、Mask 或 Cache 算子。</p><span>简化公式</span><code>Attention(Q,K,V)=softmax(QKᵀ/√Dₕ + mask)V</code><p>{dense?"Dense 层读取完整可见 KV 历史。":"Sparse 层的 Top-16 只缩小候选 KV blocks；causal / padding mask 仍在最终 attention 中执行。"}</p></aside></div></section>;
}

function DecoderDiagram({type,g,active,expanded,onExpand,onHover,onLeave,onSelect}:{type:LayerType;g:Record<string,OpNode>;active:string;expanded:ExpandedStage;onExpand:(stage:ExpandedStage)=>void;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const IW=({id,inputName,inputShape,inputGraphId,graphId,weightGraphId}:{id:string;inputName:string;inputShape:string;inputGraphId:string;graphId:string;weightGraphId:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId}/>;
  const A=({id,graphId}:{id:string;graphId:string})=><AddCircle node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const edges:GraphEdge[]=[{from:"main-x",to:"main-norm",toPort:"top-left"},{from:"main-win",to:"main-norm",toPort:"top-right"},{from:"main-norm",to:"main-attn"},{from:"main-attn",to:"main-add1"},{from:"main-x",to:"main-add1",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add1",to:"main-u"},{from:"main-u",to:"main-post",toPort:"top-left"},{from:"main-wpost",to:"main-post",toPort:"top-right"},{from:"main-post",to:"main-ffn"},{from:"main-ffn",to:"main-add2"},{from:"main-u",to:"main-add2",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add2",to:"main-out"}];
  return <div className={`decoder-workbench ${expanded?"has-zoom":""}`}><GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <IW id="norm" inputName="Xₗ · hidden_states" inputShape="[B,S,H]" inputGraphId="main-x" graphId="main-norm" weightGraphId="main-win"/><button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={()=>onExpand(expanded==="attention"?null:"attention")}><small>点击展开</small><b>{type==="dense"?"GQA + RoPE":"MSA"}</b><span>{type==="dense"?"Q/K/V · causal mask · KV cache":"Top-16 blocks · causal mask · KV cache"}</span></button><A id={type==="dense"?"add1":"addattn"} graphId="main-add1"/><IW id="postnorm" inputName="U" inputShape="[B,S,H]" inputGraphId="main-u" graphId="main-post" weightGraphId="main-wpost"/><button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={()=>onExpand(expanded==="ffn"?null:"ffn")}><small>点击展开</small><b>{type==="dense"?"SwiGLU MLP":"Top-4 MoE + Shared Expert"}</b><span>{type==="dense"?"Gate / Up 并行 → ⊙ → Down":"Routed 与 Shared 两路并行"}</span></button><A id={type==="dense"?"add2":"addout"} graphId="main-add2"/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,H]" graphId="main-out"/>
  </GraphSurface>{expanded&&<StageZoom type={type} stage={expanded} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={()=>onExpand(null)}/>}</div>;
}
/* eslint-enable react-hooks/static-components */

function LayerNavigator({type,onChange}:{type:LayerType;onChange:(type:LayerType)=>void}){
  return <div className="layer-nav layer-type-nav"><div className="layer-nav-head"><b>{type==="dense"?"GQA + MLP":"MSA + MoE"}</b></div><div className="layer-type-options"><button className={type==="dense"?"active dense":"dense"} onClick={()=>onChange("dense")}><span>L0–L2</span><b>GQA + MLP</b><small>3 层共享同一实现</small></button><button className={type==="sparse"?"active sparse":"sparse"} onClick={()=>onChange("sparse")}><span>L3–L59</span><b>MSA + Top-4 MoE</b><small>57 层共享同一实现</small></button></div></div>;
}

function LatexFormula({node}:{node:OpNode}){
  const formula=SIMPLE_FORMULA[node.kind]??String.raw`y=f(x)`;
  const html=katex.renderToString(formula,{displayMode:true,throwOnError:false,strict:"ignore",output:"htmlAndMathml"});
  return <div className="latex-render" aria-label={`${node.title} 简化公式`} dangerouslySetInnerHTML={{__html:html}}/>;
}

function symbolicShape(shape:string){
  return shape.replaceAll("[B,64,S,128]","[B,Nₕ,S,Dₕ]").replaceAll("[B,64,S,T]","[B,Nₕ,S,T]").replaceAll("[B,4,S,128]","[B,Nₖᵥ,S,Dₕ]").replaceAll("[B,4,T,128]","[B,Nₖᵥ,T,Dₕ]").replaceAll("[B,S,12288]","[B,S,H_dense]").replaceAll("[B,S,6144]","[B,S,H]").replaceAll("[B,S,8192]","[B,S,Nₕ·Dₕ]").replaceAll("[B,S,9216]","[B,S,(Nₕ+2Nₖᵥ)·Dₕ]").replaceAll("[B,S,9856]","[B,S,QKV+Index]").replaceAll("200064","V");
}

function localShape(shape:string,node:OpNode,binding?:IoBinding){
  if(binding?.kind==="weight"){
    if(binding.label.includes("experts."))return `${shape} · 每个 EP rank 约持有 E/EP 个 experts`;
    if(binding.label.includes("q_proj")||binding.label.includes("gate_proj")||binding.label.includes("up_proj"))return `${shape} · 输出维按 TP 切分`;
    if(binding.label.includes("o_proj")||binding.label.includes("down_proj"))return `${shape} · 输入维按 TP 切分`;
  }
  if(/Q|heads|q_proj|8192|64/.test(`${binding?.label??""} ${shape}`))return `${shape} · 本 rank 使用 Nₕ/TP 个 Q heads`;
  if(/K|V|kv|512/.test(`${binding?.label??""} ${shape}`))return `${shape} · 逻辑上 Nₖᵥ/TP；TP>Nₖᵥ 时可复制 KV heads`;
  if(node.id==="s-experts"||node.id==="s-router")return `${shape} · E/EP 个专家由本 EP rank 持有`;
  return `${shape} · TP/EP 不改变逻辑全局 shape`;
}

function ShapeRows({shape,node,binding}:{shape:string;node:OpNode;binding?:IoBinding}){
  return <div className="shape-rows"><span><i>符号</i><code>{symbolicShape(shape)}</code></span><span><i>全局实际</i><code>{shape}</code></span><span><i>并行局部</i><code>{localShape(shape,node,binding)}</code></span></div>;
}

function bindingsFor(node:OpNode):IoBinding[]{
  const dataInputs=INPUT_OVERRIDES[node.id]??[{kind:node.kind==="io"?"external":"upstream",label:node.input,shape:node.inputShape,from:node.kind==="io"?"模型调用方 / runtime":"图中紧邻的上游模块输出"}];
  const weightInputs=node.weights.map(weight=>({kind:"weight" as const,label:weight.key,shape:`${weight.dtype} · ${weight.shape}`,from:weight.runtime?`checkpoint → ${weight.runtime}`:`checkpoint · ${weight.shard}`,note:weight.params?`${weight.params} parameters`:undefined}));
  return [...dataInputs,...weightInputs];
}

function IoView({node}:{node:OpNode}){
  const bindings=bindingsFor(node);
  const labels:Record<BindingKind,string>={upstream:"上游张量",external:"外部输入",weight:"权重输入"};
  return <div className="io-binding-view"><section className="binding-list"><header><span>INPUT BINDINGS</span><b>{bindings.length} 路输入</b></header>{bindings.map((binding,index)=><article className={`binding binding-${binding.kind}`} key={`${binding.kind}-${binding.label}-${index}`}><div><span>{labels[binding.kind]}</span></div><b>{binding.label}</b><ShapeRows shape={binding.shape} node={node} binding={binding}/><p><i>来自</i>{binding.from}</p>{binding.note&&<small>{binding.note}</small>}</article>)}</section><div className={`io-operator op-${node.kind}`}><span>CURRENT OPERATOR</span><b>{node.title}</b><code>{node.runtime}</code></div><section className="output-binding"><header><span>OUTPUT BINDING</span><b>1 路产物</b></header><article><div><span>计算产物</span></div><b>{node.output}</b><ShapeRows shape={node.outputShape} node={node}/><p><i>送往</i>{NEXT_BY_ID[node.id]??"图中下游模块"}</p></article></section></div>;
}

function CodeView({node}:{node:OpNode}){
  const sections=(node.codeSections??[]).filter(section=>/forward|CALL|ENTER|PROJECT|ATTEND|ROUTE|SHARED/.test(`${section.title} ${section.stage}`));
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {VLLM_COMMIT.slice(0,7)}</span><b>{node.source}</b><i>↗</i></a>
    {sections.length?<section className="code-call-chain"><header><span>FORWARD ONLY</span><b>仅保留 forward / forward_native</b></header>{sections.map((section,index)=><article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><span>{section.stage}</span><b>{section.title}</b><small>{section.location}</small></div>{section.url&&<a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article>)}</section>:<div className="code-empty"><b>此节点没有独立 forward</b><p>它由所在模块的 forward 调度，或只是一个数学拆解步骤。</p></div>}
  </div>;
}

function DetailPanel({node,tab,setTab,pinned,onClear}:{node:OpNode|null;tab:Tab;setTab:(t:Tab)=>void;pinned:boolean;onClear:()=>void}){
  const tabs:[Tab,string][]=[["io","I/O + 权重"],["formula","公式"],["code","代码"]];
  if(!node)return <aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后，可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div><i className={`kind-dot op-${node.kind}`}/>{pinned&&<button className="unpin-button" onClick={onClear}>已固定 · 取消</button>}<p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div className={`detail-content detail-${tab}`}>
    {tab==="io"&&<IoView node={node}/>}
    {tab==="formula"&&<div className="formula-view"><span>作用</span><div className="formula-purpose">{node.summary}</div><span>简化 LATEX</span><LatexFormula node={node}/><div className="formula-implementation"><b>一句话解释</b><p>{FORMULA_NOTE[node.kind]??node.formulaNote}</p></div><div className="formula-terms"><span><b>x / a / b</b>输入张量</span><span><b>y / p</b>输出张量</span><span><b>W</b>权重矩阵</span><span><b>dₕ</b>head_dim = 128</span></div></div>}
    {tab==="code"&&<CodeView node={node}/>}
    </div><footer>vLLM @ {VLLM_COMMIT.slice(0,7)} · official safetensors</footer></aside>;
}

function HelpModal({onClose}:{onClose:()=>void}){
  const [section,setSection]=useState<"shape"|"config">("shape");
  const symbols=[["B","batch size"],["S","本轮 query 长度"],["T","含历史 cache 的 KV 长度"],["Nq","本轮全部 query tokens"],["H","hidden_size = 6144"],["Nₕ","query heads = 64"],["Nₖᵥ","KV heads = 4"],["Dₕ","head_dim = 128"],["Dᵣ","rotary_dim = 64"],["H_dense","dense FFN = 12288"],["H_expert","expert FFN = 3072"],["E","routed experts = 128"],["K","experts per token = 4"],["N_idx","index heads = 4"],["D_idx","index dim = 128"],["K_block","selected blocks = 16"],["TP","tensor parallel size"],["EP","expert parallel size"],["V","vocab_size = 200064"]];
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal reference-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="模型参数与 Shape Reference"><header><div><span>REFERENCE</span><h2>模型参数与 Shape Reference</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><nav className="reference-tabs"><button className={section==="shape"?"active":""} onClick={()=>setSection("shape")}>Shape · TP / EP</button><button className={section==="config"?"active":""} onClick={()=>setSection("config")}>完整 config.json</button></nav>{section==="shape"?<div className="reference-shape"><section><h3>Shape 符号</h3><div className="symbol-grid">{symbols.map(([symbol,meaning])=><div key={symbol}><b>{symbol}</b><span>{meaning}</span></div>)}</div></section><section><h3>全局 shape 与单 rank shape</h3><div className="parallel-examples"><div><code>Q: [B,Nₕ,S,Dₕ]</code><span>→ TP rank: [B,Nₕ/TP,S,Dₕ]</span></div><div><code>K/V: [B,Nₖᵥ,T,Dₕ]</code><span>→ Nₖᵥ/TP；TP 较大时可复制 KV heads</span></div><div><code>Routed experts: E=128</code><span>→ 每个 EP rank 约持有 E/EP 个 experts</span></div><div><code>Dense / shared MLP</code><span>→ 中间维按 TP 切分；不按 EP 路由</span></div></div></section><section><h3>权重名称为什么与代码不同？</h3><p className="reference-copy">权重文件保存的是训练时参数名；vLLM 为减少 kernel 次数，会把多个权重装载到一个运行时模块。它们数值一一对应，只是存储名称与执行模块名称不同。</p><div className="mapping-table"><div><code>q_proj · k_proj · v_proj</code><span>装载为</span><b>qkv_proj</b></div><div><code>gate_proj · up_proj</code><span>装载为</span><b>gate_up_proj</b></div><div><code>experts.w1 · w3</code><span>装载为</span><b>FusedMoE w13</b></div></div></section></div>:<div className="config-reference">{CONFIG_GROUPS.map(group=><section key={group.title}><h3>{group.title}</h3><table><tbody>{group.rows.map(([key,value])=><tr key={key}><th>{key}</th><td>{value}</td></tr>)}</tbody></table></section>)}</div>}</section></div>;
}

export default function Home(){
  const [layerType,setLayerType]=useState<LayerType>("sparse"); const [expanded,setExpanded]=useState<ExpandedStage>(null); const [tab,setTab]=useState<Tab>("io"); const [dark,setDark]=useState(false); const [help,setHelp]=useState(false);
  const layer=layerType==="dense"?2:3; const graph=layerType==="dense"?denseGraph(layer):sparseGraph(layer); const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null}); const active=detail.pinned??detail.hovered;
  const updateDetail=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
  const changeLayerType=(next:LayerType)=>{setLayerType(next);setExpanded(null);updateDetail({type:"clear"})};
  return <main className={`atlas-app ${dark?"dark":""}`}><header className="app-header">
    <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>模型结构概览</b></div></div>
    <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value="minimax-m3" onChange={()=>undefined}>{MODEL_REGISTRY.map(m=><option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}</option>)}</select></label>
    <nav className="resource-links"><a href={CODE_URL} target="_blank" rel="noreferrer"><b>CODE ↗</b><small>vLLM @ {VLLM_COMMIT.slice(0,7)}</small></a><a href={WEIGHTS_URL} target="_blank" rel="noreferrer"><b>WEIGHTS ↗</b><small>Hugging Face · 59 shards</small></a></nav>
    <div className="model-facts"><span><b>428B</b><small>模型总参数量</small></span><span><b>23B</b><small>每 token 激活参数</small></span><span><b>1M</b><small>最大上下文 token</small></span><span><b>869 GB</b><small>BF16 checkpoint</small></span></div>
    <button className="help-button" onClick={()=>setHelp(true)} aria-label="查看参数和符号说明">?</button><button className="theme-button" onClick={()=>setDark(v=>!v)} aria-label="切换明暗主题">{dark?"☀":"☾"}</button>
  </header><div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><div className="model-step">Text / Vision Inputs</div><Arrow/><div className="model-step">Embedding Fusion <code>[B,S,H]</code></div><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>L0–2 · GQA+MLP</span><span><i className="sparse"/>L3–59 · MSA+MoE</span></div><Arrow/><div className="model-step">Final Gemma RMSNorm <code>[B,S,H]</code></div><Arrow/><div className="model-step">LM Head <code>[B,S,V]</code></div></div>
    <LayerNavigator type={layerType} onChange={changeLayerType}/>
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按结构类型展示</span><h1>{layerType==="dense"?"GQA + MLP · L0–L2 同构":"MSA + Top-4 MoE · L3–L59 同构"}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span><i className="operator-swatch"/>OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><DecoderDiagram type={layerType} g={graph} active={active?.id??""} expanded={expanded} onExpand={setExpanded} onHover={node=>updateDetail({type:"hover",node})} onLeave={()=>updateDetail({type:"leave"})} onSelect={node=>{updateDetail({type:"pin",node});setTab("io")}}/></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>updateDetail({type:"clear"})}/></div>{help&&<HelpModal onClose={()=>setHelp(false)}/>}</main>;
}
