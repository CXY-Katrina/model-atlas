import { useId, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import katex from "katex";
import { routeGraphEdge, routeGraphFanout } from "./graph-routing";
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
type EdgePort = "top" | "top-left" | "top-right" | "right" | "bottom" | "bottom-left" | "bottom-right" | "left";
type GraphEdge = { from: string; to: string; fromPort?: EdgePort; toPort?: EdgePort; route?: "side-left" | "side-right" | "bus-left" | "bus-right"; approach?: number; departure?: number; fanout?: string };
type EdgeTone = "data" | "weight" | "external" | "residual";
type GraphPath = { d: string; tone: EdgeTone; marker?: boolean };

const VLLM_COMMIT = "edd4c8176cfd98ece8a29beda574378c42971967";
const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3";
const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
const ACTIVATION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/activation.py`;
const LINEAR_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/linear.py`;
const TRANSFORMERS_MINIMAX_M3_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L365";
const TRANSFORMERS_MOE_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L202-L239";
const TRANSFORMERS_QKV_PROJECTION_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L422-L445";
const TRANSFORMERS_INDEX_PROJECTION_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L538-L563";
const TRANSFORMERS_SPARSE_ATTENTION_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L447-L489";
const TRANSFORMERS_INDEX_SELECTION_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L552-L597";
const TRANSFORMERS_BLOCK_MASK_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L599-L635";
const VLLM_INDEXER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/common/indexer.py";
const VLLM_SPARSE_ATTENTION_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/common/sparse_attention.py";
const NORM_FORWARD_URL = `${CODE_URL}#L130-L142`;
const DECODER_FORWARD_URL = `${CODE_URL}#L752-L778`;
const FLASHINFER_GEMMA_NORM_URL = "https://docs.flashinfer.ai/generated/flashinfer.norm.gemma_rmsnorm.html";

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
    ["architectures","MiniMaxM3SparseForCausalLM"],["hidden_size","6144"],["intermediate_size","3072"],["dense_intermediate_size","12288"],["shared_intermediate_size","3072"],["num_hidden_layers","60"],["num_attention_heads","64"],["num_key_value_heads","4"],["head_dim","128"],["vocab_size","200064"],["max_position_embeddings","1048576"],["rms_norm_eps","1e−6"],["use_gemma_norm","true"],["attention_output_gate","false"],["rope_theta","5000000"],["rotary_dim","64"],["partial_rotary_factor","0.5"],["hidden_act","swigluoai"],["use_qk_norm","true"],["qk_norm_type","per_head"],["tie_word_embeddings","false"],["num_local_experts","128"],["num_experts_per_tok","4"],["n_shared_experts","1"],["scoring_func","sigmoid"],["use_routing_bias","true"],["moe_layer_freq","L0–2: 0 · L3–59: 1"],["num_mtp_modules","7"],["num_nextn_predict_layers","1"],["swiglu_alpha","1.702"],["swiglu_beta","1.0"],["swiglu_limit","7.0"],["routed_scaling_factor","2.0"],
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

const CONFIG_SYMBOLS: Record<string,string> = {
  "顶层多模态配置:image_seq_length":"S_img",
  "顶层多模态配置:image_token_index":"t_img",
  "顶层多模态配置:video_token_index":"t_video",
  "顶层多模态配置:multimodal_projector_bias":"b_proj",
  "顶层多模态配置:num_reward_heads":"N_reward",
  "顶层多模态配置:projector_hidden_act":"φ_proj",
  "顶层多模态配置:projector_hidden_size":"H",
  "顶层多模态配置:vision_feature_layer":"L_feature",
  "text_config:hidden_size":"H",
  "text_config:intermediate_size":"H_expert",
  "text_config:dense_intermediate_size":"H_dense",
  "text_config:shared_intermediate_size":"H_shared",
  "text_config:num_hidden_layers":"L",
  "text_config:num_attention_heads":"Nₕ",
  "text_config:num_key_value_heads":"Nₖᵥ",
  "text_config:head_dim":"Dₕ",
  "text_config:vocab_size":"V",
  "text_config:max_position_embeddings":"S_max",
  "text_config:rms_norm_eps":"ε_rms",
  "text_config:rope_theta":"θ_base",
  "text_config:rotary_dim":"Dᵣ",
  "text_config:partial_rotary_factor":"Dᵣ/Dₕ",
  "text_config:num_local_experts":"E",
  "text_config:num_experts_per_tok":"K",
  "text_config:n_shared_experts":"E_shared",
  "text_config:num_mtp_modules":"N_mtp",
  "text_config:num_nextn_predict_layers":"L_mtp",
  "text_config:swiglu_alpha":"α",
  "text_config:swiglu_beta":"β",
  "text_config:swiglu_limit":"c",
  "text_config:routed_scaling_factor":"s_route",
  "text_config.sparse_attention_config:sparse_index_dim":"D_idx",
  "text_config.sparse_attention_config:sparse_num_index_heads":"N_idx",
  "text_config.sparse_attention_config:sparse_topk_blocks":"K_block",
  "text_config.sparse_attention_config:sparse_block_size":"B_block",
  "text_config.sparse_attention_config:sparse_init_block":"B_init",
  "text_config.sparse_attention_config:sparse_local_block":"B_local",
  "vision_config:hidden_size":"Hᵥ",
  "vision_config:num_attention_heads":"Nₕᵥ",
  "vision_config:num_hidden_layers":"Lᵥ",
  "vision_config:intermediate_size":"H_ffnᵥ",
  "vision_config:patch_size":"P",
  "vision_config:image_size":"R",
  "vision_config:projection_dim":"H",
  "vision_config:rope_theta":"θᵥ",
  "vision_config:attention_dropout":"p_attn",
  "vision_config:initializer_factor":"s_init",
  "vision_config:initializer_range":"σ_init",
  "vision_config:layer_norm_eps":"ε_ln",
  "vision_config:num_channels":"C",
  "vision_config:vocab_size":"Vᵥ",
  "vision_config:vision_segment_max_frames":"F_max",
  "图像 token 压缩（顶层与 vision_config 内相同）:spatial_merge_size":"Mₛ",
  "图像 token 压缩（顶层与 vision_config 内相同）:temporal_patch_size":"Pₜ",
};

function configSymbol(group:string,key:string){
  return CONFIG_SYMBOLS[`${group}:${key}`]??"—";
}

const SIMPLE_FORMULA: Partial<Record<OpKind,string>> = {
  norm:String.raw`y=\operatorname{Norm}(x)`,linear:String.raw`y=xW^{\mathsf T}`,split:String.raw`(a,b,\ldots)=\operatorname{Split}(x)`,rope:String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`,matmul:String.raw`y=a\,b^{\mathsf T}`,scale:String.raw`y=x/\sqrt{d_h}`,mask:String.raw`y=x+\mathrm{mask}`,softmax:String.raw`p=\operatorname{softmax}(x)`,activation:String.raw`y=\bar g\,\sigma(\alpha\bar g)\,(\bar u+\beta)`,route:String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`,cache:String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(K,V)`,add:String.raw`y=x+f(x)`,io:String.raw`y=x`,
};

const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"把每个 token 的向量缩放到稳定范围；shape 不变。",linear:"W 是当前模块绑定的权重；最后一维由 W 的输出维决定。",split:"只切分最后一维，不做数值计算，也没有权重。",rope:"position 决定旋转角度；这里只旋转每个 head 的前 64 维。",matmul:"沿共同的 head_dim 相乘并求和。",scale:"dₕ=128；缩放避免 score 随维度增大。",mask:"不可见位置加 −∞，softmax 后概率变为 0。",softmax:"把每行 score 转为和为 1 的概率。",activation:"g 是 gate，u 是 up；实际实现还包含 limit=7 的截断。",route:"只选择去哪里计算；Top-K 本身不生成 expert 输出。",cache:"slot 与 block table 由 runtime 提供，权重不参与。",add:"残差支路与计算支路逐元素相加，shape 必须一致。",io:"这是数据入口或运行时元数据，不执行可训练计算。",
};

type FormulaTerm = readonly [symbol:string,meaning:string];
type FormulaStep = { title:string; formula:string; explanation:string };

const FORMULA_TERMS_BY_KIND: Record<OpKind,readonly FormulaTerm[]> = {
  io:[["x","输入"],["y","输出"]],
  norm:[["x","输入向量"],["y","归一化输出"],["H","归一化维度"],["γ","可训练缩放权重"],["ε","数值稳定项"],["RMS(x)","均方根"]],
  linear:[["x","输入张量"],["W","投影权重"],["y","线性投影输出"]],
  split:[["x","待切分张量"],["a,b,…","沿最后一维得到的输出"]],
  rope:[["q / k","Q 或 K 向量"],["p","token position"],["dᵣ","参与旋转的维度"],["θ","旋转角度"]],
  matmul:[["a","左输入张量"],["b","右输入张量"],["y","矩阵乘输出"]],
  scale:[["x","未缩放分数"],["dₕ","head_dim = 128"],["y","缩放后分数"]],
  mask:[["x","原始 attention score"],["M","causal / padding mask"],["y","mask 后 score"]],
  softmax:[["x","输入 score"],["p","归一化概率"]],
  activation:[["g","gate 分支"],["u","up 分支"],["α","sigmoid scale = 1.702"],["c","clamp limit = 7"],["y","SwiGLU-OAI 输出"]],
  route:[["s","路由分数"],["K","选择数量"],["𝓔 / I","选中的 expert 或 block id"]],
  cache:[["K / V","写入 cache 的张量"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"]],
  add:[["x","residual 分支"],["f(x)","当前计算分支"],["y","逐元素相加结果"]],
};

const FORMULA_TERMS_BY_ID: Partial<Record<string,readonly FormulaTerm[]>> = {
  "d-norm":[["x","hidden_states · [B,S,H]"],["y","normalized hidden_states"],["H","hidden_size = 6144"],["γ","input_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(x)","√(Σxⱼ²/H + ε)"]],
  "s-norm":[["x","hidden_states · [B,S,H]"],["y","normalized hidden_states"],["H","hidden_size = 6144"],["γ","input_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(x)","√(Σxⱼ²/H + ε)"]],
  "d-postnorm":[["U","上游 Add 节点的输出"],["Û","Gemma RMSNorm(U)"],["H","hidden_size = 6144"],["γpost","post_attention_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(U)","√(ΣUⱼ²/H + ε)"]],
  "s-postnorm":[["U","上游 Add 节点的输出"],["Û","Gemma RMSNorm(U)"],["H","hidden_size = 6144"],["γpost","post_attention_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(U)","√(ΣUⱼ²/H + ε)"]],
  "d-qnorm":[["Q","Query heads"],["Q̃","归一化后的 Q"],["Dₕ","head_dim = 128"],["γQ","q_norm.weight"],["ε","10⁻⁶"]],
  "d-knorm":[["K","Key heads"],["K̃","归一化后的 K"],["Dₕ","head_dim = 128"],["γK","k_norm.weight"],["ε","10⁻⁶"]],
  "s-mainnorm":[["Q / K","主 Attention 的 Q/K"],["Q̃ / K̃","归一化后的 Q/K"],["Dₕ","head_dim = 128"],["γQ / γK","Q/K norm weights"],["ε","10⁻⁶"]],
  "s-idxnorm":[["Qidx / Kidx","Indexer 的投影输入"],["Q̃idx / K̃idx","RMSNorm 后的 Index Q/K"],["Qidxᵣ / Kidxᵣ","加入 position 后的旋转结果"],["Didx","index_dim = 128"],["p","query/key position"]],
  "d-gateup":[["Û","归一化输入 · [B,S,H]"],["Wgate⁽ʳ⁾","当前 TP rank 的 gate 权重"],["Wup⁽ʳ⁾","当前 TP rank 的 up 权重"],["G⁽ʳ⁾","当前 rank 的 gate 投影"],["U⁽ʳ⁾","当前 rank 的 up 投影"],["H","hidden_size = 6144"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-gatesplit":[["X⁽ʳ⁾","当前 rank 的 packed gate_up"],["G⁽ʳ⁾","前 H_dense/TP 个通道"],["U⁽ʳ⁾","后 H_dense/TP 个通道"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-swiglu":[["G⁽ʳ⁾","当前 rank 的 gate 分片"],["U⁽ʳ⁾","当前 rank 的 up 分片"],["Ḡ⁽ʳ⁾ / Ū⁽ʳ⁾","clamp 后的两个分片"],["α","swiglu_alpha = 1.702"],["β","swiglu_beta = 1.0"],["c","swiglu_limit = 7.0"],["Z⁽ʳ⁾","当前 rank 的激活输出"]],
  "d-qk":[["Qᵣ","当前 rank 的 rotated Q"],["Kᵣ","当前 rank 的 rotated / visible K"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["Dₕ","head_dim = 128"],["A","当前 rank 的 attention scores"]],
  "d-pv":[["P","当前 rank 的 attention probability"],["V","当前 rank 的 visible V"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["O","当前 rank 的 context heads"]],
  "s-idxscore":[["Qidxᵣ","当前 rank 的 Index Q query"],["𝒦idx","独立 side cache 中的完整 Index K history"],["N_idx,rank","max(1,4/TP)"],["D_idx","index_dim = 128"],["Sidx","尚未 mask 的 Index token scores"]],
  "s-idxcache":[["Kidxᵣ","当前 token 的旋转后 Index K"],["slot","index slot_mapping 指定的 side-cache 位置"],["𝒦idx","独立的 key-only Index K cache"],["D_idx","每 token 一个 128 维 Index K 向量"]],
  "s-idxmask":[["Sidx","Index Q/K 点积分数"],["p","当前 query position"],["j","key position"],["S̃idx","未来 key 被置为 −∞ 后的分数"]],
  "s-qk":[["Qᵣ","当前 rank 的 rotated Q"],["paged K","主 Paged KV Cache 中的 K pages"],["block_indices","Indexer 输出的 Top-16 逻辑块索引"],["Ksel","kernel 最多读取 16×128 个候选 token"],["A","当前 rank 的 sparse scores"]],
  "s-pv":[["P","当前 rank 的 sparse probability"],["paged V","主 Paged KV Cache 中的 V pages"],["block_indices","与 Q×K 阶段相同的 Top-16 块顺序"],["O","当前 rank 的 context heads · 8192/TP"]],
  "s-topk":[["B","block scores"],["𝓛ᵢ","当前 query 的 local block 集合"],["+∞","保证 local blocks 必然进入候选"],["K_block","sparse_topk_blocks = 16"],["𝒮 / block_indices","每组、每个 query 选中的逻辑 blocks"]],
  "s-router":[["Û","Post-attn RMSNorm 输出"],["Wrouter","block_sparse_moe.gate.weight"],["E","num_local_experts = 128"],["r","router_logits · [B,S,E]"]],
  "s-experts":[["r","FP32 router_logits · [B,S,E]"],["σ","逐元素 sigmoid 函数"],["sₑ","专家 e 的 sigmoid 路由分数 σ(rₑ)"],["b","e_score_correction_bias；只影响专家选择"],["K","num_experts_per_tok = 4"],["𝓔","按 s+b 选出的 Top-K expert 集合"],["e / j","选中集合中的 expert 索引"],["ŵₑ","专家 e 的归一化混合权重"],["s_route","routed_scaling_factor = 2.0"],["Û","Post-attn RMSNorm 输出；每个专家的输入"],["Eₑ(Û)","第 e 个 expert 对 Û 的输出"],["Yᵣₒᵤₜₑd","4 个选中专家加权归并后的 routed output"]],
  "d-qkv":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["Dₕ","head_dim = 128"],["TP","tensor parallel size"],["Z","当前 rank 的 packed QKV"]],
  "d-split":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["Q / K / V","当前 rank 的三个输出"]],
  "d-ropeq":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 5000000"],["p","token position"],["Qᵣ","旋转后的 Q"]],
  "d-ropek":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 5000000"],["p","token position"],["Kᵣ","旋转后的 K"]],
  "d-oproj":[["Nₕ","num_attention_heads = 64"],["Dₕ","head_dim = 128"],["H","hidden_size = 6144"],["TP","tensor parallel size"],["W_O","o_proj.weight"]],
  "d-down":[["Z⁽ʳ⁾","当前 TP rank 的激活输出"],["Wdown⁽ʳ⁾","当前 rank 的 down 权重"],["H_dense","dense_intermediate_size = 12288"],["H","hidden_size = 6144"],["TP","tensor parallel size"]],
  "s-packed":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["N_idx","sparse_num_index_heads = 4"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"],["TP","tensor parallel size"]],
  "s-split":[["Nₕ/TP","每 rank query heads = 64/TP"],["Nₖᵥ,rank","max(1,4/TP)"],["N_idx,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"]],
  "s-blockmax":[["B_block","sparse_block_size = 128"],["S̃idx","已排除未来 key 的 Index scores"],["B","每 128 keys 取 max 后的 block scores"]],
  "s-rope":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 5000000"],["p","token position"]],
  "s-oproj":[["Nₕ","num_attention_heads = 64"],["Dₕ","head_dim = 128"],["H","hidden_size = 6144"],["TP","tensor parallel size"],["W_O","o_proj.weight"]],
  "d-add2":[["U","Attention 后的 residual stream · [B,S,H]"],["Yffn","Dense FFN 输出 · [B,S,H]"],["Xₗ₊₁","逻辑上的下一层输入"],["H","hidden_size = 6144"]],
  "s-addout":[["U","Attention 后的 residual stream · [B,S,H]"],["Ymoe","MoE 输出 · [B,S,H]"],["Xₗ₊₁","逻辑上的下一层输入"],["H","hidden_size = 6144"]],
  "d-add1":[["Xₗ","进入本层的 residual stream · [B,S,H]"],["Yattn","Attention 输出 · [B,S,H]"],["U","更新后的 residual stream"],["H","hidden_size = 6144"]],
  "s-addattn":[["Xₗ","进入本层的 residual stream · [B,S,H]"],["Yattn","Sparse Attention 输出 · [B,S,H]"],["U","更新后的 residual stream"],["H","hidden_size = 6144"]],
  "d-position":[["q_b","请求 b 本轮调度的 query 数"],["p_b,i","请求 b 的第 i 个 token position"],["N_q","本轮 query token 总数"],["B","batch size"]],
  "s-position":[["q_b","请求 b 本轮调度的 query 数"],["p_b,i","请求 b 的第 i 个 token position"],["N_q","本轮 query token 总数"],["B","batch size"]],
  "d-attnmeta":[["q_b","请求 b 的 query 数"],["c_b","请求 b 已有 context 长度"],["M","causal / padding mask"],["−∞","不可见位置的加性 mask 值"]],
  "s-attnmeta":[["q_b","请求 b 的 query 数"],["c_b","请求 b 已有 context 长度"],["M","causal / padding mask"],["−∞","不可见位置的加性 mask 值"]],
  "d-slots":[["p","token position"],["B_block","KV cache block size · runtime"],["b_phys","物理 block id"],["block_table","逻辑块到物理页映射"],["slot","KV cache 物理槽位"]],
  "s-slots":[["p","token position"],["B_block","KV cache block size · runtime"],["b_phys","物理 block id"],["block_table","逻辑块到物理页映射"],["slot","KV cache 物理槽位"]],
  "d-cache":[["Kᵣ / V","写入 cache 的 Key / Value"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"],["K≤p / V≤p","当前 token 可见的 KV"]],
  "d-scale":[["A","未缩放 attention scores"],["Ā","缩放后的 scores"],["Dₕ","head_dim = 128"]],
  "d-mask":[["Ā","缩放后的 scores"],["M","causal / padding mask"],["Ã","mask 后的 scores"],["c_b","请求 b 的 context 长度"]],
  "d-softmax":[["Ã","mask 后的 scores"],["P","attention probability"],["T","可见 KV token 数"],["m","每行最大值，用于数值稳定"]],
  "s-cache":[["Kᵣ / V","写入 sparse cache 的 Key / Value"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"]],
  "s-scale":[["A","未缩放 sparse scores"],["Ā","缩放后的 sparse scores"],["Dₕ","head_dim = 128"]],
  "s-mask":[["Ā","selected K 上的缩放 scores"],["valid_token","候选 blocks 内的 causal / padding 判定"],["Ã","token mask 后的 selected scores"],["K𝒮","block_indices 限定的候选 K view"]],
  "s-softmax":[["Ã","mask 后的 sparse scores"],["P","selected KV 上的概率"],["𝒮","当前 query 的候选 token 集合"]],
  "s-shared":[["u","Shared Expert 输入 · [B,S,H]"],["W₁,s","shared gate_proj.weight"],["W₃,s","shared up_proj.weight"],["W₂,s","shared down_proj.weight"],["H","hidden_size = 6144"],["H_shared","shared_intermediate_size = 3072"],["E_shared(u)","Shared Expert 输出"]],
  "s-sum":[["Y_routed","FusedMoE 已完成加权归并的输出"],["Y_shared","Shared Expert 输出"],["Y_moe","两路逐元素相加后的 MoE 输出"]],
};

const FORMULA_STEPS_BY_ID: Partial<Record<string,readonly FormulaStep[]>> = {
  "s-experts":[
    {title:"1 · 得分与选择",formula:String.raw`s=\sigma(r),\qquad \mathcal E=\operatorname{TopK}_{K}(s+b)`,explanation:"r 是 Router 输出的 128 个 FP32 logits；σ 对每个 logit 做 sigmoid，得到路由分数 s。b 是 correction bias，只在选择专家时加到 s 上；K=4，所以 𝓔 表示当前 token 选中的 4 个专家。目的：确定这个 token 应交给哪些专家计算。"},
    {title:"2 · 生成混合权重",formula:String.raw`\hat w_e=s_{route}\,\frac{s_e}{\sum_{j\in\mathcal E}s_j}`,explanation:"e 和 j 都是 𝓔 中的专家索引；sₑ 是专家 e 未加 correction bias 的 sigmoid 分数。分母把 4 个入选专家的分数归一化，再乘 s_route=2.0 得到 ŵₑ。目的：决定每个入选专家对最终 routed output 的贡献比例；correction bias 不进入该权重。"},
    {title:"3 · 专家计算与归并",formula:String.raw`Y_{\mathrm{routed}}=\sum_{e\in\mathcal E}\hat w_eE_e(\hat U)`,explanation:"Û 是 Post-attn RMSNorm 输出，也是各专家共享的输入；Eₑ(Û) 是专家 e 对该 token 的计算结果。每个结果乘对应的 ŵₑ，再对 4 个专家求和，得到 Y_routed。目的：把多个专家结果还原成每个 token 的一个 [H] 输出向量。"},
  ],
};

function formulaTerms(node:OpNode){
  return FORMULA_TERMS_BY_ID[node.id]??(node.latex?[]:FORMULA_TERMS_BY_KIND[node.kind]);
}

const LATEX_BY_ID: Record<string,string> = {
  "d-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "s-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "d-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "s-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "d-slots":String.raw`\begin{aligned}\ell&=\left\lfloor p/B_{block}\right\rfloor,\quad o=p\bmod B_{block}\\b_{\mathrm{phys}}&=\mathrm{block\_table}[r,\ell]\\\mathrm{slot}(r,p)&=b_{\mathrm{phys}}\cdot B_{block}+o\end{aligned}`,
  "s-slots":String.raw`\begin{aligned}\ell&=\left\lfloor p/B_{block}\right\rfloor,\quad o=p\bmod B_{block}\\b_{\mathrm{phys}}&=\mathrm{block\_table}[r,\ell]\\\mathrm{slot}(r,p)&=b_{\mathrm{phys}}\cdot B_{block}+o\end{aligned}`,
  "d-norm":String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\frac1H\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}(1+\gamma_i)\end{aligned}`,
  "d-qkv":String.raw`\begin{aligned}Z&=\hat X\,[W_Q^\top\mid W_K^\top\mid W_V^\top]\\Z&\in\mathbb R^{B\times S\times((N_h+2N_{kv})D_h/TP)}\end{aligned}`,
  "d-split":String.raw`(Q,K,V)=\operatorname{Split}\!\left(Z;\frac{N_hD_h}{TP},N_{kv,\mathrm{rank}}D_h,N_{kv,\mathrm{rank}}D_h\right)`,
  "d-qnorm":String.raw`\begin{aligned}\operatorname{RMS}(Q_{b,h,s})&=\sqrt{\frac1{D_h}\sum_{j=1}^{D_h}Q_{b,h,s,j}^2+\varepsilon}\\\tilde Q_{b,h,s,i}&=\frac{Q_{b,h,s,i}}{\operatorname{RMS}(Q_{b,h,s})}(1+\gamma_{Q,i})\end{aligned}`,
  "d-knorm":String.raw`\begin{aligned}\operatorname{RMS}(K_{b,g,s})&=\sqrt{\frac1{D_h}\sum_{j=1}^{D_h}K_{b,g,s,j}^2+\varepsilon}\\\tilde K_{b,g,s,i}&=\frac{K_{b,g,s,i}}{\operatorname{RMS}(K_{b,g,s})}(1+\gamma_{K,i})\end{aligned}`,
  "d-ropeq":String.raw`\begin{aligned}(Q_{\mathrm{rot}},Q_{\mathrm{pass}})&=\operatorname{Split}(\tilde Q;D_r,D_h-D_r)\\Q^r&=\operatorname{Concat}(\operatorname{RoPE}(Q_{\mathrm{rot}},p),Q_{\mathrm{pass}})\end{aligned}`,
  "d-ropek":String.raw`\begin{aligned}(K_{\mathrm{rot}},K_{\mathrm{pass}})&=\operatorname{Split}(\tilde K;D_r,D_h-D_r)\\K^r&=\operatorname{Concat}(\operatorname{RoPE}(K_{\mathrm{rot}},p),K_{\mathrm{pass}})\end{aligned}`,
  "d-cache":String.raw`\begin{aligned}\mathcal K[\mathrm{slot}(r,p)]&\leftarrow K^r_{r,p}\\\mathcal V[\mathrm{slot}(r,p)]&\leftarrow V_{r,p}\\K_{\le p},V_{\le p}&\leftarrow\operatorname{gather}(\mathcal K,\mathcal V,\mathrm{block\_table}_r)\end{aligned}`,
  "d-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^r_{b,h,i,m}\,K^r_{b,\lfloor h/G\rfloor,j,m}`,
  "d-scale":String.raw`\bar A_{b,h,i,j}=\frac{A_{b,h,i,j}}{\sqrt{D_h}}`,
  "d-mask":String.raw`\tilde A_{b,h,i,j}=\bar A_{b,h,i,j}+M_{b,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\le c_b+i\\-\infty,&j>c_b+i\end{cases}`,
  "d-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-m_{b,h,i})}{\sum_{t=0}^{T-1}\exp(\tilde A_{b,h,i,t}-m_{b,h,i})},\quad m_{b,h,i}=\max_t\tilde A_{b,h,i,t}`,
  "d-pv":String.raw`O_{b,h,i,m}=\sum_{j=0}^{T-1}P_{b,h,i,j}\,V_{b,\lfloor h/G\rfloor,j,m}`,
  "d-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}\!\left(\operatorname{Concat}_{h=1}^{N_h/TP}(O_h),W_O\right)\in\mathbb R^{B\times S\times H}`,
  "d-add1":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "d-postnorm":String.raw`\begin{aligned}\operatorname{RMS}(U)&=\sqrt{\frac1H\sum_{j=1}^{H}U_j^2+\varepsilon}\\\hat U_i&=\frac{U_i}{\operatorname{RMS}(U)}(1+\gamma_{\mathrm{post},i})\end{aligned}`,
  "d-gateup":String.raw`\begin{aligned}G^{(r)}&=\hat U\left(W_{\mathrm{gate}}^{(r)}\right)^\top\\U^{(r)}&=\hat U\left(W_{\mathrm{up}}^{(r)}\right)^\top\\G^{(r)},U^{(r)}&\in\mathbb R^{B\times S\times(H_{\mathrm{dense}}/TP)}\end{aligned}`,
  "d-gatesplit":String.raw`\begin{aligned}X^{(r)}&\in\mathbb R^{B\times S\times(2H_{\mathrm{dense}}/TP)}\\G^{(r)}&=X^{(r)}_{:,:,\,0:H_{\mathrm{dense}}/TP}\\U^{(r)}&=X^{(r)}_{:,:,\,H_{\mathrm{dense}}/TP:2H_{\mathrm{dense}}/TP}\end{aligned}`,
  "d-swiglu":String.raw`\begin{aligned}\bar G^{(r)}&=\min(G^{(r)},c)\\\bar U^{(r)}&=\operatorname{clip}(U^{(r)},-c,c)\\Z^{(r)}&=\bar G^{(r)}\odot\sigma(\alpha\bar G^{(r)})\odot(\bar U^{(r)}+\beta)\end{aligned}`,
  "d-down":String.raw`Y_{\mathrm{ffn}}=\sum_r Z^{(r)}\left(W_{\mathrm{down}}^{(r)}\right)^\top\in\mathbb R^{B\times S\times H}`,
  "d-add2":String.raw`X_{l+1}=U+Y_{\mathrm{ffn}}`,
  "s-norm":String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\frac1H\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}(1+\gamma_i)\end{aligned}`,
  "s-postnorm":String.raw`\begin{aligned}\operatorname{RMS}(U)&=\sqrt{\frac1H\sum_{j=1}^{H}U_j^2+\varepsilon}\\\hat U_i&=\frac{U_i}{\operatorname{RMS}(U)}(1+\gamma_{\mathrm{post},i})\end{aligned}`,
  "s-packed":String.raw`Z=\hat X[W_Q^\top\mid W_K^\top\mid W_V^\top\mid W_{Q_i}^\top\mid W_{K_i}^\top]`,
  "s-split":String.raw`Z\longrightarrow(Q_{N_hD_h/TP},K_{N_{kv,\mathrm{rank}}D_h},V_{N_{kv,\mathrm{rank}}D_h},Q^{\mathrm{idx}}_{N_{idx,\mathrm{rank}}D_{idx}},K^{\mathrm{idx}}_{D_{idx}})`,
  "s-idxnorm":String.raw`\begin{aligned}\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}}&=\operatorname{RMSNorm}(Q^{\mathrm{idx}}),\operatorname{RMSNorm}(K^{\mathrm{idx}})\\Q^{\mathrm{idx},r},K^{\mathrm{idx},r}&=\operatorname{RoPE}(\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}};\mathbf p)\end{aligned}`,
  "s-idxcache":String.raw`\mathcal K_{\mathrm{idx}}[\mathrm{slot}(b,p)]\leftarrow K^{\mathrm{idx},r}_{b,p}\in\mathbb R^{D_{idx}}`,
  "s-idxscore":String.raw`S^{(r)}_{b,i,j}=\left\langle Q^{\mathrm{idx},r}_{b,r,i,:},\mathcal K_{\mathrm{idx}}[b,j,:]\right\rangle`,
  "s-idxmask":String.raw`\tilde S^{(r)}_{b,i,j}=\begin{cases}S^{(r)}_{b,i,j},&j\le p_{b,i}\\-\infty,&j>p_{b,i}\end{cases}`,
  "s-blockmax":String.raw`B^{(r)}_{b,i,u}=\max_{j\in[B_{block}u,B_{block}(u+1))}\tilde S^{(r)}_{b,i,j}`,
  "s-topk":String.raw`\begin{aligned}B_u&\leftarrow+\infty,\quad u\in\mathcal L_i\\\mathcal S_{b,r,i}&=\operatorname{TopK}_{K_{block}}(B)\end{aligned}`,
  "s-mainnorm":String.raw`\tilde Q=\operatorname{RMSNorm}(Q),\qquad\tilde K=\operatorname{RMSNorm}(K)`,
  "s-rope":String.raw`\begin{aligned}(Q^r_{:D_r},K^r_{:D_r})&=\operatorname{RoPE}(\tilde Q_{:D_r},\tilde K_{:D_r};\mathbf p)\\(Q^r_{D_r:},K^r_{D_r:})&=(\tilde Q_{D_r:},\tilde K_{D_r:})\end{aligned}`,
  "s-cache":String.raw`\mathcal K[\mathrm{slot}(r,p)]\leftarrow K^r_{r,p},\qquad\mathcal V[\mathrm{slot}(r,p)]\leftarrow V_{r,p}`,
  "s-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^r_{b,h,i,m}(K_{\mathcal S})_{b,\lfloor h/G\rfloor,j,m},\quad j\in\mathcal S_{b,\lfloor h/G\rfloor,i}`,
  "s-scale":String.raw`\bar A_{b,h,i,j}=A_{b,h,i,j}/\sqrt{D_h}`,
  "s-mask":String.raw`\tilde A_{b,h,i,j}=\begin{cases}\bar A_{b,h,i,j},&\mathrm{valid\_token}(b,i,j)\\-\infty,&\text{future or padding}\end{cases},\quad j\in K_{\mathcal S}`,
  "s-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-\max_t\tilde A_{b,h,i,t})}{\sum_{t\in\mathcal S_i}\exp(\tilde A_{b,h,i,t}-\max_u\tilde A_{b,h,i,u})}`,
  "s-pv":String.raw`O_{b,h,i,m}=\sum_{j\in\mathcal S_i}P_{b,h,i,j}(V_{\mathcal S})_{b,\lfloor h/G\rfloor,j,m}`,
  "s-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}\!\left(\operatorname{Concat}_{h=1}^{N_h/TP}(O_h),W_O\right)`,
  "s-addattn":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "s-router":String.raw`r=\hat U W_{\mathrm{router}}^\top\in\mathbb R^{B\times S\times E}`,
  "s-experts":String.raw`\begin{aligned}s&=\sigma(r),\qquad\mathcal E=\operatorname{TopK}_K(s+b)\\\hat w_e&=s_{route}\,\frac{s_e}{\sum_{j\in\mathcal E}s_j}\\Y_{\mathrm{routed}}&=\sum_{e\in\mathcal E}\hat w_eE_e(\hat U)\end{aligned}`,
  "s-shared":String.raw`E_{\mathrm{shared}}(u)=W_{2,s}\operatorname{SwiGLUOAI}(W_{1,s}u,W_{3,s}u)`,
  "s-sum":String.raw`Y_{\mathrm{moe}}=Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)`,
  "s-addout":String.raw`X_{l+1}=U+Y_{\mathrm{moe}}`,
};

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

const QKV_INDEX_PROJECTION_SECTIONS: CodeSection[] = [
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

const QKV_INDEX_PROJECTION_SYMBOLS: CodeSymbol[] = [
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

const CODE_BY_ID: Record<string, CodeDetail> = {};
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

const NEXT_BY_ID: Record<string,string> = {
  "d-input":"Gemma RMSNorm","d-position":"Partial RoPE (Q/K)","d-attnmeta":"Apply Causal / Pad Bounds","d-slots":"Paged KV Cache","d-norm":"QKV Projection","d-qkv":"Split Q / K / V","d-split":"Q RMSNorm · K RMSNorm · Paged KV Cache","d-qnorm":"Partial RoPE (Q)","d-knorm":"Partial RoPE (K)","d-ropeq":"Q × Kᵀ","d-ropek":"Paged KV Cache","d-cache":"Q × Kᵀ · P × V","d-qk":"Scale 1/√128","d-scale":"Apply Causal / Pad Bounds","d-mask":"Softmax","d-softmax":"P × V","d-pv":"O Projection","d-oproj":"Attention Residual Merge","d-add1":"Post-attn Gemma RMSNorm","d-postnorm":"Gate + Up Projection","d-gateup":"Split Gate / Up","d-gatesplit":"SwiGLU-OAI","d-swiglu":"Down Projection","d-down":"Decoder Layer Residual Merge","d-add2":"下一 decoder layer / Final Norm",
  "s-input":"Gemma RMSNorm","s-position":"Partial RoPE","s-attnmeta":"Indexer 与 Sparse Attention mask","s-slots":"Paged KV Cache","s-norm":"QKV + Index Projection","s-packed":"Split 5 outputs","s-split":"Index Q/K Gemma RMSNorm + RoPE · Main Q/K Gemma RMSNorm · Paged KV Cache","s-idxnorm":"Index Q query · Index K Cache","s-idxcache":"Index Q × cached Kᵀ","s-idxscore":"Mask Future Index Keys","s-idxmask":"Block Max","s-blockmax":"Top-16 Blocks","s-topk":"Q × paged Kᵀ · Top-16","s-mainnorm":"Partial RoPE","s-rope":"Paged KV Cache · Q × paged Kᵀ · Top-16","s-cache":"Q × paged Kᵀ · Top-16 · P × paged V","s-qk":"Scale 1/√128","s-scale":"Apply Token Causal / Pad Mask","s-mask":"Softmax","s-softmax":"P × paged V · same Top-16","s-pv":"O Projection","s-oproj":"Attention Residual Merge","s-addattn":"Post-attn Gemma RMSNorm","s-postnorm":"FP32 Router Logits · Fused Top-4 Routing + Experts · Shared Expert","s-router":"router_logits → Fused Top-4 Routing + Experts","s-experts":"Add Routed + Shared","s-shared":"Add Routed + Shared","s-sum":"Decoder Layer Residual Merge","s-addout":"下一 decoder layer / Final Norm",
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
    norm: cloneOp(norm,{id:"d-norm",kind:"norm",title:"Gemma RMSNorm",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL}),
    qkv: cloneOp(qkv,{id:"d-qkv",kind:"linear",title:"QKV Projection"}),
    split: cloneOp(qkv,{id:"d-split",kind:"split",title:"Split Q / K / V",input:"packed qkv",inputShape:"[B,S,9216]",output:"Q · K · V",outputShape:"8192 · 512 · 512",formula:"split(qkv,[8192,512,512],dim=-1)",formulaNote:"checkpoint 中三块矩阵分离；vLLM 运行时一次 GEMM 后切分。",weights:[]}),
    qnorm: cloneOp(attn,{id:"d-qnorm",kind:"norm",title:"Q Gemma RMSNorm · per-head",summary:"对每个 Q head 的 128 维向量独立执行 Gemma 风格 RMSNorm。",input:"Q",inputShape:"[B,64,S,128]",output:"Q̃",outputShape:"[B,64,S,128]",formula:"Q̃ₕ,ᵢ=Qₕ,ᵢ/√((1/Dₕ)ΣⱼQₕ,ⱼ²+ε)·(1+γQ,ᵢ)",formulaNote:"归一化轴仅为 head_dim=128；同一组 [128] q_norm.weight 应用于各个 Q head。",runtime:"fused_minimax_m3_qknorm_rope_kv_insert · Q norm stage",weights:attn.weights.filter(w=>w.key.includes("q_norm"))}),
    knorm: cloneOp(attn,{id:"d-knorm",kind:"norm",title:"K Gemma RMSNorm · per-head",summary:"对每个 K head 的 128 维向量独立执行 Gemma 风格 RMSNorm。",input:"K",inputShape:"[B,4,T,128]",output:"K̃",outputShape:"[B,4,T,128]",formula:"K̃ₕ,ᵢ=Kₕ,ᵢ/√((1/Dₕ)ΣⱼKₕ,ⱼ²+ε)·(1+γK,ᵢ)",formulaNote:"归一化轴仅为 head_dim=128；同一组 [128] k_norm.weight 应用于各个 K head。",runtime:"fused_minimax_m3_qknorm_rope_kv_insert · K norm stage",weights:attn.weights.filter(w=>w.key.includes("k_norm"))}),
    ropeq: cloneOp(attn,{id:"d-ropeq",kind:"rope",title:"Partial RoPE (Q)",summary:"仅对每个 Q head 的前 64/128 维应用 RoPE，后 64 维保持不变。",input:"Q̃ + positions",inputShape:"[B,64,S,128] + [S]",output:"Qᵣ",outputShape:"[B,64,S,128]",formula:"split Q̃ into Qrot(64) and Qpass(64); Qᵣ=concat(RoPE(Qrot,p),Qpass)",formulaNote:"先把每个 128 维 Q head 拆成两个 64 维分段。Qrot 使用 token 位置 p 做旋转，Qpass 不变，最后按原顺序拼回 128 维。",runtime:"fused_minimax_m3_qknorm_rope_kv_insert · Q partial RoPE stage",weights:[]}),
    ropek: cloneOp(attn,{id:"d-ropek",kind:"rope",title:"Partial RoPE (K)",summary:"仅对每个 K head 的前 64/128 维应用 RoPE，后 64 维保持不变。",input:"K̃ + positions",inputShape:"[B,4,T,128] + [T]",output:"Kᵣ",outputShape:"[B,4,T,128]",formula:"split K̃ into Krot(64) and Kpass(64); Kᵣ=concat(RoPE(Krot,p),Kpass)",formulaNote:"先把每个 128 维 K head 拆成两个 64 维分段。Krot 使用 token 位置 p 做旋转，Kpass 不变，最后按原顺序拼回 128 维。",runtime:"fused_minimax_m3_qknorm_rope_kv_insert · K partial RoPE stage",weights:[]}),
    cache: cloneOp(attn,{id:"d-cache",kind:"cache",title:"Paged KV Cache",summary:"L0–L2 的 Full GQA 从 Paged KV Cache 读取全部因果可见的历史与当前 K/V。",input:"Kᵣ,V + block table",inputShape:"[T,4,128] ×2",output:"visible K,V",outputShape:"[B,4,T,128] ×2",formula:"slot = block_table[seq, logical_block] + offset",formulaNote:"“全部可见”不包括未来 token 或 padding；block table 决定逻辑 KV 位置对应的物理 page。",runtime:"Attention backend · full causal Paged KV",weights:[]}),
    qk: cloneOp(attn,{id:"d-qk",kind:"matmul",title:"Q × Kᵀ",input:"Qᵣ,Kᵣ",inputShape:"[B,64/TP,S,128] · [B,max(1,4/TP),T,128]",output:"local scores",outputShape:"[B,64/TP,S,T]",formula:"A=QᵣKᵣᵀ",formulaNote:"单个 TP rank 只计算 64/TP 个 query heads；KV heads 为 max(1,4/TP)，当 TP>4 时按 vLLM 规则复制。",weights:[]}),
    scale: cloneOp(attn,{id:"d-scale",kind:"scale",title:"Scale 1/√128",input:"A",inputShape:"[B,64/TP,S,T]",output:"scaled scores",outputShape:"[B,64/TP,S,T]",formula:"A←A/√128",weights:[]}),
    mask: cloneOp(attn,{id:"d-mask",kind:"mask",title:"Apply Causal / Pad Bounds",input:"scores + attention metadata",inputShape:"[B,64/TP,S,T] + runtime metadata",output:"masked scores",outputShape:"[B,64/TP,S,T]",formula:"Aᵢⱼ←valid(i,j) ? Aᵢⱼ : −∞",formulaNote:"图中把 mask 画成逻辑算子；vLLM 后端实际以 causal、seq_lens 和 query_start_loc 实现，不物化完整 mask 矩阵。head 维为单 TP rank 的 64/TP。",weights:[]}),
    softmax: cloneOp(attn,{id:"d-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64/TP,S,T]",output:"attention prob",outputShape:"[B,64/TP,S,T]",formula:"P=softmax(A,dim=-1)",weights:[]}),
    pv: cloneOp(attn,{id:"d-pv",kind:"matmul",title:"P × V",input:"P,V",inputShape:"[B,64/TP,S,T] · [B,max(1,4/TP),T,128]",output:"local heads",outputShape:"[B,S,8192/TP]",formula:"Oₕ=PₕV⌊h/16⌋",formulaNote:"P × V 在每个 TP rank 上独立计算，得到 (64/TP)×128=8192/TP 的局部 attention 宽度，再交给 RowParallel O Projection。",weights:[]}),
    oproj: cloneOp(out,{id:"d-oproj",kind:"linear",title:"O Projection"}),
    add1: cloneOp(out,{id:"d-add1",kind:"add",kicker:"DECODER LAYER · ATTENTION RESIDUAL",title:"Attention Residual Merge",summary:"在 Decoder Layer 内把 Attention 分支 Yattn 加入 residual stream Xₗ，得到更新后的 U；图中将 fused add 与紧随其后的 post-norm 分开表达。",input:"Xₗ + Yattn",inputShape:"2 × [B,S,6144]",output:"U · updated residual stream",outputShape:"[B,S,6144]",formula:"U=Xₗ+Yattn",formulaNote:"实际调用位于 DecoderLayer.forward L773：fused kernel 先执行 residual += hidden_states，再对更新后的 residual 执行 post-attention Gemma RMSNorm。",runtime:"fused_allreduce_gemma_rms_norm · attention residual",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L773–775",sourceUrl:`${CODE_URL}#L773-L775`,weights:[]}),
    postnorm: cloneOp(norm,{id:"d-postnorm",kind:"norm",title:"Post-attn Gemma RMSNorm",summary:"输入 U 已由上游 Add 节点计算完成；此节点只执行 Gemma RMSNorm(U)，输出唯一的 Û 作为 FFN 输入。",formulaNote:"U 是上游 Add 的单一输出；本节点只计算 RMS(U) 与 (1+γpost) 缩放，不重复执行 residual add。",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL,weights:[postNorm]}),
    gateup: cloneOp(mlp,{id:"d-gateup",kind:"linear",kicker:"DENSE FFN · H=6144 · H_dense=12288",title:"Gate + Up Projection",summary:"MergedColumnParallelLinear 让每个 TP rank 读取完整 Û，并分别计算局部 gate/up 投影；这里只做线性 GEMM。",input:"Û",inputShape:"[B,S,6144]",output:"packed gate_up (TP-local)",outputShape:"[B,S,24576/TP]",formulaNote:"Wgate⁽ʳ⁾ 与 Wup⁽ʳ⁾ 都沿输出维切分；本节点不执行 Split、clamp、SiLU 或逐元素乘。",runtime:"MergedColumnParallelLinear · gate_up_proj",weights:mlp.weights.filter(w=>!w.key.includes("down_proj"))}),
    gatesplit: cloneOp(mlp,{id:"d-gatesplit",kind:"split",kicker:"DENSE FFN · TP-LOCAL SPLIT",title:"Split Gate / Up",summary:"把当前 TP rank 的 packed gate_up 沿最后一维等分为 G⁽ʳ⁾ 和 U⁽ʳ⁾；不含权重，也不改变数值。",input:"packed gate_up (TP-local)",inputShape:"[B,S,24576/TP]",output:"G⁽ʳ⁾ · U⁽ʳ⁾",outputShape:"2 × [B,S,12288/TP]",formula:"(G⁽ʳ⁾,U⁽ʳ⁾)=split(gate_up⁽ʳ⁾,2,dim=-1)",formulaNote:"本节点只切分 view：前 H_dense/TP 个通道是 gate，后 H_dense/TP 个通道是 up；clamp 与 sigmoid 属于下一 SwiGLU-OAI 节点。",runtime:"SiluAndMulWithClamp · fused input slicing",source:"activation.py · SiluAndMulWithClamp.forward_native",sourceUrl:`${ACTIVATION_URL}#L214-L218`,weights:[]}),
    swiglu: cloneOp(mlp,{id:"d-swiglu",kind:"activation",kicker:"DENSE FFN · TP-LOCAL ACTIVATION",title:"SwiGLU-OAI",summary:"对当前 TP rank 的 G⁽ʳ⁾/U⁽ʳ⁾ 分片执行 gate 上界截断、up 双边截断、sigmoid、+β 与逐元素乘；不执行线性投影。",input:"G⁽ʳ⁾,U⁽ʳ⁾",inputShape:"2 × [B,S,12288/TP]",output:"Z⁽ʳ⁾",outputShape:"[B,S,12288/TP]",formulaNote:"forward_native 先以 c=7 截断两个分支，再计算 Ḡ⁽ʳ⁾⊙σ(αḠ⁽ʳ⁾)⊙(Ū⁽ʳ⁾+β)；α=1.702，β=1.0。",runtime:"vLLM: SiluAndMulWithClamp · layout: swigluoai_uninterleave",source:"activation.py · SiluAndMulWithClamp.forward_native · L214–218",sourceUrl:`${ACTIVATION_URL}#L214-L218`,weights:[]}),
    down: cloneOp(mlp,{id:"d-down",kind:"linear",kicker:"DENSE FFN · ROW PARALLEL · H=6144",title:"Down Projection",summary:"RowParallelLinear 消费每个 TP rank 的局部 activated 分片，将 H_dense/TP 投回 H，并归并各 rank 的部分结果。",input:"activated⁽ʳ⁾",inputShape:"[B,S,12288/TP]",output:"Yffn",outputShape:"[B,S,6144]",formulaNote:"本节点只执行 down projection；输入宽度为 H_dense/TP，输出隐藏宽度 H=6144。",weights:mlp.weights.filter(w=>w.key.includes("down_proj"))}),
    add2: cloneOp(mlp,{id:"d-add2",kind:"add",kicker:"DECODER LAYER · FFN RESIDUAL",title:"Decoder Layer Residual Merge",summary:"在 Decoder Layer 边界把 Dense FFN 分支 Yffn 与 residual stream U 逻辑合并，得到 Xₗ₊₁；该 merge 延迟融合到下一层 input RMSNorm。",input:"U + Yffn",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁ · logical next-layer input",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+Yffn",formulaNote:"当前层 L776–778 返回 Yffn 与 U 两条独立流；下一 Decoder Layer 在 L758–767 的 fused input RMSNorm 中执行实际 add。",runtime:"MiniMaxM3DecoderLayer boundary · deferred residual merge",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L758–778",sourceUrl:`${CODE_URL}#L758-L778`,weights:[]}),
  };
}

function sparseGraph(layer: number): Record<string, OpNode> {
  const [packed,indexer,topk,attn,router,experts,shared,combine]=sparseNodes(layer);
  const normBase=denseNodes(layer)[0];
  const shard=layerShard(layer);
  const inputNorm: Weight={key:`language_model.model.layers.${layer}.input_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  const postNorm: Weight={key:`language_model.model.layers.${layer}.post_attention_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  const routerGateWeights=router.weights.filter(weight=>weight.key.includes(".gate.weight"));
  return {
    input:cloneOp(packed,{id:"s-input",kind:"io",title:"Hidden states",input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[]}),
    position:cloneOp(attn,{id:"s-position",kind:"route",title:"Build Position IDs",kicker:"vLLM RUNTIME I/O",input:"num_computed_tokens + query offsets",inputShape:"[B] + [Nq]",output:"positions",outputShape:"[Nq]",formula:"position(req,i)=num_computed_tokens[req]+i",formulaNote:"positions 由 vLLM runner 在模型 forward 之前构造，再传给 MiniMax-M3 的 fused QKNorm + RoPE kernel。",source:"gpu_model_runner.py · _prepare_inputs",sourceUrl:RUNNER_URL,weights:[]}),
    attnmeta:cloneOp(attn,{id:"s-attnmeta",kind:"mask",title:"Build Attention Metadata",kicker:"vLLM RUNTIME I/O",input:"query_start_loc, seq_lens, causal=True",inputShape:"[B+1] · [B] · bool",output:"implicit causal / padding layout",outputShape:"backend metadata; 非稠密 [S,T]",formula:"valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)",formulaNote:"同一份边界元数据同时约束 indexer 的 block selection 和 main sparse attention。",source:"gpu_model_runner.py · CommonAttentionMetadata",sourceUrl:RUNNER_URL,weights:[]}),
    slots:cloneOp(attn,{id:"s-slots",kind:"route",title:"Resolve KV Slots",kicker:"vLLM RUNTIME I/O",input:"positions + block_table",inputShape:"[Nq] + [B,Nblocks]",output:"slot_mapping + block_table",outputShape:"[Nq] + [B,Nblocks]",formula:"slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)",formulaNote:"slot_mapping 用于 K/V 写入；block_table 把 indexer 选出的逻辑 block id 翻译为物理 page。",source:"gpu_model_runner.py · compute_slot_mapping",sourceUrl:RUNNER_URL,weights:[]}),
    norm:cloneOp(normBase,{id:"s-norm",kind:"norm",title:"Gemma RMSNorm",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL,weights:[inputNorm]}),
    packed:cloneOp(packed,{id:"s-packed",kind:"linear",title:"QKV + Index Projection",summary:"一次 column-parallel GEMM 同时产生 Q、K、V、Qidx、Kidx；vLLM 将五段结果保存在同一个 packed tensor 中。",formulaNote:"这是五路线性投影节点，不执行 Q/K Norm、RoPE、KV cache 写入或 Attention。",runtime:"vLLM: MinimaxM3QKVParallelLinearWithIndexer · packed [q|k|v|index_q|index_k]",codeSections:QKV_INDEX_PROJECTION_SECTIONS,codeSymbols:QKV_INDEX_PROJECTION_SYMBOLS}),
    split:cloneOp(packed,{id:"s-split",kind:"split",title:"Split 5 outputs",input:"packed projection",inputShape:"[B,S,9856]",output:"Q/K/V · Qidx/Kidx",outputShape:"8192/512/512 · 512/128",formula:"split(x,[8192,512,512,512,128],dim=-1)",weights:[]}),
    idxnorm:cloneOp(indexer,{id:"s-idxnorm",kind:"norm",title:"Index Q/K Gemma RMSNorm + RoPE",summary:"分别对 Qidx/Kidx 执行 Gemma 风格 RMSNorm，再用同一组 position embeddings 旋转；Qidxᵣ 直接进入打分，Kidxᵣ 写入独立 Index K cache。",input:"Qidx,Kidx + position embeddings",inputShape:"[B,S,4,128] · [B,T,1,128]",output:"Index Q query · current Index K",outputShape:"[B,4,S,128] · [B,S,128]",weights:indexer.weights}),
    idxcache:cloneOp(indexer,{id:"s-idxcache",kind:"cache",title:"Index K Cache · key-only",summary:"vLLM 为 Indexer 单独分配 side cache：每 token 只保存一个 128 维 Index K，不保存 Index V，也不与主 Paged KV Cache 混用。",input:"current Index K + index slot_mapping",inputShape:"[B,S,128] + [Nq]",output:"cached Index K history",outputShape:"key-only pages · [T,128]",formulaNote:"MiniMaxM3IndexerCache 使用独立 prefix 注册到 KV-cache manager；Indexer score kernel 通过 self.index_cache.kv_cache 读取历史 Index K。",runtime:"vLLM: MiniMaxM3IndexerCache · bf16/fp8_e4m3",source:"common/indexer.py · MiniMaxM3IndexerCache",sourceUrl:`${VLLM_INDEXER_URL}#L101-L151`,weights:[]}),
    idxscore:cloneOp(indexer,{id:"s-idxscore",kind:"matmul",title:"Index Q × cached Kᵀ",input:"Index Q query + cached Index K",inputShape:"[B,max(1,4/TP),S,128] · [B,T,128]",output:"Index token scores",outputShape:"[B,max(1,4/TP),S,T]",formulaNote:"这里只做 Index Q 与完整 Index K history 的点积；不缩放，也不在这个节点混入 causal mask。",weights:[]}),
    idxmask:cloneOp(indexer,{id:"s-idxmask",kind:"mask",title:"Mask Future Index Keys",summary:"在 block max 之前把未来 key 和补齐槽设为 −∞，防止不可见 token 影响选块。",input:"Index token scores + position_ids",inputShape:"[B,max(1,4/TP),S,T] + [B,S]",output:"causal Index scores",outputShape:"[B,max(1,4/TP),S,T]",weights:[]}),
    blockmax:cloneOp(indexer,{id:"s-blockmax",kind:"route",title:"Block Max · 128 keys",input:"causal Index scores",inputShape:"[B,max(1,4/TP),S,T]",output:"block scores",outputShape:"[B,max(1,4/TP),S,⌈T/128⌉]",formulaNote:"每 128 个 key 的 Index score 取最大值，得到一个 block score。",weights:[]}),
    topk:cloneOp(topk,{id:"s-topk",kind:"route",title:"Top-16 Blocks · per group",input:"block scores + local block priority",inputShape:"[B,max(1,4/TP),S,Nblocks]",output:"block_indices",outputShape:"[B,4,S,16]",formulaNote:"每个 query、每个 Index/KV group 独立选 16 个逻辑 blocks；local block 先以 +∞ 保证入选，无效槽记为 −1。"}),
    mainnorm:cloneOp(attn,{id:"s-mainnorm",kind:"norm",title:"Main Q/K Gemma RMSNorm",summary:"分别对主 Attention 的每个 Q/K head 执行 Gemma 风格 RMSNorm。",input:"Q,K",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"Q̃,K̃",outputShape:"same",weights:attn.weights.filter(w=>w.key.includes("_norm"))}),
    rope:cloneOp(attn,{id:"s-rope",kind:"rope",title:"Partial RoPE",input:"Q̃,K̃ + positions",inputShape:"Q/K + [S]",output:"Qᵣ,Kᵣ",outputShape:"Q/K unchanged",weights:[]}),
    cache:cloneOp(attn,{id:"s-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"KV pages + [B,Nblocks]",output:"paged K,V",outputShape:"[Npages,128,4,128] ×2",formula:"physical_page=block_table[logical_block]",weights:[]}),
    qk:cloneOp(attn,{id:"s-qk",kind:"matmul",title:"Q × paged Kᵀ · Top-16",input:"Qᵣ + paged K + block_indices",inputShape:"[B,64/TP,S,128] + KV pages + [B,4,S,16]",output:"local sparse scores",outputShape:"[B,64/TP,S,≤2048]",formulaNote:"没有独立的 KV-view 映射算子：paged-attention kernel 根据 block_indices 与 block_table 直接读取对应 K pages。",weights:[]}),
    scale:cloneOp(attn,{id:"s-scale",kind:"scale",title:"Scale 1/√128",input:"scores",inputShape:"[B,64/TP,S,≤2048]",output:"scaled scores",outputShape:"same",weights:[]}),
    mask:cloneOp(attn,{id:"s-mask",kind:"mask",title:"Apply Token Causal / Pad Mask",summary:"在 Top-16 候选 blocks 内继续排除未来 token 与 padding；选块范围和 token 可见性是两层不同约束。",input:"selected scores + token bounds",inputShape:"[B,64/TP,S,Ksel] + runtime metadata",output:"masked selected scores",outputShape:"[B,64/TP,S,Ksel]",formula:"Aᵢⱼ←valid_token(i,j) ? Aᵢⱼ : −∞",formulaNote:"Transformers eager/SDPA 会先把 block_indices 展开为 block_keep，再与 attention_mask 合并；部署 kernel 可直接消费 block indices 与边界元数据。",weights:[]}),
    softmax:cloneOp(attn,{id:"s-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64/TP,S,≤2048]",output:"probabilities",outputShape:"same",weights:[]}),
    pv:cloneOp(attn,{id:"s-pv",kind:"matmul",title:"P × paged V · same Top-16",input:"P + paged V",inputShape:"[B,64/TP,S,≤2048] + KV pages",output:"local heads",outputShape:"[B,S,8192/TP]",formulaNote:"kernel 按 Q×K 阶段相同的 block_indices 顺序直接读取 V pages；不物化 selected V 张量。",weights:[]}),
    oproj:cloneOp(attn,{id:"s-oproj",kind:"linear",title:"O Projection",input:"heads",inputShape:"[B,S,8192]",output:"Yattn",outputShape:"[B,S,6144]",weights:attn.weights.filter(w=>w.key.includes("o_proj"))}),
    addattn:cloneOp(combine,{id:"s-addattn",kind:"add",kicker:"DECODER LAYER · ATTENTION RESIDUAL",title:"Attention Residual Merge",summary:"在 Decoder Layer 内把 Sparse Attention 分支 Yattn 加入 residual stream Xₗ，得到更新后的 U；图中将 fused add 与紧随其后的 post-norm 分开表达。",input:"Xₗ + Yattn",inputShape:"2 × [B,S,6144]",output:"U · updated residual stream",outputShape:"[B,S,6144]",formula:"U=Xₗ+Yattn",formulaNote:"实际调用位于 DecoderLayer.forward L773：fused kernel 先执行 residual += hidden_states，再对更新后的 residual 执行 post-attention Gemma RMSNorm。",runtime:"fused_allreduce_gemma_rms_norm · attention residual",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L773–775",sourceUrl:`${CODE_URL}#L773-L775`,weights:[]}),
    postnorm:cloneOp(normBase,{id:"s-postnorm",kind:"norm",title:"Post-attn Gemma RMSNorm",summary:"输入 U 已由上游 Add 节点计算完成；此节点只执行 Gemma RMSNorm(U)，输出唯一的 Û 作为 MoE 输入。",formulaNote:"U 是上游 Add 的单一输出；本节点只计算 RMS(U) 与 (1+γpost) 缩放，不重复执行 residual add。",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL,weights:[postNorm]}),
    router:cloneOp(router,{id:"s-router",kind:"route",kicker:"FP32 ROUTER · 128 LOGITS",title:"FP32 Router Logits",summary:"每个 token 通过 GateLinear 计算 128 个 FP32 router_logits；本节点不执行 sigmoid、Top-4 或专家计算。",input:"Û",inputShape:"[B,S,6144]",output:"router_logits",outputShape:"[B,S,128]",formula:"router_logits=ÛWrouterᵀ",formulaNote:"这是 Python 层唯一显式产生的 Router 输出；下游 FusedMoE 才执行 sigmoid、correction bias、Top-4 与混合权重计算。",runtime:"GateLinear · FP32",weights:routerGateWeights}),
    experts:cloneOp(experts,{id:"s-experts",kind:"activation",kicker:"FUSED ROUTING · TOP-4 EXPERTS",title:"Fused Top-4 Routing + Experts",summary:"FusedMoE 消费 router_logits，在内部完成 sigmoid、correction bias、Top-4、归一化混合权重和 4 个专家计算。",input:"Û + router_logits",inputShape:"[B,S,6144] + [B,S,128]",output:"weighted routed output",outputShape:"[B,S,6144]",formulaNote:"先用 σ(r) 得到路由分数 s；s+b 只用于挑选 Top-4，混合权重仍取未加 bias 的 s，归一化后乘 routed_scaling_factor，最后对 4 个专家输出加权求和。expert ids 与 router weights 都是 fused kernel 内部量。",weights:experts.weights}),
    shared:cloneOp(shared,{id:"s-shared",kind:"activation",title:"Shared Expert ×1",input:"Û",inputShape:"[B,S,6144]"}),
    sum:cloneOp(combine,{id:"s-sum",kind:"add",title:"Add Routed + Shared",summary:"把路由专家分支的 Y_routed 与共享专家分支的 Y_shared 逐元素相加。",input:"Y_routed + Y_shared",inputShape:"2 × [B,S,6144]",output:"Ymoe",outputShape:"[B,S,6144]",weights:[]}),
    addout:cloneOp(combine,{id:"s-addout",kind:"add",kicker:"DECODER LAYER · MOE RESIDUAL",title:"Decoder Layer Residual Merge",summary:"在 Decoder Layer 边界把 MoE 分支 Ymoe 与 residual stream U 逻辑合并，得到 Xₗ₊₁；该 merge 延迟融合到下一层 input RMSNorm。",input:"U + Ymoe",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁ · logical next-layer input",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+Ymoe",formulaNote:"当前层 L776–778 返回 Ymoe 与 U 两条独立流；下一 Decoder Layer 在 L758–767 的 fused input RMSNorm 中执行实际 add。",runtime:"MiniMaxM3DecoderLayer boundary · deferred residual merge",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L758–778",sourceUrl:`${CODE_URL}#L758-L778`,weights:[]}),
  };
}

function GraphSurface({edges,className,children}:{edges:GraphEdge[];className:string;children:ReactNode}){
  const rootRef=useRef<HTMLDivElement>(null);
  const markerId=`graph-arrow-${useId().replace(/:/g,"")}`;
  const serializedEdges=JSON.stringify(edges);
  const edgeKey=edges.map(edge=>`${edge.from}:${edge.fromPort??"bottom"}>${edge.to}:${edge.toPort??"top"}:${edge.route??"direct"}:${edge.fanout??"single"}`).join("|");
  const [paths,setPaths]=useState<GraphPath[]>([]);
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
      if(port==="bottom-left")return [x+rect.width*.34,y+rect.height];
      if(port==="bottom-right")return [x+rect.width*.66,y+rect.height];
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
      const endpointCounts=new Map<string,number>();
      currentEdges.forEach(edge=>{
        const fromPort=edge.fromPort??"bottom"; const toPort=edge.toPort??"top";
        const fromKey=`from:${edge.from}:${fromPort}`; const toKey=`to:${edge.to}:${toPort}`;
        endpointCounts.set(fromKey,(endpointCounts.get(fromKey)??0)+1);
        endpointCounts.set(toKey,(endpointCounts.get(toKey)??0)+1);
      });
      const handledFanouts=new Set<string>();
      const next=currentEdges.flatMap(edge=>{
        const source=root.querySelector<HTMLElement>(`[data-graph-id="${edge.from}"]`);
        const target=root.querySelector<HTMLElement>(`[data-graph-id="${edge.to}"]`);
        if(!source||!target)return [];
        const fromPort=edge.fromPort??"bottom"; const toPort=edge.toPort??"top";
        const sourceRect=source.getBoundingClientRect(); const targetRect=target.getBoundingClientRect();
        const [sx,sy]=point(sourceRect,fromPort,rootRect);
        const [tx,ty]=point(targetRect,toPort,rootRect);
        if(edge.fanout){
          if(handledFanouts.has(edge.fanout))return [];
          handledFanouts.add(edge.fanout);
          const grouped=currentEdges.filter(candidate=>candidate.fanout===edge.fanout);
          const targets=grouped.flatMap(candidate=>{
            const groupedTarget=root.querySelector<HTMLElement>(`[data-graph-id="${candidate.to}"]`);
            if(!groupedTarget)return [];
            const [x,y]=point(groupedTarget.getBoundingClientRect(),candidate.toPort??"top",rootRect);
            return [{x,y}];
          });
          const tone:EdgeTone=source.classList.contains("tensor-weight")?"weight":source.classList.contains("tensor-side")?"external":"data";
          return routeGraphFanout({source:{x:sx,y:sy},targets,departure:edge.departure??72}).map(route=>({d:route.path,tone,marker:route.arrow}));
        }
        const direction=edge.route??(fromPort==="right"||fromPort==="left"||toPort==="right"||toPort==="left"?"horizontal":"vertical");
        const safeClearance=direction==="side-left"||direction==="bus-left"
          ?Math.min(24,Math.max(4,obstacleBounds.left-8))
          :direction==="side-right"||direction==="bus-right"
            ?Math.min(24,Math.max(4,rootRect.width-obstacleBounds.right-8))
            :24;
        const targetConnections=endpointCounts.get(`to:${edge.to}:${toPort}`)??1;
        const sourceConnections=endpointCounts.get(`from:${edge.from}:${fromPort}`)??1;
        const approach=edge.approach??(targetConnections>1?48:sourceConnections>1?38:toPort==="top-left"||toPort==="top-right"?30:18);
        const tone:EdgeTone=edge.route?.startsWith("side-")
          ?"residual"
          :source.classList.contains("tensor-weight")
            ?"weight"
            :source.classList.contains("tensor-side")
              ?"external"
              :"data";
        return [{d:routeGraphEdge({source:{x:sx,y:sy},target:{x:tx,y:ty},direction,obstacleBounds,clearance:safeClearance,approach,departure:edge.departure}).path,tone,marker:true}];
      });
      setPaths(next);
    };
    const observer=new ResizeObserver(()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(measure)});
    observer.observe(root);
    root.querySelectorAll<HTMLElement>("[data-graph-id]").forEach(node=>observer.observe(node));
    measure();
    frame=requestAnimationFrame(measure);
    return()=>{cancelAnimationFrame(frame);observer.disconnect()};
  },[serializedEdges]);
  const tones:EdgeTone[]=["data","weight","external","residual"];
  return <div ref={rootRef} className={`graph-surface ${className}`}>{children}<svg className="graph-connectors" aria-hidden="true"><defs>{tones.map(tone=><marker key={tone} id={`${markerId}-${tone}`} className={`edge-marker edge-marker-${tone}`} markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0.5 0.8 L 8.5 5 L 0.5 9.2 Z"/></marker>)}</defs><g className="edge-halos">{paths.map((path,index)=><path key={`${edgeKey}-halo-${index}`} className="edge-halo" d={path.d}/>)}</g><g className="edge-lines">{paths.map((path,index)=><path key={`${edgeKey}-line-${index}`} className={`edge-line edge-${path.tone}`} d={path.d} markerEnd={path.marker===false?undefined:`url(#${markerId}-${path.tone})`}/>)}</g></svg></div>;
}

function GraphPan({children}:{children:ReactNode}){
  const viewportRef=useRef<HTMLDivElement>(null);
  const contentRef=useRef<HTMLDivElement>(null);
  const dragRef=useRef({pointerId:-1,x:0,y:0,offsetX:0,offsetY:0});
  const [offset,setOffset]=useState({x:0,y:0});
  const [dragging,setDragging]=useState(false);
  const clampOffset=(x:number,y:number)=>{
    const viewport=viewportRef.current; const content=contentRef.current;
    if(!viewport||!content)return {x,y};
    const padding=28;
    const minX=Math.min(0,viewport.clientWidth-content.scrollWidth-padding);
    const minY=Math.min(0,viewport.clientHeight-content.scrollHeight-padding);
    return {x:Math.max(minX,Math.min(padding,x)),y:Math.max(minY,Math.min(0,y))};
  };
  const onPointerDown=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if((event.target as HTMLElement).closest("button,a"))return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current={pointerId:event.pointerId,x:event.clientX,y:event.clientY,offsetX:offset.x,offsetY:offset.y};
    setDragging(true);
  };
  const onPointerMove=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(dragRef.current.pointerId!==event.pointerId)return;
    setOffset(clampOffset(dragRef.current.offsetX+event.clientX-dragRef.current.x,dragRef.current.offsetY+event.clientY-dragRef.current.y));
  };
  const stopDragging=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(dragRef.current.pointerId!==event.pointerId)return;
    dragRef.current.pointerId=-1;
    setDragging(false);
  };
  return <div ref={viewportRef} className={`graph-pan-viewport ${dragging?"is-dragging":""}`} aria-label="可拖动的算子流程图" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={stopDragging} onPointerCancel={stopDragging}>
    <div ref={contentRef} className="graph-pan-content" style={{transform:`translate3d(${offset.x}px,${offset.y}px,0)`}}>{children}</div>
    <span className="graph-pan-hint">抓住空白处拖动画布</span>
  </div>;
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
    <div className="flow-row"><Tensor name="Kᵣ · V · slot_mapping" shape="KV + [Nq]"/><Arrow/><N id="cache"/><Arrow/><Tensor name="paged K · V · block_table · Top-16 ids" shape="KV pages + runtime metadata"/></div>
    <div className="flow-row attention-row"><Tensor name="Qᵣ · paged K · Top-16 ids" shape="Q + KV pages + block_indices"/><Arrow/><N id="qk"/><Arrow/><Tensor name="sparse scores" shape="[B,64,S,≤2048]"/><Arrow/><N id="scale"/><Arrow/><Tensor name="scaled scores · causal/pad layout" shape="scores + runtime metadata"/><Arrow/><N id="mask"/><Arrow/><Tensor name="masked scores" shape="same"/><Arrow/><N id="softmax"/><Arrow/><Tensor name="P" shape="same"/></div>
    <div className="flow-row"><Tensor name="P · paged V · same Top-16" shape="probabilities + KV pages"/><Arrow/><N id="pv"/><Arrow/><Tensor name="heads" shape="[B,S,8192]"/><Arrow/><N id="oproj"/><Arrow/><Tensor name="Yattn · Xₗ" shape="2 × [B,S,6144]"/><Arrow/><N id="addattn"/><Arrow/><Tensor name="U" shape="[B,S,6144]"/></div>
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
      {from:"mlp-uhat",to:"mlp-gateup",toPort:"top"},{from:"mlp-wgate",to:"mlp-gateup",toPort:"top-left"},{from:"mlp-wup",to:"mlp-gateup",toPort:"top-right"},{from:"mlp-gateup",to:"mlp-packed"},{from:"mlp-packed",to:"mlp-split"},{from:"mlp-split",to:"mlp-gate"},{from:"mlp-split",to:"mlp-up"},{from:"mlp-gate",to:"mlp-gate-act"},{from:"mlp-up",to:"mlp-up-act"},{from:"mlp-gate-act",to:"mlp-mul"},{from:"mlp-up-act",to:"mlp-mul"},{from:"mlp-mul",to:"mlp-activated"},{from:"mlp-activated",to:"mlp-down"},{from:"mlp-wdown",to:"mlp-down",fromPort:"left",toPort:"right"},{from:"mlp-down",to:"mlp-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>SWIGLU-OAI MLP · L0–2</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className="mlp-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="mlp-uhat"/><Tensor name="mlp.gate_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wgate"/><N id="gateup" graphId="mlp-gateup"/><Tensor name="mlp.up_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wup"/><Tensor name="packed gate_up⁽ʳ⁾" shape="[B,S,2H_dense/TP]" graphId="mlp-packed"/><N id="gatesplit" graphId="mlp-split"/><Tensor name="G⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-gate"/><Tensor name="U⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-up"/><button type="button" className="mini-math activation-step" data-graph-id="mlp-gate-act" aria-label="Gate 分支：先对 G⁽ʳ⁾ 做上界截断，再计算门控激活" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><b>min(G⁽ʳ⁾, C) · σ(α·min(G⁽ʳ⁾, C))</b></button><button type="button" className="mini-math activation-step" data-graph-id="mlp-up-act" aria-label="Up 分支：对 U⁽ʳ⁾ 做双边截断后加 beta" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><b>clip(U⁽ʳ⁾, −C, C) + β</b></button><button className="multiply-circle" data-graph-id="mlp-mul" aria-label="两个分支逐元素相乘" title="逐元素相乘" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.25 4.25 11.75 11.75M11.75 4.25 4.25 11.75"/></svg></button><Tensor name="Z⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-activated"/><N id="down" graphId="mlp-down"/><Tensor name="mlp.down_proj.weight⁽ʳ⁾" shape="[H,H_dense/TP]" role="weight" graphId="mlp-wdown"/><Tensor name="Yffn" shape="[B,S,H]" graphId="mlp-y"/>
    </GraphSurface></GraphPan></section>;
  }
  if(stage==="ffn"){
    const edges:GraphEdge[]=[
      {from:"moe-u",to:"moe-router",fromPort:"bottom-left",approach:34},{from:"moe-wrouter",to:"moe-router",fromPort:"right",toPort:"left"},{from:"moe-router",to:"moe-router-logits"},{from:"moe-u",to:"moe-experts",toPort:"top-right",approach:38},{from:"moe-router-logits",to:"moe-experts",toPort:"top-left",approach:28},{from:"moe-wexperts",to:"moe-experts",fromPort:"right",toPort:"left"},{from:"moe-experts",to:"moe-routed"},{from:"moe-u",to:"moe-shared",fromPort:"bottom-right",toPort:"top",approach:34},{from:"moe-wshared",to:"moe-shared",fromPort:"left",toPort:"right"},{from:"moe-shared",to:"moe-shared-out"},{from:"moe-routed",to:"moe-sum",toPort:"top-left",approach:38},{from:"moe-shared-out",to:"moe-sum",toPort:"top-right",approach:38},{from:"moe-sum",to:"moe-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>TOP-4 MOE + SHARED EXPERT · L3–59</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className="moe-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="moe-u"/>
      <div className="moe-expert-branches">
        <section className="moe-expert-branch moe-routed-branch" aria-label="Routed Expert 分支">
          <div className="moe-weighted-step moe-router-step"><Tensor name="router gate weight" shape="[E,H]" role="weight" graphId="moe-wrouter"/><N id="router" graphId="moe-router"/></div>
          <Tensor name="router logits" shape="[B,S,128]" graphId="moe-router-logits"/>
          <div className="moe-weighted-step moe-routed-step"><Tensor name="routed expert weights · correction bias" shape="E × expert weights · [E]" role="weight" graphId="moe-wexperts"/><N id="experts" graphId="moe-experts"/></div>
          <Tensor name="weighted routed output" shape="[B,S,H]" graphId="moe-routed"/>
        </section>
        <section className="moe-expert-branch moe-shared-branch" aria-label="Shared Expert 分支">
          <div className="moe-weighted-step moe-shared-step"><N id="shared" graphId="moe-shared"/><Tensor name="shared expert weights ×3" shape="gate / up / down" role="weight" graphId="moe-wshared"/></div>
          <Tensor name="shared output" shape="[B,S,H]" graphId="moe-shared-out"/>
        </section>
      </div>
      <N id="sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[B,S,H]" graphId="moe-y"/>
    </GraphSurface></GraphPan></section>;
  }
  const dense=type==="dense";
  const ids=dense?{project:"qkv",split:"split",qnorm:"qnorm",knorm:"knorm",ropeq:"ropeq",ropek:"ropek"}:{project:"packed",split:"split",qnorm:"mainnorm",knorm:"mainnorm",ropeq:"rope",ropek:"rope"};
  const edges:GraphEdge[]=[
    {from:"attn-x",to:"attn-project"},{from:"attn-project",to:"attn-packed"},{from:"attn-packed",to:"attn-split"},{from:"attn-split",to:"attn-q",fanout:"attn-five-way",departure:64},{from:"attn-split",to:"attn-k",fanout:"attn-five-way",departure:64},{from:"attn-split",to:"attn-v",fanout:"attn-five-way",departure:64},{from:"attn-q",to:"attn-qnorm",toPort:"top-left",approach:38},{from:"attn-wq",to:"attn-qnorm",toPort:"top-right",approach:38},{from:"attn-qnorm",to:"attn-qt"},{from:"attn-qt",to:"attn-qrope",toPort:"top-left",approach:38},{from:"attn-posq",to:"attn-qrope",toPort:"top-right",approach:38},{from:"attn-qrope",to:"attn-qr"},{from:"attn-k",to:"attn-knorm",toPort:"top-left",approach:38},{from:"attn-wk",to:"attn-knorm",toPort:"top-right",approach:38},{from:"attn-knorm",to:"attn-kt"},{from:"attn-kt",to:"attn-krope",toPort:"top-left",approach:38},{from:"attn-posk",to:"attn-krope",toPort:"top-right",approach:38},{from:"attn-krope",to:"attn-kr"},{from:"attn-kr",to:"attn-cache",toPort:"top-left",approach:72},{from:"attn-v",to:"attn-cache",toPort:"top-right",approach:72},{from:"attn-cache-meta",to:"attn-cache",fromPort:"left",toPort:"right"},{from:"attn-cache",to:"attn-paged-k",approach:54},{from:"attn-cache",to:"attn-paged-v",approach:54},{from:"attn-qr",to:"attn-qk",toPort:"top-left",approach:18},{from:"attn-paged-k",to:"attn-qk",toPort:dense?"top-right":"top",approach:28},{from:"attn-qk",to:"attn-scale"},{from:"attn-scale",to:"attn-scaled"},{from:"attn-scaled",to:"attn-mask"},{from:"attn-bounds",to:"attn-mask",fromPort:"left",toPort:"right"},{from:"attn-mask",to:"attn-softmax"},{from:"attn-softmax",to:"attn-p"},{from:"attn-p",to:"attn-pv",toPort:"top-left",approach:18},{from:"attn-paged-v",to:"attn-pv",toPort:"top-right",route:"bus-left",approach:28,departure:54},{from:"attn-pv",to:"attn-heads"},{from:"attn-heads",to:"attn-oproj"},{from:"attn-oproj",to:"attn-y"},
    ...(!dense?([{from:"attn-split",to:"attn-qidx",fanout:"attn-five-way",departure:64},{from:"attn-split",to:"attn-kidx",fanout:"attn-five-way",departure:64},{from:"attn-qidx",to:"attn-idxnorm",toPort:"top-left",approach:34},{from:"attn-kidx",to:"attn-idxnorm",toPort:"top-right",approach:34},{from:"attn-idxnorm",to:"attn-idxquery",fromPort:"bottom-left",approach:30},{from:"attn-idxnorm",to:"attn-idxcache",fromPort:"bottom-right",approach:30},{from:"attn-idxslots",to:"attn-idxcache",fromPort:"left",toPort:"right"},{from:"attn-idxquery",to:"attn-idxscore",toPort:"top-left",approach:30},{from:"attn-idxcache",to:"attn-idxscore",toPort:"top-right",approach:30},{from:"attn-idxscore",to:"attn-idxmask"},{from:"attn-idxbounds",to:"attn-idxmask",fromPort:"left",toPort:"right"},{from:"attn-idxmask",to:"attn-blockmax"},{from:"attn-blockmax",to:"attn-topk"},{from:"attn-topk",to:"attn-topids"},{from:"attn-topids",to:"attn-qk",toPort:"top-right",approach:28}] satisfies GraphEdge[]) : []),
  ];
  return <section className="stage-zoom lesson-zoom attention-lesson"><header><span>{dense?"GQA + PARTIAL ROPE · L0–2":"MINIMAX SPARSE ATTENTION + PARTIAL ROPE · L3–59"}</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className={`attention-flowchart connected-attention-graph ${dense?"dense-attention":"sparse-attention"}`} edges={edges}>
    <div className="compact-chain"><Tensor name="X̂" shape="[B,S,H]" graphId="attn-x"/><N id={ids.project} graphId="attn-project"/><Tensor name="packed" shape={dense?"[B,S,9216]":"[B,S,9856]"} graphId="attn-packed"/><N id={ids.split} graphId="attn-split"/></div>
    <div className={`attention-branches ${dense?"dense":""}`}>{!dense&&<div className="index-ribbon"><header className="index-ribbon-label">LIGHTNING INDEXER · per query / KV group</header><div className="multi-source"><Tensor name="Qidx" shape="[B,S,4,128]" graphId="attn-qidx"/><Tensor name="Kidx" shape="[B,T,1,128]" graphId="attn-kidx"/></div><N id="idxnorm" graphId="attn-idxnorm"/><Tensor name="Index Q query" shape="[B,4,S,128]" graphId="attn-idxquery"/><N id="idxcache" graphId="attn-idxcache"/><Tensor name="index slot_mapping" shape="[Nq]" role="side" graphId="attn-idxslots"/><N id="idxscore" graphId="attn-idxscore"/><Tensor name="position_ids · future bound" shape="[B,S]" role="side" graphId="attn-idxbounds"/><N id="idxmask" graphId="attn-idxmask"/><N id="blockmax" graphId="attn-blockmax"/><N id="topk" graphId="attn-topk"/><Tensor name="block_indices · Top-16" shape="[B,4,S,16]" graphId="attn-topids"/></div>}
      <div className="attention-data-path"><div className="qkv-lanes"><section><header>Q PATH</header><IW id={ids.qnorm} inputName="Q" inputShape="[B,Nₕ,S,Dₕ]" inputGraphId="attn-q" graphId="attn-qnorm" weightGraphId="attn-wq"/><div className="two-source"><Tensor name="Q̃" shape="same" graphId="attn-qt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posq"/></div><N id={ids.ropeq} graphId="attn-qrope"/><Tensor name="Qᵣ" shape="[B,Nₕ,S,Dₕ]" graphId="attn-qr"/></section><section><header>K PATH</header><IW id={ids.knorm} inputName="K" inputShape="[B,Nₖᵥ,S,Dₕ]" weightIndex={dense?0:1} inputGraphId="attn-k" graphId="attn-knorm" weightGraphId="attn-wk"/><div className="two-source"><Tensor name="K̃" shape="same" graphId="attn-kt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posk"/></div><N id={ids.ropek} graphId="attn-krope"/><Tensor name="Kᵣ" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-kr"/></section><section><header>V PATH</header><Tensor name="V" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-v"/></section></div><div className="kv-cache-flow"><Tensor name="slot_mapping · block_table" shape="runtime" role="side" graphId="attn-cache-meta"/><N id="cache" graphId="attn-cache"/><div className="two-source"><Tensor name="paged K" shape="KV pages" graphId="attn-paged-k"/><Tensor name="paged V" shape="KV pages" graphId="attn-paged-v"/></div></div></div></div>
    <div className="score-pipeline"><header className="score-pipeline-label">ATTENTION SCORE PIPELINE · selected blocks 内计算概率 P</header><N id="qk" graphId="attn-qk"/><N id="scale" graphId="attn-scale"/><Tensor name="scaled scores" shape={dense?"[B,Nₕ,S,T]":"[B,Nₕ,S,Ksel]"} graphId="attn-scaled"/><Tensor name={dense?"causal / pad bounds":"token causal / pad bounds"} shape="runtime metadata" role="side" graphId="attn-bounds"/><N id="mask" graphId="attn-mask"/><N id="softmax" graphId="attn-softmax"/><Tensor name="P" shape={dense?"[B,Nₕ,S,T]":"[B,Nₕ,S,Ksel]"} graphId="attn-p"/></div>
    <div className="context-pipeline"><N id="pv" graphId="attn-pv"/><Tensor name="heads" shape="[B,S,Nₕ·Dₕ]" graphId="attn-heads"/><N id="oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[B,S,H]" graphId="attn-y"/></div>
  </GraphSurface></GraphPan></section>;
}

function DecoderDiagram({type,g,active,expanded,onExpand,onHover,onLeave,onSelect}:{type:LayerType;g:Record<string,OpNode>;active:string;expanded:ExpandedStage;onExpand:(stage:ExpandedStage)=>void;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const IW=({id,inputName,inputShape,inputGraphId,graphId,weightGraphId}:{id:string;inputName:string;inputShape:string;inputGraphId:string;graphId:string;weightGraphId:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId}/>;
  const A=({id,graphId}:{id:string;graphId:string})=><AddCircle node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const edges:GraphEdge[]=[{from:"main-x",to:"main-norm",toPort:"top-left",approach:48},{from:"main-win",to:"main-norm",toPort:"top-right",approach:48},{from:"main-norm",to:"main-attn"},{from:"main-attn",to:"main-add1"},{from:"main-x",to:"main-add1",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add1",to:"main-u",approach:52},{from:"main-u",to:"main-post",toPort:"top-left",approach:48},{from:"main-wpost",to:"main-post",toPort:"top-right",approach:48},{from:"main-post",to:"main-ffn"},{from:"main-ffn",to:"main-add2"},{from:"main-u",to:"main-add2",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add2",to:"main-out",approach:44}];
  return <div className={`decoder-workbench ${expanded?"has-zoom":""}`}>{!expanded&&<GraphPan><GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <IW id="norm" inputName="Xₗ · hidden_states / residual stream" inputShape="[B,S,H]" inputGraphId="main-x" graphId="main-norm" weightGraphId="main-win"/><button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={()=>onExpand(expanded==="attention"?null:"attention")}><small>点击展开</small><b>{type==="dense"?"GQA + Partial RoPE":"MiniMax Sparse Attention + Partial RoPE"}</b></button><A id={type==="dense"?"add1":"addattn"} graphId="main-add1"/><IW id="postnorm" inputName="U · updated residual stream" inputShape="[B,S,H]" inputGraphId="main-u" graphId="main-post" weightGraphId="main-wpost"/><button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={()=>onExpand(expanded==="ffn"?null:"ffn")}><small>点击展开</small><b>{type==="dense"?"SwiGLU-OAI MLP":"Top-4 MoE + Shared Expert"}</b></button><A id={type==="dense"?"add2":"addout"} graphId="main-add2"/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,H]" graphId="main-out"/>
  </GraphSurface></GraphPan>}{expanded&&<StageZoom type={type} stage={expanded} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={()=>onExpand(null)}/>}</div>;
}
/* eslint-enable react-hooks/static-components */

function LayerNavigator({type,onChange}:{type:LayerType;onChange:(type:LayerType)=>void}){
  return <div className="layer-nav layer-type-nav"><div className="layer-nav-head"><b>{type==="dense"?"GQA + SwiGLU-OAI MLP":"MiniMax Sparse Attention + MoE"}</b></div><div className="layer-type-options"><button className={type==="dense"?"active dense":"dense"} onClick={()=>onChange("dense")}><span>L0–L2</span><b>GQA + Partial RoPE + SwiGLU-OAI MLP</b><small>3 层共享同一实现</small></button><button className={type==="sparse"?"active sparse":"sparse"} onClick={()=>onChange("sparse")}><span>L3–L59</span><b>MiniMax Sparse Attention + Partial RoPE + Top-4 MoE</b><small>57 层共享同一实现</small></button></div></div>;
}

function LatexExpression({formula,label,className=""}:{formula:string;label:string;className?:string}){
  const html=katex.renderToString(formula,{displayMode:true,throwOnError:false,strict:"ignore",output:"htmlAndMathml"});
  return <div className={`latex-render ${className}`.trim()} aria-label={label} dangerouslySetInnerHTML={{__html:html}}/>;
}

function LatexFormula({node}:{node:OpNode}){
  const formula=node.latex??SIMPLE_FORMULA[node.kind]??String.raw`y=f(x)`;
  return <LatexExpression formula={formula} label={`${node.title} 简化公式`}/>;
}

function symbolicShape(shape:string){
  return shape
    .replaceAll("[B,64/TP,S,128]","[B,Nₕ/TP,S,Dₕ]")
    .replaceAll("[B,max(1,4/TP),T,128]","[B,Nₖᵥ,rank,T,Dₕ]")
    .replaceAll("[B,max(1,4/TP),S,128]","[B,N_idx,rank,S,D_idx]")
    .replaceAll("[B,64/TP,S,T]","[B,Nₕ/TP,S,T]")
    .replaceAll("[B,64/TP,S,Ksel]","[B,Nₕ/TP,S,Ksel]")
    .replaceAll("[B,64/TP,S,≤2048]","[B,Nₕ/TP,S,Ksel]")
    .replaceAll("[B,S,8192/TP]","[B,S,Nₕ·Dₕ/TP]")
    .replaceAll("[B,64,S,128]","[B,Nₕ,S,Dₕ]")
    .replaceAll("[B,64,S,T]","[B,Nₕ,S,T]")
    .replaceAll("[B,4,S,128]","[B,Nₖᵥ,S,Dₕ]")
    .replaceAll("[B,4,T,128]","[B,Nₖᵥ,T,Dₕ]")
    .replaceAll("[B,S,24576/TP]","[B,S,2H_dense/TP]")
    .replaceAll("[B,S,12288/TP]","[B,S,H_dense/TP]")
    .replaceAll("[B,S,24576]","[B,S,2H_dense]")
    .replaceAll("[B,S,12288]","[B,S,H_dense]")
    .replaceAll("[B,S,6144]","[B,S,H]")
    .replaceAll("[B,S,8192]","[B,S,Nₕ·Dₕ]")
    .replaceAll("[B,S,9216]","[B,S,(Nₕ+2Nₖᵥ)·Dₕ]")
    .replaceAll("[B,S,9856]","[B,S,QKV+Index]")
    .replaceAll("24576/TP","2H_dense/TP")
    .replaceAll("12288/TP","H_dense/TP")
    .replaceAll("8192/TP","Nₕ·Dₕ/TP")
    .replaceAll("max(1,4/TP)","Nₖᵥ,rank")
    .replaceAll("64/TP","Nₕ/TP")
    .replaceAll("24576","2H_dense")
    .replaceAll("12288","H_dense")
    .replaceAll("6144","H")
    .replaceAll("8192","Nₕ·Dₕ")
    .replaceAll("9216","(Nₕ+2Nₖᵥ)·Dₕ")
    .replaceAll("9856","QKV+Index")
    .replaceAll("200064","V");
}

function ShapeRows({shape}:{shape:string}){
  return <div className="shape-rows"><span><i>符号</i><code title={symbolicShape(shape)}>{symbolicShape(shape)}</code></span><span><i>实际</i><code title={shape}>{shape}</code></span></div>;
}

function bindingsFor(node:OpNode):IoBinding[]{
  const dataInputs=INPUT_OVERRIDES[node.id]??[{kind:node.kind==="io"?"external":"upstream",label:node.input,shape:node.inputShape,from:node.kind==="io"?"模型调用方 / runtime":"图中紧邻的上游模块输出"}];
  const weightInputs=node.weights.map(weight=>{
    const tpShape=node.id==="d-gateup"?weight.shape.replace("[12288,6144]","[12288/TP,6144]"):node.id==="d-down"?weight.shape.replace("[6144,12288]","[6144,12288/TP]"):null;
    return {kind:"weight" as const,label:weight.key,shape:tpShape?`${weight.dtype} · TP shard ${tpShape}`:`${weight.dtype} · ${weight.shape}`,from:weight.runtime?`checkpoint → ${weight.runtime}`:`checkpoint · ${weight.shard}`,note:weight.params?`${weight.params} parameters`:undefined};
  });
  return [...dataInputs,...weightInputs];
}

function IoView({node}:{node:OpNode}){
  const bindings=bindingsFor(node);
  const labels:Record<BindingKind,string>={upstream:"上游张量",external:"外部输入",weight:"权重输入"};
  return <div className="io-binding-view"><section className="binding-list"><header><span>INPUT BINDINGS</span><b>{bindings.length} 路输入</b></header>{bindings.map((binding,index)=><article className={`binding binding-${binding.kind}`} key={`${binding.kind}-${binding.label}-${index}`}><div><span>{labels[binding.kind]}</span></div><b>{binding.label}</b><ShapeRows shape={binding.shape}/><p><i>来自</i>{binding.from}</p>{binding.note&&<small>{binding.note}</small>}</article>)}</section><section className="output-binding"><header><span>OUTPUT BINDING</span><b>1 路产物</b></header><article><div><span>计算产物</span></div><b>{node.output}</b><ShapeRows shape={node.outputShape}/><p><i>送往</i>{NEXT_BY_ID[node.id]??"图中下游模块"}</p></article></section></div>;
}

function codeSourceLabel(section:CodeSection){
  return section.url?.includes("github.com/huggingface/transformers")?"Transformers":"vLLM";
}

function CodeView({node}:{node:OpNode}){
  const sections=node.codeSections??[];
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {VLLM_COMMIT.slice(0,7)}</span><b>{node.source}</b><i>↗</i></a>
    {sections.length?<section className="code-call-chain"><header><span>IMPLEMENTATION TRACE</span></header>{sections.map((section,index)=>{const source=codeSourceLabel(section);return <article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><div className="code-section-kicker"><span>{section.stage}</span><span className={`code-source-tag source-${source.toLowerCase()}`}>{source}</span></div><b>{section.title}</b><small>{section.location}</small></div>{section.url&&<a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article>})}</section>:<div className="code-empty"><b>此节点没有独立 forward</b><p>它由所在模块的 forward 调度，或只是一个数学拆解步骤。</p></div>}
  </div>;
}

type StageOverview = { kicker:string; title:string; summary:string; flow:string; formula:string; formulaNote?:string; notes:string[]; parameters:readonly (readonly [string,string,string])[] };

function stageOverview(type:LayerType,stage:Exclude<ExpandedStage,null>):StageOverview{
  if(type==="dense"&&stage==="ffn")return {
    kicker:"DENSE FFN · L0–L2",title:"SwiGLU-OAI MLP",summary:"逐 token 扩维、门控，再投回 hidden size。",
    flow:"Û → TP-local Gate + Up Projection → Split → SwiGLU-OAI → RowParallel Down Projection → Yffn",
    formula:"G⁽ʳ⁾ = Û (W_gate⁽ʳ⁾)ᵀ\nU⁽ʳ⁾ = Û (W_up⁽ʳ⁾)ᵀ\nḠ⁽ʳ⁾ = min(G⁽ʳ⁾, c)\nŪ⁽ʳ⁾ = clip(U⁽ʳ⁾, −c, c)\nZ⁽ʳ⁾ = Ḡ⁽ʳ⁾ ⊙ σ(αḠ⁽ʳ⁾) ⊙ (Ū⁽ʳ⁾ + β)\nYffn = Σᵣ Z⁽ʳ⁾ (W_down⁽ʳ⁾)ᵀ",
    formulaNote:"r 表示 TP rank；G⁽ʳ⁾、U⁽ʳ⁾、Z⁽ʳ⁾ 的最后一维都是 H_dense/TP。σ 表示 sigmoid，⊙ 表示逐元素相乘；Down Projection 最后归并各 rank 的部分结果。",
    notes:["Gate 与 Up 权重和同一个输入 Û 直接进入 fused projection。","投影结果沿最后一维 Split；MLP 不混合不同 token。"],
    parameters:[["H","6144","hidden_size"],["H_dense","12288","dense_intermediate_size"],["α","1.702","swiglu_alpha"],["β","1.0","swiglu_beta"],["c","7.0","swiglu_limit"]],
  };
  if(stage==="attention")return type==="dense"?{
    kicker:"DENSE ATTENTION · L0–L2",title:"GQA + Partial RoPE",summary:"用 Q 检索完整可见 KV 历史，再按概率聚合 V。",
    flow:"QKV Projection → Split → Q/K Norm + Partial RoPE → Attention → O Projection",
    formula:"Attention(Q,K,V)=softmax(QKᵀ/√Dₕ + mask)V",
    notes:["Q / K / V 从 Split 节点分叉，并在 Attention 算子中汇合。","Partial RoPE 作用于 Q/K 每个 head 的前 64 维。"],
    parameters:[["Nₕ","64","query heads"],["Nₖᵥ","4","KV heads"],["Dₕ","128","head_dim"],["Dᵣ","64","rotary_dim"]],
  }:{
    kicker:"SPARSE ATTENTION · L3–L59",title:"MiniMax Sparse Attention + Partial RoPE",summary:"Indexer 先选 Top-16 KV blocks，主 Attention 再在候选页内计算。",
    flow:"QKV + Index Projection → Indexer → Top-16 Blocks → Sparse Attention → O Projection",
    formula:"Attention(Q,Kselected,Vselected)=softmax(QKselectedᵀ/√Dₕ + mask)Vselected",
    notes:["Top-16 只缩小候选 KV blocks，不替代 causal / padding mask。","Indexer 与主 Attention 通过明确的 KV page 边连接。"],
    parameters:[["K_block","16","selected blocks"],["B_block","128","tokens / block"],["N_idx","4","index heads"],["D_idx","128","index_dim"]],
  };
  return {
    kicker:"SPARSE FFN · L3–L59",title:"Top-4 MoE + Shared Expert",summary:"每个 token 进入 4 个路由专家，同时经过 1 个共享专家。",
    flow:"路由选择：Û → FP32 Router → router_logits\n路由专家：Û + router_logits → Fused Top-4 Routing + Experts → Y_routed\n共享专家：Û → Shared Expert → Y_shared\n最终合并：Y_routed + Y_shared → Add → Y_moe",
    formula:"Y_routed=Σₑ ŵₑEₑ(Û)\nY_shared=E_shared(Û)\nY_moe=Y_routed+Y_shared",
    notes:["Router 在 Python 层只产生 [B,S,128] 的 router_logits。","expert ids 与 router weights 由 FusedMoE 内部计算；Shared Expert 不经过 Top-K。"],
    parameters:[["E","128","routed experts"],["K","4","experts / token"],["E_shared","1","shared expert"],["H_expert","3072","expert width"]],
  };
}

function StageOverviewPanel({type,stage}:{type:LayerType;stage:Exclude<ExpandedStage,null>}){
  const overview=stageOverview(type,stage);
  return <aside className="detail-panel stage-overview-panel"><header className="stage-overview-header"><span>{overview.kicker}</span><h2>{overview.title}</h2><p>{overview.summary}</p></header><div className="stage-overview-body"><section className="stage-flow-section"><span>数据流</span><code>{overview.flow}</code></section><section className="stage-formula-section"><span>计算语义</span><code>{overview.formula}</code>{overview.formulaNote&&<p>{overview.formulaNote}</p>}</section><section className="stage-parameter-section"><span>关键参数</span><div className="stage-parameters">{overview.parameters.map(([symbol,value,source])=><article key={symbol}><b>{symbol}</b><strong>{value}</strong><small>{source}</small></article>)}</div></section><section><span>边界说明</span>{overview.notes.map(note=><p key={note}>{note}</p>)}</section></div><footer>展开图说明 · 点击算子查看独立详情</footer></aside>;
}

function DetailPanel({node,tab,setTab,pinned,onClear,expanded,layerType}:{node:OpNode|null;tab:Tab;setTab:(t:Tab)=>void;pinned:boolean;onClear:()=>void;expanded:ExpandedStage;layerType:LayerType}){
  const tabs:[Tab,string][]=[["io","I/O + 权重"],["formula","公式"],["code","代码"]];
  if(!node&&expanded)return <StageOverviewPanel type={layerType} stage={expanded}/>;
  if(!node)return <aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后，可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;
  const formulaSteps=FORMULA_STEPS_BY_ID[node.id];
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div>{pinned?<button className="unpin-button" aria-label="取消固定" title="取消固定" onClick={onClear}>×</button>:<i className={`kind-dot op-${node.kind}`}/>}<p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div key={tab} className={`detail-content detail-${tab}`}>
    {tab==="io"&&<IoView node={node}/>}
    {tab==="formula"&&<div className="formula-view"><span>作用</span><div className="formula-purpose">{node.summary}</div><span>实际公式</span><LatexFormula node={node}/>{formulaSteps?<div className="formula-steps">{formulaSteps.map(step=><article className="formula-step" key={step.title}><header>{step.title}</header><LatexExpression formula={step.formula} label={`${step.title} 公式`} className="formula-step-math"/><p>{step.explanation}</p></article>)}</div>:<><div className="formula-implementation"><b>一句话解释</b><p>{node.formulaNote??FORMULA_NOTE[node.kind]}</p></div><div className="formula-terms">{formulaTerms(node).map(([symbol,meaning])=><span key={symbol}><b>{symbol}</b>{meaning}</span>)}</div></>}</div>}
    {tab==="code"&&<CodeView node={node}/>}
    </div><footer>vLLM @ {VLLM_COMMIT.slice(0,7)} · official safetensors</footer></aside>;
}

function HelpModal({onClose}:{onClose:()=>void}){
  const [activeGroup,setActiveGroup]=useState(0);
  const group=CONFIG_GROUPS[activeGroup];
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal reference-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="完整 config.json 参数与符号"><header><div><span>CONFIG REFERENCE</span><h2>完整 config.json · 参数与符号</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><nav className="config-tabs" role="tablist" aria-label="选择 config.json 分组">{CONFIG_GROUPS.map((item,index)=><button key={item.title} role="tab" aria-selected={activeGroup===index} className={activeGroup===index?"active":""} onClick={()=>setActiveGroup(index)}>{item.title}</button>)}</nav><div className="config-reference" role="tabpanel"><section><h3>{group.title}</h3><table><colgroup><col className="config-key-column"/><col className="config-symbol-column"/><col/></colgroup><thead><tr><th>参数</th><th>符号</th><th>值</th></tr></thead><tbody>{group.rows.map(([key,value])=><tr key={key}><th scope="row">{key}</th><td className="config-symbol"><code>{configSymbol(group.title,key)}</code></td><td>{value}</td></tr>)}</tbody></table></section></div></section></div>;
}

export default function Home(){
  const [layerType,setLayerType]=useState<LayerType>("sparse"); const [expanded,setExpanded]=useState<ExpandedStage>(null); const [tab,setTab]=useState<Tab>("io"); const [dark,setDark]=useState(false); const [help,setHelp]=useState(false);
  const layer=layerType==="dense"?2:3; const graph=layerType==="dense"?denseGraph(layer):sparseGraph(layer); const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null}); const active=detail.pinned??detail.hovered;
  const updateDetail=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
  const changeLayerType=(next:LayerType)=>{setLayerType(next);setExpanded(null);updateDetail({type:"clear"})};
  return <main className={`atlas-app ${dark?"dark":""} ${expanded?"stage-expanded":""}`}><header className="app-header">
    <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>Model Atlas</b></div></div>
    <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value="minimax-m3" onChange={()=>undefined}>{MODEL_REGISTRY.map(m=><option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}</option>)}</select></label>
    <nav className="resource-links"><a href={CODE_URL} target="_blank" rel="noreferrer"><b>CODE ↗</b><small>vLLM @ {VLLM_COMMIT.slice(0,7)}</small></a><a href={TRANSFORMERS_MOE_URL} target="_blank" rel="noreferrer"><b>TRANSFORMERS ↗</b><small>MiniMax-M3 readable reference</small></a><a href={WEIGHTS_URL} target="_blank" rel="noreferrer"><b>WEIGHTS ↗</b><small>Hugging Face · 59 shards</small></a></nav>
    <div className="model-facts"><span><b>428B</b><small>模型总参数量</small></span><span><b>23B</b><small>每 token 激活参数</small></span><span><b>1M</b><small>最大上下文 token</small></span><span><b>869 GB</b><small>BF16 checkpoint</small></span></div>
    <button className="help-button" onClick={()=>setHelp(true)} aria-label="查看参数和符号说明">?</button><button className="theme-button" onClick={()=>setDark(v=>!v)} aria-label="切换明暗主题">{dark?"☀":"☾"}</button>
  </header><div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><div className="model-step">Text / Vision Inputs</div><Arrow/><div className="model-step">Embedding Fusion <code>[B,S,H]</code></div><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>L0–2 · GQA + Partial RoPE + SwiGLU-OAI MLP</span><span><i className="sparse"/>L3–59 · MiniMax Sparse Attention + Partial RoPE + MoE</span></div><Arrow/><div className="model-step">Final Gemma RMSNorm <code>[B,S,H]</code></div><Arrow/><div className="model-step">LM Head <code>[B,S,V]</code></div></div>
    <LayerNavigator type={layerType} onChange={changeLayerType}/>
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按结构类型展示</span><h1>{layerType==="dense"?"GQA + Partial RoPE + SwiGLU-OAI MLP · L0–L2 同构":"MiniMax Sparse Attention + Partial RoPE + Top-4 MoE · L3–L59 同构"}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span title="颜色区分算子类型"><i className="operator-swatch"/>OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><DecoderDiagram type={layerType} g={graph} active={active?.id??""} expanded={expanded} onExpand={setExpanded} onHover={node=>updateDetail({type:"hover",node})} onLeave={()=>updateDetail({type:"leave"})} onSelect={node=>{updateDetail({type:"pin",node});setTab("io")}}/></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>updateDetail({type:"clear"})} expanded={expanded} layerType={layerType}/></div>{help&&<HelpModal onClose={()=>setHelp(false)}/>}</main>;
}
