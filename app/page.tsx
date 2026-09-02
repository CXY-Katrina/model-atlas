import { useState } from "react";
import { nextDetailState, type DetailEvent, type DetailState } from "./detail-selection";
import { denseNodes, layerShard, sparseNodes } from "./model-data";
import {
  AddCircle,
  Arrow,
  FORMULA_NOTE_DEFAULT,
  GraphSurface,
  InputWeightedOp,
  LatexFormula,
  Op,
  RuntimeIORail,
  Tensor,
  type BindingKind,
  type CodeDetail,
  type CodeSection,
  type CodeSymbol,
  type ConfigGroup,
  type EdgePort,
  type ExpandedStage,
  type FormulaTerm,
  type GraphEdge,
  type IoBinding,
  type Node,
  type OpKind,
  type OpNode,
  type StageOverview,
  type Tab,
  type Weight,
} from "./atlas-shared";
import type { ModelModule } from "./models/model-module";

type LayerType = "dense" | "sparse";

const VLLM_COMMIT = "edd4c8176cfd98ece8a29beda574378c42971967";
const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3";
const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
const ACTIVATION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/activation.py`;
const LINEAR_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/linear.py`;
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

const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"把每个 token 的向量缩放到稳定范围；shape 不变。",linear:"W 是当前模块绑定的权重；最后一维由 W 的输出维决定。",split:"只切分最后一维，不做数值计算，也没有权重。",rope:"position 决定旋转角度；这里只旋转每个 head 的前 64 维。",matmul:"沿共同的 head_dim 相乘并求和。",scale:"dₕ=128；缩放避免 score 随维度增大。",mask:"不可见位置加 −∞，softmax 后概率变为 0。",softmax:"把每行 score 转为和为 1 的概率。",activation:"g 是 gate，u 是 up；实际实现还包含 limit=7 的截断。",route:"只选择去哪里计算；Top-K 本身不生成 expert 输出。",cache:"slot 与 block table 由 runtime 提供，权重不参与。",add:"残差支路与计算支路逐元素相加，shape 必须一致。",io:"这是数据入口或运行时元数据，不执行可训练计算。",
};

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
  "s-idxnorm":[["Qidx / Kidx","Indexer Q/K"],["Q̃idx / K̃idx","归一化后的 Indexer Q/K"],["Didx","index_dim = 128"],["ε","10⁻⁶"]],
  "d-gateup":[["Û","归一化输入 · [B,S,H]"],["Wgate⁽ʳ⁾","当前 TP rank 的 gate 权重"],["Wup⁽ʳ⁾","当前 TP rank 的 up 权重"],["G⁽ʳ⁾","当前 rank 的 gate 投影"],["U⁽ʳ⁾","当前 rank 的 up 投影"],["H","hidden_size = 6144"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-gatesplit":[["X⁽ʳ⁾","当前 rank 的 packed gate_up"],["G⁽ʳ⁾","前 H_dense/TP 个通道"],["U⁽ʳ⁾","后 H_dense/TP 个通道"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-swiglu":[["G⁽ʳ⁾","当前 rank 的 gate 分片"],["U⁽ʳ⁾","当前 rank 的 up 分片"],["Ḡ⁽ʳ⁾ / Ū⁽ʳ⁾","clamp 后的两个分片"],["α","swiglu_alpha = 1.702"],["β","swiglu_beta = 1.0"],["c","swiglu_limit = 7.0"],["Z⁽ʳ⁾","当前 rank 的激活输出"]],
  "d-qk":[["Qᵣ","当前 rank 的 rotated Q"],["Kᵣ","当前 rank 的 rotated / visible K"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["Dₕ","head_dim = 128"],["A","当前 rank 的 attention scores"]],
  "d-pv":[["P","当前 rank 的 attention probability"],["V","当前 rank 的 visible V"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["O","当前 rank 的 context heads"]],
  "s-idxscore":[["Q̃idx","当前 rank 的 normalized Index Q"],["K̃idx","共享 / 复制的 normalized Index K"],["N_idx,rank","max(1,4/TP)"],["D_idx","index_dim = 128"],["Sidx","当前 rank 的 token scores"]],
  "s-qk":[["Qᵣ","当前 rank 的 rotated Q"],["K𝒮","当前 rank 选中的 K pages"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Ksel","最多 16×128 个候选 token"],["A","当前 rank 的 sparse scores"]],
  "s-pv":[["P","当前 rank 的 sparse probability"],["V𝒮","当前 rank 选中的 V pages"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["O","当前 rank 的 context heads · 8192/TP"]],
  "s-topk":[["B","block scores"],["λ_local","local priority = 10²⁹"],["λ_init","init priority = 10³⁰"],["K_block","sparse_topk_blocks = 16"],["𝒮","选中的逻辑 block ids"]],
  "s-router":[["E","num_local_experts = 128"],["K","num_experts_per_tok = 4"],["s_route","routed_scaling_factor = 2.0"],["b","e_score_correction_bias"],["ŵₑ","归一化并缩放的 expert weight"]],
  "s-experts":[["gₑ / vₑ","expert gate / up 分支"],["c","swiglu_limit = 7.0"],["α","swiglu_alpha = 1.702"],["β","swiglu_beta = 1.0"],["H_expert","intermediate_size = 3072"],["Eₑ(u)","第 e 个 expert 输出"]],
  "d-qkv":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["Dₕ","head_dim = 128"],["TP","tensor parallel size"],["Z","当前 rank 的 packed QKV"]],
  "d-split":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["Q / K / V","当前 rank 的三个输出"]],
  "d-ropeq":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 5000000"],["p","token position"],["Qᵣ","旋转后的 Q"]],
  "d-ropek":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 5000000"],["p","token position"],["Kᵣ","旋转后的 K"]],
  "d-oproj":[["Nₕ","num_attention_heads = 64"],["Dₕ","head_dim = 128"],["H","hidden_size = 6144"],["TP","tensor parallel size"],["W_O","o_proj.weight"]],
  "d-down":[["Z⁽ʳ⁾","当前 TP rank 的激活输出"],["Wdown⁽ʳ⁾","当前 rank 的 down 权重"],["H_dense","dense_intermediate_size = 12288"],["H","hidden_size = 6144"],["TP","tensor parallel size"]],
  "s-packed":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["N_idx","sparse_num_index_heads = 4"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"],["TP","tensor parallel size"]],
  "s-split":[["Nₕ/TP","每 rank query heads = 64/TP"],["Nₖᵥ,rank","max(1,4/TP)"],["N_idx,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"]],
  "s-blockmax":[["B_block","sparse_block_size = 128"],["Sidx","index token scores"],["B","block scores"]],
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
  "s-select":[["𝒮","Top-K 选中的逻辑 block ids"],["block_table","逻辑块到物理页映射"],["𝒫","选中的物理 pages"],["K𝒮 / V𝒮","从 pages gather 的 KV"]],
  "s-scale":[["A","未缩放 sparse scores"],["Ā","缩放后的 sparse scores"],["Dₕ","head_dim = 128"]],
  "s-mask":[["Ā","缩放后的 sparse scores"],["𝒮","Indexer 选中的 token 集合"],["Ã","causal / padding mask 后的 scores"],["c_b","请求 b 的 context 长度"]],
  "s-softmax":[["Ã","mask 后的 sparse scores"],["P","selected KV 上的概率"],["𝒮","当前 query 的候选 token 集合"]],
  "s-shared":[["u","Shared Expert 输入 · [B,S,H]"],["W₁,s","shared gate_proj.weight"],["W₃,s","shared up_proj.weight"],["W₂,s","shared down_proj.weight"],["H","hidden_size = 6144"],["H_shared","shared_intermediate_size = 3072"],["E_shared(u)","Shared Expert 输出"]],
  "s-sum":[["𝓔","当前 token 选中的 routed experts"],["ŵₑ","第 e 个 routed expert 权重"],["Eₑ(U)","第 e 个 routed expert 输出"],["E_shared(U)","Shared Expert 输出"],["Ymoe","两路求和后的 MoE 输出"]],
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
  "d-qnorm":String.raw`\tilde Q_{b,h,s,:}=\frac{Q_{b,h,s,:}}{\sqrt{\frac1{D_h}\lVert Q_{b,h,s,:}\rVert_2^2+\varepsilon}}\odot(1+\gamma_Q)`,
  "d-knorm":String.raw`\tilde K_{b,g,s,:}=\frac{K_{b,g,s,:}}{\sqrt{\frac1{D_h}\lVert K_{b,g,s,:}\rVert_2^2+\varepsilon}}\odot(1+\gamma_K)`,
  "d-ropeq":String.raw`\begin{aligned}\theta_{p,j}&=p\,\theta_{\mathrm{base}}^{-2j/D_r}\\\binom{Q^r_{2j}}{Q^r_{2j+1}}&=\begin{bmatrix}\cos\theta_{p,j}&-\sin\theta_{p,j}\\\sin\theta_{p,j}&\cos\theta_{p,j}\end{bmatrix}\binom{\tilde Q_{2j}}{\tilde Q_{2j+1}}\\Q^r_{D_r:D_h}&=\tilde Q_{D_r:D_h}\end{aligned}`,
  "d-ropek":String.raw`\begin{aligned}\theta_{p,j}&=p\,\theta_{\mathrm{base}}^{-2j/D_r}\\\binom{K^r_{2j}}{K^r_{2j+1}}&=\begin{bmatrix}\cos\theta_{p,j}&-\sin\theta_{p,j}\\\sin\theta_{p,j}&\cos\theta_{p,j}\end{bmatrix}\binom{\tilde K_{2j}}{\tilde K_{2j+1}}\\K^r_{D_r:D_h}&=\tilde K_{D_r:D_h}\end{aligned}`,
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
  "s-idxnorm":String.raw`\tilde Q^{\mathrm{idx}}=\operatorname{RMSNorm}(Q^{\mathrm{idx}}),\qquad\tilde K^{\mathrm{idx}}=\operatorname{RMSNorm}(K^{\mathrm{idx}})`,
  "s-idxscore":String.raw`S^{(r)}_{b,i,j}=\frac{\langle\tilde Q^{\mathrm{idx}}_{b,r,i,:},\tilde K^{\mathrm{idx}}_{b,0,j,:}\rangle}{\sqrt{D_{idx}}}+M_{b,i,j}`,
  "s-blockmax":String.raw`B^{(r)}_{b,i,u}=\max_{j\in[B_{block}u,B_{block}(u+1))}S^{(r)}_{b,i,j}`,
  "s-topk":String.raw`\begin{aligned}\hat B_u&=B_u+\lambda_{local}\mathbf1[u\in\mathcal L_i]+\lambda_{init}\mathbf1[u\in\mathcal I]\\\mathcal S_{b,r,i}&=\operatorname{TopK}_{K_{block}}(\hat B)\end{aligned}`,
  "s-mainnorm":String.raw`\tilde Q=\operatorname{RMSNorm}(Q),\qquad\tilde K=\operatorname{RMSNorm}(K)`,
  "s-rope":String.raw`\begin{aligned}(Q^r_{:D_r},K^r_{:D_r})&=\operatorname{RoPE}(\tilde Q_{:D_r},\tilde K_{:D_r};\mathbf p)\\(Q^r_{D_r:},K^r_{D_r:})&=(\tilde Q_{D_r:},\tilde K_{D_r:})\end{aligned}`,
  "s-cache":String.raw`\mathcal K[\mathrm{slot}(r,p)]\leftarrow K^r_{r,p},\qquad\mathcal V[\mathrm{slot}(r,p)]\leftarrow V_{r,p}`,
  "s-select":String.raw`\begin{aligned}\mathcal P_{b,r,i}&=\{\mathrm{block\_table}[b,u]\mid u\in\mathcal S_{b,r,i}\}\\(K_{\mathcal S},V_{\mathcal S})&=\operatorname{gather}(\mathcal K,\mathcal V;\mathcal P_{b,r,i})\end{aligned}`,
  "s-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^r_{b,h,i,m}(K_{\mathcal S})_{b,\lfloor h/G\rfloor,j,m},\quad j\in\mathcal S_{b,\lfloor h/G\rfloor,i}`,
  "s-scale":String.raw`\bar A_{b,h,i,j}=A_{b,h,i,j}/\sqrt{D_h}`,
  "s-mask":String.raw`\tilde A_{b,h,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\in\mathcal S_i\ \land\ j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}`,
  "s-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-\max_t\tilde A_{b,h,i,t})}{\sum_{t\in\mathcal S_i}\exp(\tilde A_{b,h,i,t}-\max_u\tilde A_{b,h,i,u})}`,
  "s-pv":String.raw`O_{b,h,i,m}=\sum_{j\in\mathcal S_i}P_{b,h,i,j}(V_{\mathcal S})_{b,\lfloor h/G\rfloor,j,m}`,
  "s-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}\!\left(\operatorname{Concat}_{h=1}^{N_h/TP}(O_h),W_O\right)`,
  "s-addattn":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "s-router":String.raw`\begin{aligned}r&=UW_{\mathrm{router}}^\top\in\mathbb R^{B\times S\times E}\\s&=\sigma(r),\qquad\mathcal E=\operatorname{TopK}_K(s+b)\\\hat w_e&=s_{route}\,\frac{s_e}{\sum_{j\in\mathcal E}s_j},\quad e\in\mathcal E\end{aligned}`,
  "s-experts":String.raw`\begin{aligned}g_e&=W_{1,e}u,\quad v_e=W_{3,e}u\\\bar g_e&=\min(g_e,c),\quad\bar v_e=\operatorname{clip}(v_e,-c,c)\\E_e(u)&=W_{2,e}[\bar g_e\odot\sigma(\alpha\bar g_e)\odot(\bar v_e+\beta)]\end{aligned}`,
  "s-shared":String.raw`E_{\mathrm{shared}}(u)=W_{2,s}\operatorname{SwiGLUOAI}(W_{1,s}u,W_{3,s}u)`,
  "s-sum":String.raw`Y_{\mathrm{moe}}=\sum_{e\in\mathcal E}\hat w_eE_e(U)+E_{\mathrm{shared}}(U)`,
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
for(const id of ["d-norm","d-postnorm","s-norm","s-postnorm"]) CODE_BY_ID[id]={sections:NORM_SECTIONS,symbols:NORM_SYMBOLS};
CODE_BY_ID["d-gateup"]={sections:GATE_UP_SECTIONS,symbols:GATE_UP_SYMBOLS};
CODE_BY_ID["d-swiglu"]={sections:SWIGLU_SECTIONS,symbols:SWIGLU_SYMBOLS};
CODE_BY_ID["d-down"]={sections:DOWN_SECTIONS,symbols:DOWN_SYMBOLS};
CODE_BY_ID["s-shared"]={sections:MLP_SECTIONS,symbols:MLP_SYMBOLS};
for(const id of ["d-add2","s-addout"]) CODE_BY_ID[id]={sections:RESIDUAL_MERGE_SECTIONS,symbols:RESIDUAL_MERGE_SYMBOLS};
for(const id of ["d-add1","s-addattn"]) CODE_BY_ID[id]={sections:ATTENTION_RESIDUAL_SECTIONS,symbols:ATTENTION_RESIDUAL_SYMBOLS};
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
  "d-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE (Q) 输出"},{kind:"upstream",label:"visible K (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "d-mask":[{kind:"upstream",label:"scaled local scores",shape:"[B,64/TP,S,T]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "d-pv":[{kind:"upstream",label:"local attention probability P",shape:"[B,64/TP,S,T]",from:"Softmax 输出"},{kind:"upstream",label:"visible V (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "s-rope":[{kind:"upstream",label:"Q̃ · K̃",shape:"Q/K unchanged",from:"Main Q/K Norm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "s-cache":[{kind:"upstream",label:"Kᵣ · V",shape:"KV pages",from:"Partial RoPE 与 Split 5 outputs"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "s-topk":[{kind:"upstream",label:"local block scores",shape:"[B,max(1,4/TP),S,Nblocks]",from:"Block Max 输出"},{kind:"external",label:"local / init priority",shape:"logical block flags",from:"Indexer 配置：local_blocks=1, init_blocks=0"}],
  "s-select":[{kind:"upstream",label:"local logical block ids",shape:"[B,S,max(1,4/TP),16]",from:"Top-16 Blocks 输出"},{kind:"upstream",label:"paged K · V",shape:"KV pages",from:"Paged KV Cache 输出"},{kind:"external",label:"block_table",shape:"[B,Nblocks]",from:"KV cache manager"}],
  "s-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE 输出"},{kind:"upstream",label:"selected K (local KV groups)",shape:"≤16 pages/local group",from:"Select KV Pages 输出"}],
  "s-mask":[{kind:"upstream",label:"scaled local selected scores",shape:"[B,64/TP,S,Ksel]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "s-pv":[{kind:"upstream",label:"local selected attention P",shape:"[B,64/TP,S,Ksel]",from:"Softmax 输出"},{kind:"upstream",label:"selected V (local KV groups)",shape:"≤16 pages/local group",from:"Select KV Pages 输出"}],
  "s-router":[{kind:"upstream",label:"post-attn normalized hidden Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出"}],
  "s-experts":[{kind:"upstream",label:"normalized hidden + router logits",shape:"[B,S,6144] + [B,S,128]",from:"Post-attn RMSNorm 与 FP32 Router 输出"}],
  "s-shared":[{kind:"upstream",label:"all normalized tokens Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出；不经过 Top-K"}],
  "s-sum":[{kind:"upstream",label:"4 routed outputs",shape:"4 × [B,S,6144]",from:"Routed Experts ×4 输出"},{kind:"upstream",label:"shared output",shape:"[B,S,6144]",from:"Shared Expert ×1 输出"}],
  "d-add2":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Yffn · FFN branch",shape:"[B,S,6144]",from:"Down Projection 输出"}],
  "s-addout":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Ymoe · MoE branch",shape:"[B,S,6144]",from:"Weighted Sum 输出"}],
  "d-add1":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn · attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
  "s-addattn":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn · sparse attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
};

const NEXT_BY_ID: Record<string,string> = {
  "d-input":"Gemma RMSNorm","d-position":"Partial RoPE (Q/K)","d-attnmeta":"Apply Causal / Pad Bounds","d-slots":"Paged KV Cache","d-norm":"QKV Projection","d-qkv":"Split Q / K / V","d-split":"Q RMSNorm · K RMSNorm · Paged KV Cache","d-qnorm":"Partial RoPE (Q)","d-knorm":"Partial RoPE (K)","d-ropeq":"Q × Kᵀ","d-ropek":"Paged KV Cache","d-cache":"Q × Kᵀ · P × V","d-qk":"Scale 1/√128","d-scale":"Apply Causal / Pad Bounds","d-mask":"Softmax","d-softmax":"P × V","d-pv":"O Projection","d-oproj":"Attention Residual Merge","d-add1":"Post-attn Gemma RMSNorm","d-postnorm":"Gate + Up Projection","d-gateup":"Split Gate / Up","d-gatesplit":"SwiGLU-OAI","d-swiglu":"Down Projection","d-down":"Decoder Layer Residual Merge","d-add2":"下一 decoder layer / Final Norm",
  "s-input":"Gemma RMSNorm","s-position":"Partial RoPE","s-attnmeta":"Indexer 与 Sparse Attention mask","s-slots":"Paged KV Cache","s-norm":"QKV + Index Projection","s-packed":"Split 5 outputs","s-split":"Index Q/K Norm · Main Q/K Norm · Paged KV Cache","s-idxnorm":"Index Q × Kᵀ","s-idxscore":"Block Max","s-blockmax":"Top-16 Blocks","s-topk":"Select KV Pages","s-mainnorm":"Partial RoPE","s-rope":"Paged KV Cache · Q × selected Kᵀ","s-cache":"Select KV Pages","s-select":"Q × selected Kᵀ · P × selected V","s-qk":"Scale 1/√128","s-scale":"Apply Causal / Pad Bounds","s-mask":"Softmax","s-softmax":"P × selected V","s-pv":"O Projection","s-oproj":"Attention Residual Merge","s-addattn":"Post-attn Gemma RMSNorm","s-postnorm":"FP32 Router · Routed Experts · Shared Expert","s-router":"Routed Experts ×4","s-experts":"Weighted Sum","s-shared":"Weighted Sum","s-sum":"Decoder Layer Residual Merge","s-addout":"下一 decoder layer / Final Norm",
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
    qnorm: cloneOp(attn,{id:"d-qnorm",kind:"norm",title:"Q RMSNorm",input:"Q",inputShape:"[B,64,S,128]",output:"Q̃",outputShape:"[B,64,S,128]",formula:"Q̃=Q/√(mean(Q²)+ε)⊙(1+γq)",weights:attn.weights.filter(w=>w.key.includes("q_norm"))}),
    knorm: cloneOp(attn,{id:"d-knorm",kind:"norm",title:"K RMSNorm",input:"K",inputShape:"[B,4,T,128]",output:"K̃",outputShape:"[B,4,T,128]",formula:"K̃=K/√(mean(K²)+ε)⊙(1+γk)",weights:attn.weights.filter(w=>w.key.includes("k_norm"))}),
    ropeq: cloneOp(attn,{id:"d-ropeq",kind:"rope",title:"Partial RoPE (Q)",input:"Q̃ + positions",inputShape:"[B,64,S,128] + [S]",output:"Qᵣ",outputShape:"[B,64,S,128]",formula:"Qᵣ[:64]=RoPE(Q̃[:64],pos); Qᵣ[64:]=Q̃[64:]",weights:[]}),
    ropek: cloneOp(attn,{id:"d-ropek",kind:"rope",title:"Partial RoPE (K)",input:"K̃ + positions",inputShape:"[B,4,T,128] + [T]",output:"Kᵣ",outputShape:"[B,4,T,128]",formula:"Kᵣ[:64]=RoPE(K̃[:64],pos); Kᵣ[64:]=K̃[64:]",weights:[]}),
    cache: cloneOp(attn,{id:"d-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"[T,4,128] ×2",output:"visible K,V",outputShape:"[B,4,T,128] ×2",formula:"slot = block_table[seq, logical_block] + offset",formulaNote:"Dense 层读取完整可见历史；block table 决定物理 page。",weights:[]}),
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
    swiglu: cloneOp(mlp,{id:"d-swiglu",kind:"activation",kicker:"DENSE FFN · TP-LOCAL ACTIVATION",title:"SiluAndMulWithClamp · SwiGLU-OAI",summary:"对当前 TP rank 的 G⁽ʳ⁾/U⁽ʳ⁾ 分片执行 gate 上界截断、up 双边截断、sigmoid、+β 与逐元素乘；不执行线性投影。",input:"G⁽ʳ⁾,U⁽ʳ⁾",inputShape:"2 × [B,S,12288/TP]",output:"Z⁽ʳ⁾",outputShape:"[B,S,12288/TP]",formulaNote:"forward_native 先以 c=7 截断两个分支，再计算 Ḡ⁽ʳ⁾⊙σ(αḠ⁽ʳ⁾)⊙(Ū⁽ʳ⁾+β)；α=1.702，β=1.0。",runtime:"SiluAndMulWithClamp.forward_native",source:"activation.py · SiluAndMulWithClamp.forward_native · L214–218",sourceUrl:`${ACTIVATION_URL}#L214-L218`,weights:[]}),
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
  return {
    input:cloneOp(packed,{id:"s-input",kind:"io",title:"Hidden states",input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[]}),
    position:cloneOp(attn,{id:"s-position",kind:"route",title:"Build Position IDs",kicker:"vLLM RUNTIME I/O",input:"num_computed_tokens + query offsets",inputShape:"[B] + [Nq]",output:"positions",outputShape:"[Nq]",formula:"position(req,i)=num_computed_tokens[req]+i",formulaNote:"positions 由 vLLM runner 在模型 forward 之前构造，再传给 MiniMax-M3 的 fused QKNorm + RoPE kernel。",source:"gpu_model_runner.py · _prepare_inputs",sourceUrl:RUNNER_URL,weights:[]}),
    attnmeta:cloneOp(attn,{id:"s-attnmeta",kind:"mask",title:"Build Attention Metadata",kicker:"vLLM RUNTIME I/O",input:"query_start_loc, seq_lens, causal=True",inputShape:"[B+1] · [B] · bool",output:"implicit causal / padding layout",outputShape:"backend metadata; 非稠密 [S,T]",formula:"valid(req,q,k)=(k<seq_len[req]) ∧ (k≤context_len[req]+q)",formulaNote:"同一份边界元数据同时约束 indexer 的 block selection 和 main sparse attention。",source:"gpu_model_runner.py · CommonAttentionMetadata",sourceUrl:RUNNER_URL,weights:[]}),
    slots:cloneOp(attn,{id:"s-slots",kind:"route",title:"Resolve KV Slots",kicker:"vLLM RUNTIME I/O",input:"positions + block_table",inputShape:"[Nq] + [B,Nblocks]",output:"slot_mapping + block_table",outputShape:"[Nq] + [B,Nblocks]",formula:"slot=block_table[req,⌊position/block_size⌋]·block_size+(position mod block_size)",formulaNote:"slot_mapping 用于 K/V 写入；block_table 把 indexer 选出的逻辑 block id 翻译为物理 page。",source:"gpu_model_runner.py · compute_slot_mapping",sourceUrl:RUNNER_URL,weights:[]}),
    norm:cloneOp(normBase,{id:"s-norm",kind:"norm",title:"Gemma RMSNorm",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL,weights:[inputNorm]}),
    packed:cloneOp(packed,{id:"s-packed",kind:"linear",title:"QKV + Index Projection"}),
    split:cloneOp(packed,{id:"s-split",kind:"split",title:"Split 5 outputs",input:"packed projection",inputShape:"[B,S,9856]",output:"Q/K/V · Qidx/Kidx",outputShape:"8192/512/512 · 512/128",formula:"split(x,[8192,512,512,512,128],dim=-1)",weights:[]}),
    idxnorm:cloneOp(indexer,{id:"s-idxnorm",kind:"norm",title:"Index Q/K Norm",input:"Qidx,Kidx",inputShape:"[B,S,4,128] · [B,T,1,128]",output:"Q̃idx,K̃idx",outputShape:"same",weights:indexer.weights}),
    idxscore:cloneOp(indexer,{id:"s-idxscore",kind:"matmul",title:"Index Q × Kᵀ",input:"Q̃idx,K̃idx",inputShape:"[B,max(1,4/TP),S,128] · [B,1,T,128]",output:"local token scores",outputShape:"[B,max(1,4/TP),S,T]",formulaNote:"Index Q 与 KV heads 采用相同 TP 切分：每 rank 为 max(1,4/TP) heads；单个 Index K head 在需要时复制。",weights:[]}),
    blockmax:cloneOp(indexer,{id:"s-blockmax",kind:"route",title:"Block Max (128 tokens)",input:"causal token scores",inputShape:"[B,max(1,4/TP),S,T]",output:"block scores",outputShape:"[B,max(1,4/TP),S,⌈T/128⌉]",weights:[]}),
    topk:cloneOp(topk,{id:"s-topk",kind:"route",title:"Top-16 Blocks",input:"block scores + local priority",inputShape:"[B,max(1,4/TP),S,Nblocks]",output:"logical block ids",outputShape:"[B,S,max(1,4/TP),16]"}),
    mainnorm:cloneOp(attn,{id:"s-mainnorm",kind:"norm",title:"Main Q/K Norm",input:"Q,K",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"Q̃,K̃",outputShape:"same",weights:attn.weights.filter(w=>w.key.includes("_norm"))}),
    rope:cloneOp(attn,{id:"s-rope",kind:"rope",title:"Partial RoPE",input:"Q̃,K̃ + positions",inputShape:"Q/K + [S]",output:"Qᵣ,Kᵣ",outputShape:"Q/K unchanged",weights:[]}),
    cache:cloneOp(attn,{id:"s-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"KV pages + [B,Nblocks]",output:"paged K,V",outputShape:"[Npages,128,4,128] ×2",formula:"physical_page=block_table[logical_block]",weights:[]}),
    select:cloneOp(attn,{id:"s-select",kind:"route",title:"Select KV Pages",input:"paged K,V + Top-16 block ids",inputShape:"KV pages + [B,S,4,16]",output:"selected K,V",outputShape:"≤2048 KV tokens / group",formula:"physical_page=block_table[logical_top16]",weights:[]}),
    qk:cloneOp(attn,{id:"s-qk",kind:"matmul",title:"Q × selected Kᵀ",input:"Qᵣ, selected K",inputShape:"[B,64/TP,S,128] · ≤16 pages/local KV group",output:"local sparse scores",outputShape:"[B,64/TP,S,≤2048]",formulaNote:"单 rank 只持有 64/TP 个 query heads；selected K 按本 rank 的 max(1,4/TP) KV groups 提供。",weights:[]}),
    scale:cloneOp(attn,{id:"s-scale",kind:"scale",title:"Scale 1/√128",input:"scores",inputShape:"[B,64/TP,S,≤2048]",output:"scaled scores",outputShape:"same",weights:[]}),
    mask:cloneOp(attn,{id:"s-mask",kind:"mask",title:"Apply Causal / Pad Bounds",input:"scores + attention metadata",inputShape:"[B,64/TP,S,≤2048] + runtime metadata",output:"masked scores",outputShape:"same",formula:"Aᵢⱼ←valid_sparse(i,j) ? Aᵢⱼ : −∞",formulaNote:"Top-16 只决定候选 KV blocks；causal/padding 边界仍会在最终 Attention kernel 内再次约束可见 token。head 维为单 TP rank 的 64/TP。",weights:[]}),
    softmax:cloneOp(attn,{id:"s-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64/TP,S,≤2048]",output:"probabilities",outputShape:"same",weights:[]}),
    pv:cloneOp(attn,{id:"s-pv",kind:"matmul",title:"P × selected V",input:"P, selected V",inputShape:"[B,64/TP,S,≤2048] · selected V pages",output:"local heads",outputShape:"[B,S,8192/TP]",formulaNote:"P × selected V 在每个 TP rank 上输出 8192/TP 的局部 attention 宽度，再进入 RowParallel O Projection。",weights:[]}),
    oproj:cloneOp(attn,{id:"s-oproj",kind:"linear",title:"O Projection",input:"heads",inputShape:"[B,S,8192]",output:"Yattn",outputShape:"[B,S,6144]",weights:attn.weights.filter(w=>w.key.includes("o_proj"))}),
    addattn:cloneOp(combine,{id:"s-addattn",kind:"add",kicker:"DECODER LAYER · ATTENTION RESIDUAL",title:"Attention Residual Merge",summary:"在 Decoder Layer 内把 Sparse Attention 分支 Yattn 加入 residual stream Xₗ，得到更新后的 U；图中将 fused add 与紧随其后的 post-norm 分开表达。",input:"Xₗ + Yattn",inputShape:"2 × [B,S,6144]",output:"U · updated residual stream",outputShape:"[B,S,6144]",formula:"U=Xₗ+Yattn",formulaNote:"实际调用位于 DecoderLayer.forward L773：fused kernel 先执行 residual += hidden_states，再对更新后的 residual 执行 post-attention Gemma RMSNorm。",runtime:"fused_allreduce_gemma_rms_norm · attention residual",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L773–775",sourceUrl:`${CODE_URL}#L773-L775`,weights:[]}),
    postnorm:cloneOp(normBase,{id:"s-postnorm",kind:"norm",title:"Post-attn Gemma RMSNorm",summary:"输入 U 已由上游 Add 节点计算完成；此节点只执行 Gemma RMSNorm(U)，输出唯一的 Û 作为 MoE 输入。",formulaNote:"U 是上游 Add 的单一输出；本节点只计算 RMS(U) 与 (1+γpost) 缩放，不重复执行 residual add。",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",source:"nvidia/model.py · MiniMAXGemmaRMSNorm.forward · L130–142",sourceUrl:NORM_FORWARD_URL,weights:[postNorm]}),
    router:cloneOp(router,{id:"s-router",kind:"route",title:"FP32 Router → Top-4",input:"Û",inputShape:"[B,S,6144]"}),
    experts:cloneOp(experts,{id:"s-experts",kind:"activation",title:"Routed Experts ×4",input:"Û + expert ids + weights",inputShape:"[B,S,6144] + 2×[B,S,4]"}),
    shared:cloneOp(shared,{id:"s-shared",kind:"activation",title:"Shared Expert ×1",input:"Û",inputShape:"[B,S,6144]"}),
    sum:cloneOp(combine,{id:"s-sum",kind:"add",title:"Weighted Sum",input:"4 routed + shared",inputShape:"5 × [B,S,6144]",output:"Ymoe",outputShape:"[B,S,6144]",weights:[]}),
    addout:cloneOp(combine,{id:"s-addout",kind:"add",kicker:"DECODER LAYER · MOE RESIDUAL",title:"Decoder Layer Residual Merge",summary:"在 Decoder Layer 边界把 MoE 分支 Ymoe 与 residual stream U 逻辑合并，得到 Xₗ₊₁；该 merge 延迟融合到下一层 input RMSNorm。",input:"U + Ymoe",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁ · logical next-layer input",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+Ymoe",formulaNote:"当前层 L776–778 返回 Ymoe 与 U 两条独立流；下一 Decoder Layer 在 L758–767 的 fused input RMSNorm 中执行实际 add。",runtime:"MiniMaxM3DecoderLayer boundary · deferred residual merge",source:"nvidia/model.py · MiniMaxM3DecoderLayer.forward · L758–778",sourceUrl:`${CODE_URL}#L758-L778`,weights:[]}),
  };
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
      {from:"mlp-uhat",to:"mlp-gateup",toPort:"top"},{from:"mlp-wgate",to:"mlp-gateup",toPort:"top-left"},{from:"mlp-wup",to:"mlp-gateup",toPort:"top-right"},{from:"mlp-gateup",to:"mlp-packed"},{from:"mlp-packed",to:"mlp-split"},{from:"mlp-split",to:"mlp-gate"},{from:"mlp-split",to:"mlp-up"},{from:"mlp-gate",to:"mlp-gate-act"},{from:"mlp-up",to:"mlp-up-act"},{from:"mlp-gate-act",to:"mlp-mul"},{from:"mlp-up-act",to:"mlp-mul"},{from:"mlp-mul",to:"mlp-activated"},{from:"mlp-activated",to:"mlp-down"},{from:"mlp-wdown",to:"mlp-down",fromPort:"left",toPort:"right"},{from:"mlp-down",to:"mlp-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>SWIGLU-OAI MLP · L0–2</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="mlp-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="mlp-uhat"/><Tensor name="mlp.gate_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wgate"/><N id="gateup" graphId="mlp-gateup"/><Tensor name="mlp.up_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wup"/><Tensor name="packed gate_up⁽ʳ⁾" shape="[B,S,2H_dense/TP]" graphId="mlp-packed"/><N id="gatesplit" graphId="mlp-split"/><Tensor name="G⁽ʳ⁾ · gate" shape="[B,S,H_dense/TP]" graphId="mlp-gate"/><Tensor name="U⁽ʳ⁾ · up" shape="[B,S,H_dense/TP]" graphId="mlp-up"/><button type="button" className="mini-math activation-step" data-graph-id="mlp-gate-act" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}>clamp → Ḡ⁽ʳ⁾·σ(αḠ⁽ʳ⁾)</button><button type="button" className="mini-math activation-step" data-graph-id="mlp-up-act" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}>clamp → Ū⁽ʳ⁾ + β</button><button className="multiply-circle" data-graph-id="mlp-mul" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}>×</button><Tensor name="Z⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-activated"/><N id="down" graphId="mlp-down"/><Tensor name="mlp.down_proj.weight⁽ʳ⁾" shape="[H,H_dense/TP]" role="weight" graphId="mlp-wdown"/><Tensor name="Yffn" shape="[B,S,H]" graphId="mlp-y"/>
    </GraphSurface></section>;
  }
  if(stage==="ffn"){
    const edges:GraphEdge[]=[
      {from:"moe-u",to:"moe-router"},{from:"moe-wrouter",to:"moe-router",fromPort:"right",toPort:"left"},{from:"moe-router",to:"moe-ids"},{from:"moe-router",to:"moe-rweights"},{from:"moe-u",to:"moe-experts",toPort:"top-right"},{from:"moe-ids",to:"moe-experts",toPort:"top-left"},{from:"moe-rweights",to:"moe-experts"},{from:"moe-wexperts",to:"moe-experts",fromPort:"right",toPort:"left"},{from:"moe-experts",to:"moe-routed"},{from:"moe-u",to:"moe-shared",fromPort:"bottom",toPort:"top"},{from:"moe-wshared",to:"moe-shared",fromPort:"left",toPort:"right"},{from:"moe-shared",to:"moe-shared-out"},{from:"moe-routed",to:"moe-sum"},{from:"moe-shared-out",to:"moe-sum"},{from:"moe-sum",to:"moe-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>TOP-4 MOE + SHARED EXPERT · L3–59</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="moe-node-graph" edges={edges}><Tensor name="Û" shape="[B,S,H]" graphId="moe-u"/><Tensor name="router gate · correction bias" shape="[E,H] · [E]" role="weight" graphId="moe-wrouter"/><N id="router" graphId="moe-router"/><Tensor name="expert ids" shape="[B,S,K]" graphId="moe-ids"/><Tensor name="router weights" shape="[B,S,K]" graphId="moe-rweights"/><Tensor name="routed expert weights · w1/w3/w2" shape="E × expert weights" role="weight" graphId="moe-wexperts"/><N id="experts" graphId="moe-experts"/><Tensor name="weighted routed output" shape="[B,S,H]" graphId="moe-routed"/><N id="shared" graphId="moe-shared"/><Tensor name="shared expert weights ×3" shape="gate / up / down" role="weight" graphId="moe-wshared"/><Tensor name="shared output" shape="[B,S,H]" graphId="moe-shared-out"/><N id="sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[B,S,H]" graphId="moe-y"/></GraphSurface></section>;
  }
  const dense=type==="dense";
  const ids=dense?{project:"qkv",split:"split",qnorm:"qnorm",knorm:"knorm",ropeq:"ropeq",ropek:"ropek"}:{project:"packed",split:"split",qnorm:"mainnorm",knorm:"mainnorm",ropeq:"rope",ropek:"rope"};
  const keyId=dense?"attn-paged-k":"attn-selected-k"; const valueId=dense?"attn-paged-v":"attn-selected-v";
  const edges:GraphEdge[]=[
    {from:"attn-x",to:"attn-project",fromPort:"right",toPort:"left"},{from:"attn-project",to:"attn-packed",fromPort:"right",toPort:"left"},{from:"attn-packed",to:"attn-split",fromPort:"right",toPort:"left"},{from:"attn-split",to:"attn-q"},{from:"attn-split",to:"attn-k"},{from:"attn-split",to:"attn-v"},{from:"attn-q",to:"attn-qnorm",toPort:"top-left"},{from:"attn-wq",to:"attn-qnorm",toPort:"top-right"},{from:"attn-qnorm",to:"attn-qt"},{from:"attn-qt",to:"attn-qrope"},{from:"attn-posq",to:"attn-qrope",fromPort:"left",toPort:"right"},{from:"attn-qrope",to:"attn-qr"},{from:"attn-k",to:"attn-knorm",toPort:"top-left"},{from:"attn-wk",to:"attn-knorm",toPort:"top-right"},{from:"attn-knorm",to:"attn-kt"},{from:"attn-kt",to:"attn-krope"},{from:"attn-posk",to:"attn-krope",fromPort:"left",toPort:"right"},{from:"attn-krope",to:"attn-kr"},{from:"attn-kr",to:"attn-cache"},{from:"attn-v",to:"attn-cache"},{from:"attn-cache-meta",to:"attn-cache",fromPort:"left",toPort:"right"},{from:"attn-cache",to:"attn-paged-k"},{from:"attn-cache",to:"attn-paged-v"},{from:"attn-qr",to:"attn-qk"},{from:keyId,to:"attn-qk"},{from:"attn-qk",to:"attn-scale",fromPort:"right",toPort:"left"},{from:"attn-scale",to:"attn-scaled",fromPort:"right",toPort:"left"},{from:"attn-scaled",to:"attn-mask",fromPort:"right",toPort:"left"},{from:"attn-bounds",to:"attn-mask"},{from:"attn-mask",to:"attn-softmax",fromPort:"right",toPort:"left"},{from:"attn-softmax",to:"attn-p",fromPort:"right",toPort:"left"},{from:"attn-p",to:"attn-pv",fromPort:"right",toPort:"left"},{from:valueId,to:"attn-pv"},{from:"attn-pv",to:"attn-heads",fromPort:"right",toPort:"left"},{from:"attn-heads",to:"attn-oproj",fromPort:"right",toPort:"left"},{from:"attn-oproj",to:"attn-y",fromPort:"right",toPort:"left"},
    ...(!dense?[{from:"attn-split",to:"attn-qidx"},{from:"attn-split",to:"attn-kidx"},{from:"attn-qidx",to:"attn-idxnorm"},{from:"attn-kidx",to:"attn-idxnorm"},{from:"attn-idxnorm",to:"attn-idxscore",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-idxbounds",to:"attn-idxscore"},{from:"attn-idxscore",to:"attn-blockmax",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-blockmax",to:"attn-topk",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-topk",to:"attn-topids",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-topids",to:"attn-select"},{from:"attn-paged-k",to:"attn-select"},{from:"attn-paged-v",to:"attn-select"},{from:"attn-select",to:"attn-selected-k",fromPort:"right" as EdgePort,toPort:"left" as EdgePort},{from:"attn-select",to:"attn-selected-v",fromPort:"right" as EdgePort,toPort:"left" as EdgePort}] : []),
  ];
  return <section className="stage-zoom lesson-zoom attention-lesson"><header><span>{dense?"GQA + PARTIAL ROPE · L0–2":"MINIMAX SPARSE ATTENTION + PARTIAL ROPE · L3–59"}</span><button onClick={onClose}>收起 ×</button></header><GraphSurface className="attention-flowchart connected-attention-graph" edges={edges}>
    <div className="compact-chain"><Tensor name="X̂" shape="[B,S,H]" graphId="attn-x"/><N id={ids.project} graphId="attn-project"/><Tensor name="packed" shape={dense?"[B,S,9216]":"[B,S,9856]"} graphId="attn-packed"/><N id={ids.split} graphId="attn-split"/></div>
    {!dense&&<div className="index-ribbon"><div className="multi-source"><Tensor name="Qidx" shape="[B,S,4,128]" graphId="attn-qidx"/><Tensor name="Kidx" shape="[B,T,1,128]" graphId="attn-kidx"/></div><N id="idxnorm" graphId="attn-idxnorm"/><N id="idxscore" graphId="attn-idxscore"/><Tensor name="causal bounds" shape="runtime" role="side" graphId="attn-idxbounds"/><N id="blockmax" graphId="attn-blockmax"/><N id="topk" graphId="attn-topk"/><Tensor name="Top-16 block ids" shape="[B,S,4,16]" graphId="attn-topids"/></div>}
    <div className="qkv-lanes"><section><header>Q PATH</header><IW id={ids.qnorm} inputName="Q" inputShape="[B,Nₕ,S,Dₕ]" inputGraphId="attn-q" graphId="attn-qnorm" weightGraphId="attn-wq"/><div className="two-source"><Tensor name="Q̃" shape="same" graphId="attn-qt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posq"/></div><N id={ids.ropeq} graphId="attn-qrope"/><Tensor name="Qᵣ" shape="[B,Nₕ,S,Dₕ]" graphId="attn-qr"/></section><section><header>K PATH</header><IW id={ids.knorm} inputName="K" inputShape="[B,Nₖᵥ,S,Dₕ]" weightIndex={dense?0:1} inputGraphId="attn-k" graphId="attn-knorm" weightGraphId="attn-wk"/><div className="two-source"><Tensor name="K̃" shape="same" graphId="attn-kt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posk"/></div><N id={ids.ropek} graphId="attn-krope"/><Tensor name="Kᵣ" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-kr"/></section><section><header>V + KV CACHE</header><Tensor name="V" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-v"/><Tensor name="slot_mapping · block_table" shape="runtime" role="side" graphId="attn-cache-meta"/><N id="cache" graphId="attn-cache"/><div className="two-source"><Tensor name="paged K" shape="KV pages" graphId="attn-paged-k"/><Tensor name="paged V" shape="KV pages" graphId="attn-paged-v"/></div></section></div>
    {!dense&&<div className="selection-chain"><N id="select" graphId="attn-select"/><div className="two-source"><Tensor name="selected K" shape="≤2048 tokens/group" graphId="attn-selected-k"/><Tensor name="selected V" shape="≤2048 tokens/group" graphId="attn-selected-v"/></div></div>}
    <div className="score-pipeline"><N id="qk" graphId="attn-qk"/><N id="scale" graphId="attn-scale"/><Tensor name="scaled scores" shape="[B,Nₕ,S,T]" graphId="attn-scaled"/><Tensor name="causal / pad bounds" shape="runtime metadata" role="side" graphId="attn-bounds"/><N id="mask" graphId="attn-mask"/><N id="softmax" graphId="attn-softmax"/><Tensor name="P" shape="[B,Nₕ,S,T]" graphId="attn-p"/></div>
    <div className="context-pipeline"><N id="pv" graphId="attn-pv"/><Tensor name="heads" shape="[B,S,Nₕ·Dₕ]" graphId="attn-heads"/><N id="oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[B,S,H]" graphId="attn-y"/></div>
  </GraphSurface></section>;
}

function DecoderDiagram({type,g,active,expanded,onExpand,onHover,onLeave,onSelect}:{type:LayerType;g:Record<string,OpNode>;active:string;expanded:ExpandedStage;onExpand:(stage:ExpandedStage)=>void;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const IW=({id,inputName,inputShape,inputGraphId,graphId,weightGraphId}:{id:string;inputName:string;inputShape:string;inputGraphId:string;graphId:string;weightGraphId:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId}/>;
  const A=({id,graphId}:{id:string;graphId:string})=><AddCircle node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const edges:GraphEdge[]=[{from:"main-x",to:"main-norm",toPort:"top-left"},{from:"main-win",to:"main-norm",toPort:"top-right"},{from:"main-norm",to:"main-attn"},{from:"main-attn",to:"main-add1"},{from:"main-x",to:"main-add1",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add1",to:"main-u"},{from:"main-u",to:"main-post",toPort:"top-left"},{from:"main-wpost",to:"main-post",toPort:"top-right"},{from:"main-post",to:"main-ffn"},{from:"main-ffn",to:"main-add2"},{from:"main-u",to:"main-add2",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add2",to:"main-out"}];
  return <div className={`decoder-workbench ${expanded?"has-zoom":""}`}>{!expanded&&<GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <IW id="norm" inputName="Xₗ · hidden_states / residual stream" inputShape="[B,S,H]" inputGraphId="main-x" graphId="main-norm" weightGraphId="main-win"/><button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={()=>onExpand(expanded==="attention"?null:"attention")}><small>点击展开</small><b>{type==="dense"?"GQA + Partial RoPE":"MiniMax Sparse Attention + Partial RoPE"}</b></button><A id={type==="dense"?"add1":"addattn"} graphId="main-add1"/><IW id="postnorm" inputName="U · updated residual stream" inputShape="[B,S,H]" inputGraphId="main-u" graphId="main-post" weightGraphId="main-wpost"/><button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={()=>onExpand(expanded==="ffn"?null:"ffn")}><small>点击展开</small><b>{type==="dense"?"SwiGLU-OAI MLP":"Top-4 MoE + Shared Expert"}</b></button><A id={type==="dense"?"add2":"addout"} graphId="main-add2"/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,H]" graphId="main-out"/>
  </GraphSurface>}{expanded&&<StageZoom type={type} stage={expanded} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={()=>onExpand(null)}/>}</div>;
}
/* eslint-enable react-hooks/static-components */

function LayerNavigator({value,onChange}:{value:string;onChange:(next:string)=>void}){
  const type:LayerType = value==="dense"?"dense":"sparse";
  return <div className="layer-nav layer-type-nav"><div className="layer-nav-head"><b>{type==="dense"?"GQA + SwiGLU-OAI MLP":"MiniMax Sparse Attention + MoE"}</b></div><div className="layer-type-options"><button className={type==="dense"?"active dense":"dense"} onClick={()=>onChange("dense")}><span>L0–L2</span><b>GQA + Partial RoPE + SwiGLU-OAI MLP</b><small>3 层共享同一实现</small></button><button className={type==="sparse"?"active sparse":"sparse"} onClick={()=>onChange("sparse")}><span>L3–L59</span><b>MiniMax Sparse Attention + Partial RoPE + Top-4 MoE</b><small>57 层共享同一实现</small></button></div></div>;
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

function ShapeRows({shape,symbolic}:{shape:string;symbolic:(shape:string)=>string}){
  return <div className="shape-rows"><span><i>符号</i><code title={symbolic(shape)}>{symbolic(shape)}</code></span><span><i>实际</i><code title={shape}>{shape}</code></span></div>;
}

function bindingsFor(node:OpNode):IoBinding[]{
  const dataInputs=INPUT_OVERRIDES[node.id]??[{kind:node.kind==="io"?"external":"upstream",label:node.input,shape:node.inputShape,from:node.kind==="io"?"模型调用方 / runtime":"图中紧邻的上游模块输出"}];
  const weightInputs=node.weights.map(weight=>{
    const tpShape=node.id==="d-gateup"?weight.shape.replace("[12288,6144]","[12288/TP,6144]"):node.id==="d-down"?weight.shape.replace("[6144,12288]","[6144,12288/TP]"):null;
    return {kind:"weight" as const,label:weight.key,shape:tpShape?`${weight.dtype} · TP shard ${tpShape}`:`${weight.dtype} · ${weight.shape}`,from:weight.runtime?`checkpoint → ${weight.runtime}`:`checkpoint · ${weight.shard}`,note:weight.params?`${weight.params} parameters`:undefined};
  });
  return [...dataInputs,...weightInputs];
}

function IoView({node,bindings,symbolic,next}:{node:OpNode;bindings:IoBinding[];symbolic:(shape:string)=>string;next:(nodeId:string)=>string|undefined}){
  const labels:Record<BindingKind,string>={upstream:"上游张量",external:"外部输入",weight:"权重输入"};
  return <div className="io-binding-view"><section className="binding-list"><header><span>INPUT BINDINGS</span><b>{bindings.length} 路输入</b></header>{bindings.map((binding,index)=><article className={`binding binding-${binding.kind}`} key={`${binding.kind}-${binding.label}-${index}`}><div><span>{labels[binding.kind]}</span></div><b>{binding.label}</b><ShapeRows shape={binding.shape} symbolic={symbolic}/><p><i>来自</i>{binding.from}</p>{binding.note&&<small>{binding.note}</small>}</article>)}</section><section className="output-binding"><header><span>OUTPUT BINDING</span><b>1 路产物</b></header><article><div><span>计算产物</span></div><b>{node.output}</b><ShapeRows shape={node.outputShape} symbolic={symbolic}/><p><i>送往</i>{next(node.id)??"图中下游模块"}</p></article></section></div>;
}

function CodeView({node,commit}:{node:OpNode;commit:string}){
  const sections=(node.codeSections??[]).filter(section=>/forward|INIT|CALL|ENTER|PROJECT|ATTEND|ROUTE|SHARED/.test(`${section.title} ${section.stage}`));
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {commit.slice(0,7)}</span><b>{node.source}</b><i>↗</i></a>
    {sections.length?<section className="code-call-chain"><header><span>IMPLEMENTATION TRACE</span><b>forward → fused kernel → 数学定义</b></header>{sections.map((section,index)=><article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><span>{section.stage}</span><b>{section.title}</b><small>{section.location}</small></div>{section.url&&<a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article>)}</section>:<div className="code-empty"><b>此节点没有独立 forward</b><p>它由所在模块的 forward 调度，或只是一个数学拆解步骤。</p></div>}
  </div>;
}


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
    flow:"Û → Router Top-4 ↘ Routed Experts · Shared Expert ↗ Weighted Sum → Ymoe",
    formula:"Ymoe=Σₑ pₑEₑ(Û)+Eshared(Û)",
    notes:["同一个 Û 直接进入 Router、Routed Experts 与 Shared Expert。","Router 只决定专家 id 与权重；Shared Expert 不经过 Top-K。"],
    parameters:[["E","128","routed experts"],["K","4","experts / token"],["E_shared","1","shared expert"],["H_expert","3072","expert width"]],
  };
}

function StageOverviewPanel({overview}:{overview:StageOverview}){
  return <aside className="detail-panel stage-overview-panel"><header className="stage-overview-header"><span>{overview.kicker}</span><h2>{overview.title}</h2><p>{overview.summary}</p></header><div className="stage-overview-body"><section><span>数据流</span><code>{overview.flow}</code></section><section className="stage-formula-section"><span>计算语义</span><code>{overview.formula}</code>{overview.formulaNote&&<p>{overview.formulaNote}</p>}</section><section className="stage-parameter-section"><span>关键参数</span><div className="stage-parameters">{overview.parameters.map(([symbol,value,source])=><article key={symbol}><b>{symbol}</b><strong>{value}</strong><small>{source}</small></article>)}</div></section><section><span>边界说明</span>{overview.notes.map(note=><p key={note}>{note}</p>)}</section></div><footer>展开图说明 · 点击算子查看独立详情</footer></aside>;
}

function DetailPanel({node,tab,setTab,pinned,onClear,expanded,layerType,module}:{node:OpNode|null;tab:Tab;setTab:(t:Tab)=>void;pinned:boolean;onClear:()=>void;expanded:ExpandedStage;layerType:string;module:ModelModule}){
  const tabs:[Tab,string][]=[["io","I/O + 权重"],["formula","公式"],["code","代码"]];
  if(!node&&expanded){const overview=module.stageOverviewFor(layerType,expanded);return overview?<StageOverviewPanel overview={overview}/>:<aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后，可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;}
  if(!node)return <aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后，可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div>{pinned?<button className="unpin-button" aria-label="取消固定" title="取消固定" onClick={onClear}>×</button>:<i className={`kind-dot op-${node.kind}`}/>}<p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div className={`detail-content detail-${tab}`}>
    {tab==="io"&&<IoView node={node} bindings={module.inputBindingsFor(node)} symbolic={module.symbolicShape} next={module.nextFor}/>}
    {tab==="formula"&&<div className="formula-view"><span>作用</span><div className="formula-purpose">{node.summary}</div><span>实际公式</span><LatexFormula node={node}/><div className="formula-implementation"><b>一句话解释</b><p>{module.formulaNoteFor(node)}</p></div><div className="formula-terms">{module.formulaTermsFor(node).map(([symbol,meaning])=><span key={symbol}><b>{symbol}</b>{meaning}</span>)}</div></div>}
    {tab==="code"&&<CodeView node={node} commit={module.vllmCommit}/>}
    </div><footer>vLLM @ {module.vllmCommit.slice(0,7)} · official safetensors</footer></aside>;
}

function HelpModal({onClose,groups,symbols}:{onClose:()=>void;groups:readonly ConfigGroup[];symbols:Record<string,string>}){
  const [activeGroup,setActiveGroup]=useState(0);
  const group=groups[activeGroup];
  const symbolFor=(groupTitle:string,key:string)=>symbols[`${groupTitle}:${key}`]??"—";
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal reference-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="完整 config.json 参数与符号"><header><div><span>CONFIG REFERENCE</span><h2>完整 config.json · 参数与符号</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><nav className="config-tabs" role="tablist" aria-label="选择 config.json 分组">{groups.map((item,index)=><button key={item.title} role="tab" aria-selected={activeGroup===index} className={activeGroup===index?"active":""} onClick={()=>setActiveGroup(index)}>{item.title}</button>)}</nav><div className="config-reference" role="tabpanel"><section><h3>{group.title}</h3><table><colgroup><col className="config-key-column"/><col className="config-symbol-column"/><col/></colgroup><thead><tr><th>参数</th><th>符号</th><th>值</th></tr></thead><tbody>{group.rows.map(([key,value])=><tr key={key}><th scope="row">{key}</th><td className="config-symbol"><code>{symbolFor(group.title,key)}</code></td><td>{value}</td></tr>)}</tbody></table></section></div></section></div>;
}

function MiniMaxM3Overview(){
  return <div className="model-overview"><div className="model-step">Text / Vision Inputs</div><Arrow/><div className="model-step">Embedding Fusion <code>[B,S,H]</code></div><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>L0–2 · GQA + Partial RoPE + SwiGLU-OAI MLP</span><span><i className="sparse"/>L3–59 · MiniMax Sparse Attention + Partial RoPE + MoE</span></div><Arrow/><div className="model-step">Final Gemma RMSNorm <code>[B,S,H]</code></div><Arrow/><div className="model-step">LM Head <code>[B,S,V]</code></div></div>;
}

const minimaxM3Module: ModelModule = {
  id: "minimax-m3",
  name: "MiniMax-M3",
  facts: { total: "428B", active: "23B", context: "1M", checkpoint: "869 GB" },
  links: { codeUrl: CODE_URL, codeLabel: `vLLM @ ${VLLM_COMMIT.slice(0, 7)}`, weightsUrl: WEIGHTS_URL, weightsLabel: "Hugging Face · 59 shards" },
  vllmCommit: VLLM_COMMIT,
  defaultLayerType: "sparse",
  configGroups: CONFIG_GROUPS,
  configSymbols: CONFIG_SYMBOLS,
  graphFor: (layerType: string) => layerType === "dense" ? denseGraph(2) : sparseGraph(3),
  inputBindingsFor: bindingsFor,
  nextFor: (nodeId: string) => NEXT_BY_ID[nodeId],
  symbolicShape,
  stageOverviewFor: (layerType: string, stage: Exclude<ExpandedStage, null>) => stageOverview(layerType === "dense" ? "dense" : "sparse", stage),
  formulaTermsFor: formulaTerms,
  formulaNoteFor: (node: OpNode) => node.formulaNote ?? FORMULA_NOTE[node.kind] ?? "",
  Overview: MiniMaxM3Overview,
  Navigator: LayerNavigator,
  canvasHeading: (layerType: string) => ({
    kicker: "DECODER LAYER · 按结构类型展示",
    title: layerType === "dense" ? "GQA + Partial RoPE + SwiGLU-OAI MLP · L0–L2 同构" : "MiniMax Sparse Attention + Partial RoPE + Top-4 MoE · L3–L59 同构",
  }),
  Workbench: (props) => <DecoderDiagram type={props.layerType === "dense" ? "dense" : "sparse"} g={props.graph} active={props.activeId} expanded={props.expanded} onExpand={props.onExpand} onHover={props.onHover} onLeave={props.onLeave} onSelect={props.onSelect} />,
};

const MODULES: Record<string, ModelModule> = { "minimax-m3": minimaxM3Module };

function initialModelFromHash(): string {
  const id = window.location.hash.replace(/^#\//, "");
  return MODULES[id]?.id ?? "minimax-m3";
}

export default function Home(){
  const [modelId,setModelId]=useState(initialModelFromHash()); const module=MODULES[modelId]??minimaxM3Module;
  const [layerType,setLayerType]=useState<string>(module.defaultLayerType); const [expanded,setExpanded]=useState<ExpandedStage>(null); const [tab,setTab]=useState<Tab>("io"); const [dark,setDark]=useState(false); const [help,setHelp]=useState(false);
  const graph=module.graphFor(layerType); const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null}); const active=detail.pinned??detail.hovered;
  const updateDetail=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
  const changeModel=(nextId:string)=>{const next=MODULES[nextId];if(!next)return;setModelId(nextId);setLayerType(next.defaultLayerType);setExpanded(null);updateDetail({type:"clear"});window.location.hash=`/${nextId}`;};
  const changeLayerType=(next:string)=>{setLayerType(next);setExpanded(null);updateDetail({type:"clear"})};
  return <main className={`atlas-app ${dark?"dark":""}`}><header className="app-header">
    <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>Model Atlas</b></div></div>
    <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value={module.id} onChange={event=>changeModel(event.target.value)}>{MODEL_REGISTRY.map(m=><option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}</option>)}</select></label>
    <nav className="resource-links"><a href={module.links.codeUrl} target="_blank" rel="noreferrer"><b>CODE ↗</b><small>{module.links.codeLabel}</small></a><a href={module.links.weightsUrl} target="_blank" rel="noreferrer"><b>WEIGHTS ↗</b><small>{module.links.weightsLabel}</small></a></nav>
    <div className="model-facts"><span><b>{module.facts.total}</b><small>模型总参数量</small></span><span><b>{module.facts.active}</b><small>每 token 激活参数</small></span><span><b>{module.facts.context}</b><small>最大上下文 token</small></span><span><b>{module.facts.checkpoint}</b><small>BF16 checkpoint</small></span></div>
    <button className="help-button" onClick={()=>setHelp(true)} aria-label="查看参数和符号说明">?</button><button className="theme-button" onClick={()=>setDark(v=>!v)} aria-label="切换明暗主题">{dark?"☀":"☾"}</button>
  </header><div className="screen-grid"><section className="map-panel">
    <module.Overview/>
    <module.Navigator value={layerType} onChange={changeLayerType}/>
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按结构类型展示</span><h1>{module.canvasHeading(layerType).title}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span><i className="operator-swatch"/>OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><module.Workbench layerType={layerType} graph={graph} activeId={active?.id??""} expanded={expanded} onExpand={setExpanded} onHover={node=>updateDetail({type:"hover",node})} onLeave={()=>updateDetail({type:"leave"})} onSelect={node=>{updateDetail({type:"pin",node});setTab("io")}}/></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>updateDetail({type:"clear"})} expanded={expanded} layerType={layerType} module={module}/></div>{help&&<HelpModal onClose={()=>setHelp(false)} groups={module.configGroups} symbols={module.configSymbols}/>}</main>;
}
