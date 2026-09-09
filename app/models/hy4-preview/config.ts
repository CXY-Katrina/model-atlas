// 官方 checkpoint config.json(tencent/Hy4-preview @ 2026-09-09)。
// 注意:vllm 配置类 hy_v4.py 的默认值(hidden 2816/34 层/index 16 头)是占位默认,不是本模型的值。
export const CONFIG_GROUPS = [
  {title:"顶层",rows:[
    ["architectures","HYV4ForCausalLM"],["model_type","hy_v4"],["dtype","bfloat16"],["hidden_size","6144"],["num_hidden_layers","78"],["vocab_size","120832"],["max_position_embeddings","1048576"],["rms_norm_eps","1e−5"],["tie_word_embeddings","false"],["enable_lm_head_fp32","true"],["num_nextn_predict_layers","1"],
  ]},
  {title:"attention(MLA)",rows:[
    ["q_lora_rank","2048"],["kv_lora_rank","512"],["qk_nope_head_dim","192"],["qk_rope_head_dim","64"],["qk_head_dim","256"],["v_head_dim","256"],["num_attention_heads","64"],["num_key_value_heads","8"],["rope_theta","1e7(INTERLEAVED 布局)"],["gated_mla","true(elementwise)"],["learnable_sink","true(初始 0.0)"],
  ]},
  {title:"DSA indexer",rows:[
    ["use_dsa","true"],["layer_types","78 × deepseek_sparse_attention"],["indexer_types","21 full / 57 shared(full@L0,1,5,…,77)"],["index_topk","2048"],["index_n_heads","32"],["index_head_dim","128"],
  ]},
  {title:"MoE / FFN",rows:[
    ["mlp_layer_types","[dense, 77 × sparse]"],["intermediate_size(L0)","18432"],["moe_intermediate_size","2048"],["n_routed_experts","256"],["n_shared_experts","1"],["num_experts_per_tok","8"],["scoring_func","sigmoid"],["norm_topk_prob","true"],["routed_scaling_factor","2.827"],["swiglu_limit","10.0(仅 routed)"],["hidden_act","silu"],
  ]},
  {title:"iHC",rows:[
    ["enable_ihc","true"],["hc_mult","4"],["hc_magnitude","2.0"],["hc_eps","1e−6"],
  ]},
  {title:"MTP",rows:[
    ["mtp_loss_factor","0.1"],["MTP 块内 iHC","无(checkpoint 无对应权重)"],["MTP indexer","full(始终自算索引)"],
  ]},
] as const;

export const CONFIG_SYMBOLS: Record<string,string> = {
  "顶层:hidden_size":"H",
  "顶层:num_hidden_layers":"L",
  "顶层:vocab_size":"V",
  "顶层:max_position_embeddings":"S_max",
  "顶层:rms_norm_eps":"ε_rms",
  "顶层:num_nextn_predict_layers":"L_mtp",
  "attention(MLA):q_lora_rank":"R_q",
  "attention(MLA):kv_lora_rank":"C_kv",
  "attention(MLA):qk_nope_head_dim":"D_nope",
  "attention(MLA):qk_rope_head_dim":"D_rope",
  "attention(MLA):qk_head_dim":"D_qk",
  "attention(MLA):v_head_dim":"D_v",
  "attention(MLA):num_attention_heads":"N_h",
  "attention(MLA):num_key_value_heads":"N_kv",
  "attention(MLA):rope_theta":"θ",
  "attention(MLA):gated_mla":"gated MLA",
  "attention(MLA):learnable_sink":"sink_h",
  "DSA indexer:index_topk":"T_idx",
  "DSA indexer:index_n_heads":"N_idx",
  "DSA indexer:index_head_dim":"D_idx",
  "MoE / FFN:intermediate_size(L0)":"H_ffn",
  "MoE / FFN:moe_intermediate_size":"H_exp",
  "MoE / FFN:n_routed_experts":"E",
  "MoE / FFN:n_shared_experts":"E_sh",
  "MoE / FFN:num_experts_per_tok":"K",
  "MoE / FFN:routed_scaling_factor":"s_route",
  "MoE / FFN:swiglu_limit":"c",
  "iHC:hc_mult":"C_hc",
  "iHC:hc_magnitude":"m_hc",
  "iHC:hc_eps":"ε_hc",
};

export function configSymbol(group:string,key:string){
  return CONFIG_SYMBOLS[`${group}:${key}`]??"—";
}
