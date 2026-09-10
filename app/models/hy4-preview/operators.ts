import type { Node, OpNode, OpKind } from "../types";
import { coreNodes, attentionNodes, denseFfnNodes, moeNodes } from "./data";
import { CODE_BY_ID, INPUT_OVERRIDES, NEXT_BY_ID } from "./evidence";
import { LATEX_BY_ID } from "./formulas";
import { VLLM_COMMIT } from "./sources";
import type { LayerVariant } from "./types";

const cloneOp = (base: Node, values: Partial<OpNode> & { id: string; kind: OpKind; title: string }): OpNode => {
  const detail = CODE_BY_ID[values.id];
  return { ...base, ...values, latex: values.latex ?? LATEX_BY_ID[values.id], codeSections: values.codeSections ?? detail?.sections, codeSymbols: values.codeSymbols ?? detail?.symbols };
};
export const pinSource = (url: string) => url.replace("/blob/main/", `/blob/${VLLM_COMMIT}/`);

export function buildGraph(variant: LayerVariant): Record<string, OpNode> {
  const [input, hcpa, norm1, hcpo1, hcpm, norm2, hcpo2] = coreNodes();
  const attn = attentionNodes(variant === "moe-shared" ? "shared" : "full");
  const [aproj, qan, qb, kvn, kvb, rope, cache] = attn;
  const indexer = attn.slice(7, -4);
  const [qk, sink, pv, gate, oproj] = attn.slice(-5);
  const layerNo = variant === "dense-full" ? "0" : variant === "moe-full" ? "1" : "2";
  const g: Record<string, OpNode> = {};
  g.input = cloneOp(input, { id: "input", kind: "io", title: "hidden_states", kicker: `L${layerNo} · ${variant === "dense-full" ? "DENSE" : "MOE"} ${variant === "moe-shared" ? "+ SHARED INDEXER" : "+ FULL INDEXER"}` });
  g.hcprea = cloneOp(hcpa, { id: "hcpre-attn", kind: "norm", title: "hc_pre · attn(4→1)" });
  g.norm1 = cloneOp(norm1, { id: "norm1", kind: "norm", title: "input_layernorm" });
  g.hcposta = cloneOp(hcpo1, { id: "hcpost1", kind: "add", title: "hc_post · attn(散回 4 通道)" });
  g.hcprem = cloneOp(hcpm, { id: "hcpre-mlp", kind: "norm", title: "hc_pre · mlp(4→1)" });
  g.norm2 = cloneOp(norm2, { id: "norm2", kind: "norm", title: "post_attention_layernorm" });
  g.hcpostm = cloneOp(hcpo2, { id: "hcpost2", kind: "add", title: "hc_post · mlp(层输出 4 通道)" });
  // attention zoom nodes
  g.aproj = cloneOp(aproj, { id: "aproj", kind: "linear", title: "fused_qkv_a_proj" });
  g.asplit = cloneOp(aproj, { id: "aproj", kind: "split", title: "切分 [q_a | kv | k_pe]", input: "qkv_lora", inputShape: "[T,2624]", output: "q_a | kv | k_pe", outputShape: "2048 | 512 | 64", formula: "split(qkv,[2048,512,64])", formulaNote: "qkv_lora 最后一维切分;无权重。", weights: [] });
  g.qan = cloneOp(qan, { id: "qan", kind: "norm", title: "q_a_layernorm" });
  g.qb = cloneOp(qb, { id: "qb", kind: "linear", title: "q_b_proj" });
  g.kvn = cloneOp(kvn, { id: "kvn", kind: "norm", title: "kv_a_layernorm" });
  g.kvb = cloneOp(kvb, { id: "kvb", kind: "linear", title: "kv_b_proj" });
  g.rope = cloneOp(rope, { id: "rope", kind: "rope", title: "RoPE(interleaved)" });
  g.cache = cloneOp(cache, { id: "cache", kind: "cache", title: "MLA 压缩 KV cache" });
  if (variant === "moe-shared") {
    const [ish] = indexer;
    g.ishared = cloneOp(ish, { id: "ishared", kind: "route", title: "复用最近 full 层索引" });
  } else {
    const [iwqb, iwk, ikn, iquant, iscore] = indexer as Node[];
    g.iwqb = cloneOp(iwqb, { id: "iwqb", kind: "linear", title: "indexer wq_b" });
    g.iwk = cloneOp(iwk, { id: "iwk", kind: "linear", title: "wk_weights_proj(融合)" });
    g.ikn = cloneOp(ikn, { id: "ikn", kind: "norm", title: "k_norm = LayerNorm" });
    g.iquant = cloneOp(iquant, { id: "iquant", kind: "scale", title: "FP8 量化 + 权重折叠" });
    g.iscore = cloneOp(iscore, { id: "iscore", kind: "route", title: "打分 + Top-2048" });
  }
  g.qk = cloneOp(qk, { id: "qk", kind: "matmul", title: "QKᵀ/√256(候选 2048)" });
  g.sink = cloneOp(sink, { id: "sink", kind: "softmax", title: "softmax + learnable sink" });
  g.pv = cloneOp(pv, { id: "pv", kind: "matmul", title: "P·V" });
  g.gate = cloneOp(gate, { id: "gate", kind: "activation", title: "gated MLA(⊙ σ)" });
  g.oproj = cloneOp(oproj, { id: "oproj", kind: "linear", title: "o_proj" });
  if (variant === "dense-full") {
    const [gateup, act, down] = denseFfnNodes();
    g.gateup = cloneOp(gateup, { id: "gateup", kind: "linear", title: "gate/up 投影(L0 dense)" });
    g.act = cloneOp(act, { id: "act", kind: "activation", title: "SwiGLU(无 clamp)" });
    g.down = cloneOp(down, { id: "down", kind: "linear", title: "down 投影" });
  } else {
    const [router, select, experts, shared, sum] = moeNodes();
    g.router = cloneOp(router, { id: "router", kind: "linear", title: "FP32 router(256)" });
    g.select = cloneOp(select, { id: "select", kind: "route", title: "σ + Top-8 + ×2.827" });
    g.experts = cloneOp(experts, { id: "experts", kind: "activation", title: "256 routed experts(clamp 10)" });
    g.mshare = cloneOp(shared, { id: "shared", kind: "activation", title: "shared expert(无 clamp)" });
    g.msum = cloneOp(sum, { id: "sum", kind: "add", title: "routed ⊕ shared" });
  }
  void INPUT_OVERRIDES; void NEXT_BY_ID;
  return g;
}

// zoom graphs(与 detail 面板共用节点定义,仅重新指定紧凑标题)
export function attentionZoomNodes(variant: LayerVariant): string[] {
  const base = ["aproj", "asplit", "qan", "qb", "kvn", "kvb", "rope", "cache", "qk", "sink", "pv", "gate", "oproj"];
  const indexer = variant === "moe-shared" ? ["ishared"] : ["iwqb", "iwk", "ikn", "iquant", "iscore"];
  return [...indexer, ...base];
}
