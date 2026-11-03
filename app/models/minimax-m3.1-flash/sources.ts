// MiniMax-M3.1-Flash 的 vLLM 支持已通过 PR #64111 并入 main（原 squash PR #63232 被拆分）。
// 此处 pin 的是该支持代码的一致性快照 commit：全部行号锚点已针对它逐段核对；
// main 上的实现此后仍在演进（MegaMoE staging、DecoderLayer.forward 拆分等），行号体系不同，
// 因此不跟随 main 漂移。
export const VLLM_COMMIT = "8fae0ff58bae8bde68b52bf3e547432250763e5c";
export const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
export const INDEXER_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/common/indexer.py`;
export const SPARSE_ATTENTION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/common/sparse_attention.py`;
export const MSA_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py`;
export const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash";
export const HF_CONFIG_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/config.json";
export const VLLM_BLOG_URL = "https://vllm.ai/blog/2026-10-23-minimax-m31-flash";
export const TRANSFORMERS_REF_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py";
export const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
export const ACTIVATION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/activation.py`;
export const LINEAR_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/linear.py`;
export const VLLM_INDEXER_URL = INDEXER_URL;
export const VLLM_SPARSE_ATTENTION_URL = SPARSE_ATTENTION_URL;
export const NORM_FORWARD_URL = `${CODE_URL}#L134-L146`;
export const DECODER_FORWARD_URL = `${CODE_URL}#L764-L790`;
export const FLASHINFER_GEMMA_NORM_URL = "https://docs.flashinfer.ai/generated/flashinfer.norm.gemma_rmsnorm.html";
export const TRANSFORMERS_MINIMAX_M3_URL = `${TRANSFORMERS_REF_URL}#L365`;
export const TRANSFORMERS_MOE_URL = `${TRANSFORMERS_REF_URL}#L202-L239`;
export const TRANSFORMERS_QKV_PROJECTION_URL = `${TRANSFORMERS_REF_URL}#L422-L445`;
export const TRANSFORMERS_INDEX_PROJECTION_URL = `${TRANSFORMERS_REF_URL}#L538-L563`;
export const TRANSFORMERS_SPARSE_ATTENTION_URL = `${TRANSFORMERS_REF_URL}#L447-L489`;
export const TRANSFORMERS_INDEX_SELECTION_URL = `${TRANSFORMERS_REF_URL}#L552-L597`;
export const TRANSFORMERS_BLOCK_MASK_URL = `${TRANSFORMERS_REF_URL}#L599-L635`;