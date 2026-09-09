export const VLLM_COMMIT = "b2f685834a6456197e7033966fdef52a23f1abcd";
const VLLM = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}`;

export const CODE_URL = `${VLLM}/vllm/models/hy_v4/nvidia/model.py`;
export const ATTENTION_URL = `${VLLM}/vllm/models/hy_v4/nvidia/attention.py`;
export const HC_URL = `${VLLM}/vllm/models/hy_v4/nvidia/hc.py`;
export const MOE_URL = `${VLLM}/vllm/models/hy_v4/nvidia/moe.py`;
export const MTP_URL = `${VLLM}/vllm/models/hy_v4/nvidia/mtp.py`;
export const FLASHMLA_SPARSE_URL = `${VLLM}/vllm/models/hy_v4/nvidia/flashmla_sparse.py`;
export const CONFIG_CLASS_URL = `${VLLM}/vllm/transformers_utils/configs/hy_v4.py`;
export const RUNNER_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py";
export const LINEAR_URL = `${VLLM}/vllm/model_executor/layers/linear.py`;
export const FUSED_MOE_URL = `${VLLM}/vllm/model_executor/layers/fused_moe/fused_moe.py`;

export const DECODER_IHC_FORWARD_URL = `${CODE_URL}#L187-L209`;
export const MODEL_FORWARD_URL = `${CODE_URL}#L356-L393`;
export const WEIGHT_MAPPING_URL = `${CODE_URL}#L620-L640`;
export const ATTENTION_FORWARD_URL = `${ATTENTION_URL}#L658-L734`;
export const INDEXER_PREPARE_URL = `${ATTENTION_URL}#L212-L263`;
export const SINK_FORCE_URL = `${ATTENTION_URL}#L635-L656`;
export const HC_PRE_URL = `${HC_URL}#L97-L139`;
export const HC_POST_URL = `${HC_URL}#L163-L186`;
export const HC_HEAD_URL = `${HC_URL}#L251-L277`;
export const MOE_FACTORY_URL = `${MOE_URL}#L152-L170`;
export const DENSE_FF_URL = `${MOE_URL}#L62-L66`;
export const MTP_FORWARD_URL = `${MTP_URL}#L369-L390`;
export const SINK_MATH_URL = `${FLASHMLA_SPARSE_URL}#L44-L51`;

export const HF_CONFIG_URL = "https://huggingface.co/tencent/Hy4-preview/raw/main/config.json";
export const WEIGHTS_URL = "https://huggingface.co/tencent/Hy4-preview";
export const ASCEND_TUTORIAL_URL = "https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Hy4-preview.md";

export const CONFIG_FETCH_NOTE = "HF tencent/Hy4-preview · config.json @ 2026-09-09";
