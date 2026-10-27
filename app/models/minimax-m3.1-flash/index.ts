import type { ModelDefinition } from "../registry";
import "./layout.css";
import { FlashView } from "./view";
import { HelpModal } from "./reference";
import { CODE_URL, TRANSFORMERS_REF_URL, WEIGHTS_URL, VLLM_COMMIT } from "./sources";

export const minimaxM31Flash = {
  id: "minimax-m3.1-flash",
  name: "MiniMax-M3.1-Flash",
  resources: [
    { label: "CODE", description: `vLLM @ ${VLLM_COMMIT.slice(0, 7)} · 已并入主线（PR #64111）`, url: CODE_URL },
    { label: "TRANSFORMERS", description: "同架构可读参考（minimax_m3_vl）", url: TRANSFORMERS_REF_URL },
    { label: "WEIGHTS", description: "Hugging Face · NVFP4+MXFP8", url: WEIGHTS_URL },
  ],
  facts: [
    { value: "428B", label: "模型总参数量" },
    { value: "23B", label: "每 token 激活参数" },
    { value: "1M", label: "最大上下文 token" },
    { value: "250 GB", label: "NVFP4+MXFP8 checkpoint" },
  ],
  View: FlashView,
  Reference: HelpModal,
} satisfies ModelDefinition;