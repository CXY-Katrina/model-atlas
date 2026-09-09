import type { ModelDefinition } from "../registry";
import "./layout.css";
import { MiniMaxView } from "./view";
import { HelpModal } from "./reference";
import { CODE_URL, TRANSFORMERS_MOE_URL, WEIGHTS_URL, VLLM_COMMIT } from "./sources";

export const minimaxM3 = {
  id: "minimax-m3",
  name: "MiniMax-M3",
  resources: [
    { label: "CODE", description: `vLLM @ ${VLLM_COMMIT.slice(0, 7)}`, url: CODE_URL },
    { label: "TRANSFORMERS", description: "MiniMax-M3 readable reference", url: TRANSFORMERS_MOE_URL },
    { label: "WEIGHTS", description: "Hugging Face · 59 shards", url: WEIGHTS_URL },
  ],
  facts: [
    { value: "428B", label: "模型总参数量" },
    { value: "23B", label: "每 token 激活参数" },
    { value: "1M", label: "最大上下文 token" },
    { value: "869 GB", label: "BF16 checkpoint" },
  ],
  View: MiniMaxView,
  Reference: HelpModal,
} satisfies ModelDefinition;
