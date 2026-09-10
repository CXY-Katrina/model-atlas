import type { ModelDefinition } from "../registry";
import "./layout.css";
import { Hy4View } from "./view";
import { HelpModal } from "./reference";
import { VLLM_COMMIT, WEIGHTS_URL, ASCEND_TUTORIAL_URL, HF_CONFIG_URL } from "./sources";

export const hy4Preview = {
  id: "hy4-preview",
  name: "Hy4-preview",
  resources: [
    { label: "CODE", description: `vLLM @ ${VLLM_COMMIT.slice(0, 7)}`, url: `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/hy_v4/nvidia/model.py` },
    { label: "WEIGHTS", description: "Hugging Face · 131 shards", url: WEIGHTS_URL },
    { label: "CONFIG", description: "官方 config.json", url: HF_CONFIG_URL },
    { label: "ASCEND", description: "W8A8 部署指引(A3)", url: ASCEND_TUTORIAL_URL },
  ],
  facts: [
    { value: "770B", label: "模型总参数量" },
    { value: "~49B", label: "每 token 激活参数" },
    { value: "1M", label: "最大上下文 token" },
    { value: "78", label: "iHC 4 通道层数" },
  ],
  View: Hy4View,
  Reference: HelpModal,
} satisfies ModelDefinition;
