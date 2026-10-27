import { createModelRegistry } from "./registry";
import { minimaxM3 } from "./minimax-m3";
import { minimaxM31Flash } from "./minimax-m3.1-flash";

// Register verified model modules here. No disabled placeholder models.
export const modelRegistry = createModelRegistry([minimaxM3, minimaxM31Flash]);
