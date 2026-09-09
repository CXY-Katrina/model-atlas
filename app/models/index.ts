import { createModelRegistry } from "./registry";
import { minimaxM3 } from "./minimax-m3";

// Register verified model modules here. No disabled placeholder models.
export const modelRegistry = createModelRegistry([minimaxM3]);
