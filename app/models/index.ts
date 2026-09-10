import { createModelRegistry } from "./registry";
import { minimaxM3 } from "./minimax-m3";
import { hy4Preview } from "./hy4-preview";

// Register verified model modules here. No disabled placeholder models.
export const modelRegistry = createModelRegistry([hy4Preview, minimaxM3]);
