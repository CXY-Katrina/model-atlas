export type EdgePort = "top" | "top-left" | "top-right" | "right" | "bottom" | "bottom-left" | "bottom-right" | "left";
export type GraphEdge = { from: string; to: string; fromPort?: EdgePort; toPort?: EdgePort; route?: "side-left" | "side-right" | "bus-left" | "bus-right"; approach?: number; departure?: number; fanout?: string };
export type EdgeTone = "data" | "weight" | "external" | "residual";
export type GraphPath = { d: string; tone: EdgeTone; marker?: boolean };

export type GraphAlignment = {node:string;between:string[]};
