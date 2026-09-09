export type Tone = "norm" | "projection" | "attention" | "index" | "moe" | "vision" | "output";

export type Weight = {
  key: string;
  shape: string;
  dtype: "BF16" | "F32";
  shard: string;
  runtime?: string;
  note?: string;
  params?: string;
};

export type Node = {
  id: string;
  tone: Tone;
  kicker: string;
  title: string;
  summary: string;
  input: string;
  inputShape: string;
  output: string;
  outputShape: string;
  formula: string;
  formulaNote: string;
  runtime: string;
  source: string;
  sourceUrl: string;
  code: string;
  weights: Weight[];
};


export type Tab = "io" | "formula" | "code";
export type OpKind = "io" | "norm" | "linear" | "split" | "rope" | "matmul" | "scale" | "mask" | "softmax" | "activation" | "route" | "cache" | "add";
export type BindingKind = "upstream" | "external" | "weight";
export type IoBinding = { kind: BindingKind; label: string; shape: string; from: string; note?: string };
export type CodeSection = { stage: string; title: string; location: string; code: string; url?: string };
export type CodeSymbol = { symbol: string; resolvesTo: string; meaning: string };
export type CodeDetail = { sections: CodeSection[]; symbols: CodeSymbol[] };
export type OpNode = Node & { kind: OpKind; latex?: string; codeSections?: CodeSection[]; codeSymbols?: CodeSymbol[] };
