import type { ReactNode } from "react";
import type { ConfigGroup, ExpandedStage, FormulaTerm, IoBinding, OpNode, StageOverview } from "../atlas-shared";

// Contract between the Atlas shell (app/page.tsx) and a model module.
// A module owns every model-specific fact: header facts, config reference,
// operator-graph dataset, symbol maps, and the diagram layouts. The shell owns
// chrome that is shared across models: model switcher, detail panel tabs,
// config modal, theme toggle.

export type WorkbenchProps = {
  layerType: string;
  graph: Record<string, OpNode>;
  activeId: string;
  expanded: ExpandedStage;
  onExpand: (stage: ExpandedStage) => void;
  onHover: (node: OpNode) => void;
  onLeave: () => void;
  onSelect: (node: OpNode) => void;
};

export type NavigatorProps = { value: string; onChange: (next: string) => void };

export type CanvasHeading = { kicker: string; title: string };

export type ModelModule = {
  id: string;
  name: string;
  /** Header facts strip: total params, active params, context, checkpoint size. */
  facts: { total: string; active: string; context: string; checkpoint: string };
  /** Header resource links. */
  links: { codeUrl: string; codeLabel: string; weightsUrl: string; weightsLabel: string };
  /** vLLM commit the source links are pinned to; also shown in the footer. */
  vllmCommit: string;
  defaultLayerType: string;
  configGroups: readonly ConfigGroup[];
  configSymbols: Record<string, string>;
  graphFor(layerType: string): Record<string, OpNode>;
  /** Full input-binding list for a node (data inputs + weight inputs). */
  inputBindingsFor(node: OpNode): IoBinding[];
  nextFor(nodeId: string): string | undefined;
  symbolicShape(shape: string): string;
  stageOverviewFor(layerType: string, stage: Exclude<ExpandedStage, null>): StageOverview | null;
  formulaTermsFor(node: OpNode): readonly FormulaTerm[];
  formulaNoteFor(node: OpNode): string;
  /** Top overview strip (input → embedding → decoder stack → head). */
  Overview: () => ReactNode;
  /** Layer-type navigator shown under the overview strip. */
  Navigator: (props: NavigatorProps) => ReactNode;
  /** Heading for the layer canvas header. */
  canvasHeading(layerType: string): CanvasHeading;
  /** Decoder-layer workbench: collapsed column plus expanded stage zoom. */
  Workbench: (props: WorkbenchProps) => ReactNode;
};
