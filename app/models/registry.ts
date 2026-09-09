import type { ComponentType } from "react";

/** A model owns its architecture and evidence; the shell owns navigation and theme. */
export interface ModelDefinition {
  id: string;
  name: string;
  resources: readonly { label: string; description: string; url: string }[];
  facts: readonly { value: string; label: string }[];
  View: ComponentType<{ onExpandedChange: (expanded: boolean) => void }>;
  Reference?: ComponentType<{ onClose: () => void }>;
}

export function createModelRegistry(definitions: readonly ModelDefinition[]) {
  if (!definitions.length) throw new Error("Register at least one model");
  const models = Object.freeze([...definitions]);
  const byId = new Map<string, ModelDefinition>();
  for (const model of models) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(model.id)) throw new Error(`Invalid model id: ${model.id}`);
    if (byId.has(model.id)) throw new Error(`Duplicate model id: ${model.id}`);
    byId.set(model.id, model);
  }
  return { models, resolve: (id?: string | null) => byId.get(id ?? "") ?? models[0] };
}
