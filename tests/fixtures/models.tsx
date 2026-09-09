import { useState } from "react";
import { createRoot } from "react-dom/client";
import Home from "../../app/page";
import { createModelRegistry, type ModelDefinition } from "../../app/models/registry";

function EncoderView({ onExpandedChange }: { onExpandedChange: (expanded: boolean) => void }) {
  const [count, setCount] = useState(0);
  return <section>
    <p data-test="architecture">Encoder fixture: {count}</p>
    <button data-test="increment" onClick={() => setCount(value => value + 1)}>Increment</button>
    <button data-test="expand" onClick={() => onExpandedChange(true)}>Expand</button>
  </section>;
}

const encoder: ModelDefinition = {
  id: "test-encoder", name: "Test Encoder", facts: [{ value: "encoder", label: "Architecture" }],
  resources: [{ label: "CODE", description: "Encoder source", url: "https://example.invalid/encoder" }],
  View: EncoderView,
  Reference: ({ onClose }) => <button data-test="reference" onClick={onClose}>Encoder reference</button>,
};
const decoder: ModelDefinition = {
  id: "test-decoder", name: "Test Decoder", facts: [{ value: "decoder", label: "Architecture" }], resources: [],
  View: () => <p data-test="architecture">Decoder fixture</p>,
};

// Imported only by the browser regression runner, never by the production registry.
export function mountFixtureAtlas() {
  document.getElementById("root")!.style.display = "none";
  const container = document.createElement("div");
  container.id = "fixture-atlas";
  document.body.append(container);
  createRoot(container).render(<Home registry={createModelRegistry([encoder, decoder])}/>);
}
