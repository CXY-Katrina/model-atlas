import { readFileSync, readdirSync } from "node:fs";

// Evidence contracts follow the model module, not the application's former monolith.
export function readModelSource() {
  const folder = new URL("../app/models/minimax-m3/", import.meta.url);
  const files = readdirSync(folder).filter(file => /\.(ts|tsx)$/.test(file) && file !== "data.ts");
  return [
    readFileSync(new URL("../app/models/types.ts", import.meta.url), "utf8"),
    ...files.map(file => readFileSync(new URL(file, folder), "utf8")),
    ...["surface", "pan", "nodes"].map(file => readFileSync(new URL(`../app/graph/${file}.tsx`, import.meta.url), "utf8")),
  ].join("\n");
}

export function readStyles() {
  const tokens = readFileSync(new URL("../app/styles/tokens.css", import.meta.url), "utf8");
  const variables = new Map([...tokens.slice(0, tokens.indexOf(".atlas-app.dark")).matchAll(/(--[\w-]+):\s*([^;]+);/g)].map(m => [m[1], m[2]]));
  let css = tokens + "\n" + ["styles/ui.css", "models/minimax-m3/layout.css"].map(file => readFileSync(new URL(`../app/${file}`, import.meta.url), "utf8")).join("\n");
  // Existing geometry contracts assert computed constants, while token contracts assert reuse.
  for (let i = 0; i < 4; i++) css = css.replace(/var\((--[\w-]+)\)/g, (value, key) =>
    /--(?:space-|radius-|text-|graph-node-|graph-group-|graph-row-|weight-border)/.test(key) ? variables.get(key) ?? value : value);
  return css.replaceAll(':where([data-model="minimax-m3"])', "").replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\s*([{};,])\s*/g, "$1").replace(/:\s+/g, ":").replace(/@media \(/g,"@media(").replace(/;}/g, "}").replace(/\s+!important/g,"!important").trim();
}
