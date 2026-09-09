import { useEffect, useState } from "react";
import { modelRegistry } from "./models";
import type { createModelRegistry } from "./models/registry";

type Registry = ReturnType<typeof createModelRegistry>;

export default function Home({ registry = modelRegistry }: { registry?: Registry }) {
  const [modelId, setModelId] = useState(() => registry.resolve(
    typeof window === "undefined" ? null : new URL(window.location.href).searchParams.get("model"),
  ).id);
  const [dark, setDark] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [help, setHelp] = useState(false);
  const model = registry.resolve(modelId);
  const { View, Reference } = model;

  const selectModel = (id: string) => {
    setModelId(registry.resolve(id).id);
    setExpanded(false);
    setHelp(false);
  };

  useEffect(() => {
    document.title = `Model Atlas · ${model.name}`;
  }, [model.name]);

  useEffect(() => {
    const onPopState = () => selectModel(new URL(window.location.href).searchParams.get("model") ?? "");
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [registry]);

  const changeModel = (id: string) => {
    selectModel(id);
    const url = new URL(window.location.href);
    url.searchParams.set("model", registry.resolve(id).id);
    window.history.pushState(null, "", url);
  };

  return <main data-model={model.id} className={`atlas-app ${dark ? "dark" : ""} ${expanded ? "stage-expanded" : ""}`}>
    <header className="app-header">
      <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>Model Atlas</b></div></div>
      <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value={model.id} onChange={event => changeModel(event.target.value)}>
        {registry.models.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label>
      <nav className="resource-links">{model.resources.map(resource => <a key={resource.label} href={resource.url} target="_blank" rel="noreferrer"><b>{resource.label} ↗</b><small>{resource.description}</small></a>)}</nav>
      <div className="model-facts">{model.facts.map(fact => <span key={fact.label}><b>{fact.value}</b><small>{fact.label}</small></span>)}</div>
      <button className="help-button" disabled={!Reference} onClick={() => setHelp(true)} aria-label="查看参数和符号说明">?</button>
      <button className="theme-button" onClick={() => setDark(value => !value)} aria-label="切换明暗主题">{dark ? "☀" : "☾"}</button>
    </header>
    <View key={model.id} onExpandedChange={setExpanded}/>
    {help && Reference && <Reference key={`${model.id}-reference`} onClose={() => setHelp(false)}/>}
  </main>;
}
