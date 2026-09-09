import { useState } from "react";
import { CONFIG_GROUPS, configSymbol } from "./config";

export function HelpModal({onClose}:{onClose:()=>void}){
  const [activeGroup,setActiveGroup]=useState(0);
  const group=CONFIG_GROUPS[activeGroup];
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal reference-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="完整 config.json 参数与符号"><header><div><span>CONFIG REFERENCE</span><h2>完整 config.json · 参数与符号</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><nav className="config-tabs" role="tablist" aria-label="选择 config.json 分组">{CONFIG_GROUPS.map((item,index)=><button key={item.title} role="tab" aria-selected={activeGroup===index} className={activeGroup===index?"active":""} onClick={()=>setActiveGroup(index)}>{item.title}</button>)}</nav><div className="config-reference" role="tabpanel"><section><h3>{group.title}</h3><table><colgroup><col className="config-key-column"/><col className="config-symbol-column"/><col/></colgroup><thead><tr><th>参数</th><th>符号</th><th>值</th></tr></thead><tbody>{group.rows.map(([key,value])=><tr key={key}><th scope="row">{key}</th><td className="config-symbol"><code>{configSymbol(group.title,key)}</code></td><td>{value}</td></tr>)}</tbody></table></section></div></section></div>;
}
