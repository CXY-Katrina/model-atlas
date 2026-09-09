import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const profile=await mkdtemp(join(tmpdir(),'minimax-cache-check-'));
const browser=spawn(process.env.BROWSER_PATH ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
try{
 let port;
 for(let i=0;i<80;i++){try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break}catch{await pause(250)}}
 if(!port)throw Error('Browser did not start');
 const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 const socket=new WebSocket(tabs.find(tab=>tab.type==='page').webSocketDebuggerUrl);
 await new Promise(resolve=>socket.addEventListener('open',resolve,{once:true}));
 let seq=0;const pending=new Map();
 socket.addEventListener('message',({data})=>{const m=JSON.parse(data);if(pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result)}});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 await send('Emulation.setDeviceMetricsOverride',{width:2048,height:1320,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:process.env.TEST_URL ?? 'http://127.0.0.1:5173/'});await pause(1500);
 for(let i=0;i<40;i++){if(await evaluate(`!!document.querySelector('.layer-type-options .sparse')`))break;await pause(250)}
 console.log(await evaluate(`JSON.stringify({url:location.href,body:document.body.innerText.slice(0,800)})`));
 await evaluate(`document.querySelector('.layer-type-options .sparse').click()`);await pause(250);
 await evaluate(`document.querySelector('.attention-stage').click()`);await pause(800);
 const metrics=await evaluate(`(()=>{const ids=['attn-cache','attn-paged-k','attn-paged-v','attn-qk','attn-p','attn-pv'];return Object.fromEntries(ids.map(id=>{const r=document.querySelector('[data-graph-id="'+id+'"]').getBoundingClientRect();return [id,{x:r.x,y:r.y,cx:r.x+r.width/2,width:r.width,height:r.height}]}))})()`);
 console.log(JSON.stringify(metrics));
 const indexLines=await evaluate(`(()=>{const root=document.querySelector('.connected-attention-graph');const origin=root.getBoundingClientRect();const ids=['attn-kidx','attn-idxnorm','attn-idxcache','attn-idxscore'];const rects=ids.map(id=>root.querySelector('[data-graph-id="'+id+'"]').getBoundingClientRect());return rects.slice(1).map((r,i)=>{const source=rects[i];const sx=source.x+source.width/2-origin.x;const sy=source.bottom-origin.y;const path=[...root.querySelectorAll('.edge-lines path')].map(n=>n.getAttribute('d')).find(d=>{const m=d.match(/^M ([\\d.-]+) ([\\d.-]+)/);return m&&Math.abs(Number(m[1])-sx)<.02&&Math.abs(Number(m[2])-sy)<.02});return {from:ids[i],to:ids[i+1],path,dx:r.x+r.width/2-(source.x+source.width/2)}})})()`);
 for(const line of indexLines){assert.ok(Math.abs(line.dx)<.02,JSON.stringify(line));assert.match(line.path,/^M [\d.-]+ [\d.-]+ L [\d.-]+ [\d.-]+$/)}console.log('straight index chain',indexLines);
 await evaluate(`document.querySelector('.graph-pan-content').style.transform='translate(-350px,0px)'`);await pause(200);
 const upper=await send('Page.captureScreenshot',{format:'png'});await writeFile(join(tmpdir(),'minimax-group-padding.png'),Buffer.from(upper.data,'base64'));
 console.log('frames',await evaluate(`JSON.stringify([...document.querySelectorAll('.qkv-lanes>section,.index-ribbon,.score-pipeline')].map(group=>{const r=group.getBoundingClientRect();const nodes=[...group.querySelectorAll('[data-graph-id]')].map(n=>n.getBoundingClientRect());return {name:group.querySelector('header')?.textContent,width:r.width,columns:getComputedStyle(group).gridTemplateColumns,left:Math.min(...nodes.map(n=>n.left))-r.left,right:r.right-Math.max(...nodes.map(n=>n.right))}}))`));
 await evaluate(`document.querySelector('.graph-pan-content').style.transform='translate(0px,-750px)'`);await pause(250);
 const cachePosition=await evaluate(`(()=>{const r=document.querySelector('[data-graph-id="attn-cache"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
 await send('Input.dispatchMouseEvent',{type:'mouseMoved',...cachePosition});await pause(400);
 console.log('after-hover',await evaluate(`(()=>{const ids=['attn-cache','attn-paged-k','attn-paged-v','attn-qk'];return Object.fromEntries(ids.map(id=>{const r=document.querySelector('[data-graph-id="'+id+'"]').getBoundingClientRect();return [id,r.x+r.width/2]}))})()`));
 const capture=await send('Page.captureScreenshot',{format:'png'});
 const screenshot=join(tmpdir(),'minimax-cache-check.png');await writeFile(screenshot,Buffer.from(capture.data,'base64'));console.log(screenshot);
 for(const width of [1440,1920,2048]){
  await send('Emulation.setDeviceMetricsOverride',{width,height:1320,deviceScaleFactor:1,mobile:false});await pause(350);
  const geometry=await evaluate(`(()=>{
   const root=document.querySelector('.connected-attention-graph');const origin=root.getBoundingClientRect();
   const rect=id=>document.querySelector('[data-graph-id="'+id+'"]').getBoundingClientRect();
   const center=id=>{const r=rect(id);return r.x+r.width/2};const cache=rect('attn-cache');
   const incoming=[...root.querySelectorAll('.edge-lines path')].map(p=>p.getAttribute('d').match(/L ([\\d.-]+) ([\\d.-]+)$/)).filter(m=>m&&Math.abs(Number(m[2])-(cache.top-origin.top))<2).map(m=>Number(m[1])+origin.left);
   const panel=root.querySelector('.score-pipeline').getBoundingClientRect();
   return {leftPadding:rect('attn-bounds').left-panel.left,rightPadding:panel.right-rect('attn-paged-v').right,cacheError:Math.abs(center('attn-cache')-(center('attn-paged-k')+center('attn-paged-v'))/2),straightError:Math.abs(center('attn-qk')-center('attn-paged-k')),inputs:incoming.length,inputError:Math.abs(incoming.reduce((sum,x)=>sum+x,0)/incoming.length-center('attn-cache'))};
  })()`);
  assert.ok(geometry.cacheError<.02,JSON.stringify(geometry));assert.ok(geometry.straightError<.02,JSON.stringify(geometry));assert.equal(geometry.inputs,2);assert.ok(geometry.inputError<.02,JSON.stringify(geometry));console.log(width,geometry);
 }
 for(const type of ['sparse','dense']){
  if(type==='dense'){await evaluate(`document.querySelector('.layer-type-options .dense').click()`);await pause(250);await evaluate(`document.querySelector('.attention-stage').click()`);await pause(400)}
  for(const width of [1440,1920,2048]){
   await send('Emulation.setDeviceMetricsOverride',{width,height:1320,deviceScaleFactor:1,mobile:false});await pause(250);
   const groups=await evaluate(`([...document.querySelectorAll('.qkv-lanes>section,.index-ribbon,.score-pipeline')].map(group=>{const r=group.getBoundingClientRect();const nodes=[...group.querySelectorAll('[data-graph-id]')].map(n=>n.getBoundingClientRect());return {name:group.querySelector('header')?.textContent,left:Math.min(...nodes.map(n=>n.left))-r.left,right:r.right-Math.max(...nodes.map(n=>n.right))}}))`);
   for(const group of groups){assert.ok(Math.abs(group.left-25)<.05,JSON.stringify({type,width,group}));assert.ok(Math.abs(group.right-25)<.05,JSON.stringify({type,width,group}))}
   const textBounds=await evaluate(`['attn-packed','attn-idxslots'].flatMap(id=>{const node=document.querySelector('[data-graph-id="'+id+'"]');if(!node)return [];const rect=node.getBoundingClientRect();return [...node.children].map(child=>{const r=child.getBoundingClientRect();return {id,text:child.textContent,top:r.top-rect.top,bottom:rect.bottom-r.bottom,left:r.left-rect.left,right:rect.right-r.right}})})`);
   for(const item of textBounds)assert.ok(Math.min(item.top,item.bottom,item.left,item.right)>=7,JSON.stringify({width,item}));
   console.log(`${type} ${width}: ${groups.length} frames have equal 24px padding + 1px border`);
  }
 }

 // All expanded stage families and theme toggles remain available after the split.
 for(const type of ['dense','sparse']){
  await evaluate(`document.querySelector('.layer-type-options .'+'${type}').click()`);await pause(100);
  await evaluate(`document.querySelector('.ffn-stage').click()`);await pause(300);
  for(const dark of [true,false]){
   await evaluate(`document.querySelector('.theme-button').click()`);await pause(100);
   const state=await evaluate(`({dark:document.querySelector('.atlas-app').classList.contains('dark'),nodes:document.querySelectorAll('.graph-surface [data-graph-id]').length,paths:document.querySelectorAll('.edge-lines path').length})`);
   assert.equal(state.dark,dark);assert.ok(state.nodes>5);assert.ok(state.paths>5);
  }
 }
 await evaluate(`import('/tests/fixtures/models.tsx').then(m=>m.mountFixtureAtlas())`);await pause(300);
 const fixtureState=()=>evaluate(`(()=>{const root=document.querySelector('#fixture-atlas');return {model:root.querySelector('main').dataset.model,text:root.querySelector('[data-test="architecture"]').textContent,dark:root.querySelector('main').classList.contains('dark'),expanded:root.querySelector('main').classList.contains('stage-expanded'),help:!!root.querySelector('[data-test="reference"]'),helpDisabled:root.querySelector('.help-button').disabled,links:root.querySelectorAll('.resource-links a').length,title:document.title}})()`);
 const choose=async id=>{console.log('choose',await evaluate(`(()=>{const select=document.querySelector('#fixture-atlas select');select.value='${id}';select.dispatchEvent(new Event('change',{bubbles:true}));return {value:select.value,options:select.innerHTML}})()`));await pause(300);console.log('selected',await fixtureState())};
 await evaluate(`['[data-test="increment"]','[data-test="expand"]','.help-button','.theme-button'].forEach(s=>document.querySelector('#fixture-atlas '+s).click())`);await pause(150);
 let state=await fixtureState();assert.equal(state.text,'Encoder fixture: 1');assert.ok(state.expanded&&state.help&&state.dark);
 await choose('test-decoder');state=await fixtureState();assert.equal(state.text,'Decoder fixture');assert.equal(state.links,0);assert.ok(state.helpDisabled);assert.ok(!state.expanded&&!state.help&&state.dark);assert.equal(state.title,'Model Atlas · Test Decoder');
 await choose('test-encoder');state=await fixtureState();assert.equal(state.text,'Encoder fixture: 0');assert.equal(state.links,1);assert.ok(!state.help&&!state.expanded);
 await evaluate('history.back()');await pause(200);assert.equal((await fixtureState()).model,'test-decoder');
 console.log('Model switching, history, state reset, resource isolation and themes passed');
 socket.close();
}finally{browser.kill()}
