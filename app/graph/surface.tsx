import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { GRAPH_GEOMETRY, routeGraphEdge, routeGraphFanout } from "./routing";
import type { GraphAlignment, GraphEdge, GraphPath, EdgePort, EdgeTone } from "./types";

export function GraphSurface({edges,className,children,alignments=[]}:{edges:GraphEdge[];className:string;children:ReactNode;alignments?:GraphAlignment[]}){
  const rootRef=useRef<HTMLDivElement>(null);
  const markerId=`graph-arrow-${useId().replace(/:/g,"")}`;
  const serializedEdges=JSON.stringify(edges);
  const serializedAlignments=JSON.stringify(alignments);
  const edgeKey=edges.map(edge=>`${edge.from}:${edge.fromPort??"bottom"}>${edge.to}:${edge.toPort??"top"}:${edge.route??"direct"}:${edge.fanout??"single"}`).join("|");
  const [paths,setPaths]=useState<GraphPath[]>([]);
  useLayoutEffect(()=>{
    const root=rootRef.current;
    if(!root)return;
    let frame=0;
    const point=(rect:DOMRect,port:EdgePort,rootRect:DOMRect)=>{
      const x=rect.left-rootRect.left; const y=rect.top-rootRect.top;
      if(port==="top")return [x+rect.width/2,y];
      if(port==="top-left")return [x+rect.width*GRAPH_GEOMETRY.portNear,y];
      if(port==="top-right")return [x+rect.width*GRAPH_GEOMETRY.portFar,y];
      if(port==="right")return [x+rect.width,y+rect.height/2];
      if(port==="left")return [x,y+rect.height/2];
      if(port==="bottom-left")return [x+rect.width*GRAPH_GEOMETRY.portNear,y+rect.height];
      if(port==="bottom-right")return [x+rect.width*GRAPH_GEOMETRY.portFar,y+rect.height];
      return [x+rect.width/2,y+rect.height];
    };
    const measure=()=>{
      // Align actual card centers before routing; preserve alignment as text and viewport sizes change.
      for(const alignment of JSON.parse(serializedAlignments) as GraphAlignment[]){
        const node=root.querySelector<HTMLElement>(`[data-graph-id="${alignment.node}"]`);
        const references=alignment.between.map(id=>root.querySelector<HTMLElement>(`[data-graph-id="${id}"]`));
        if(!node||references.some(reference=>!reference))continue;
        const centers=references.map(reference=>{const rect=reference!.getBoundingClientRect();return rect.left+rect.width/2});
        const rect=node.getBoundingClientRect();
        const property=`--graph-shift-${alignment.node}`;
        const currentShift=parseFloat(root.style.getPropertyValue(property))||0;
        const shift=currentShift+centers.reduce((sum,x)=>sum+x,0)/centers.length-(rect.left+rect.width/2);
        root.style.setProperty(property,`${shift.toFixed(2)}px`);
      }
      const rootRect=root.getBoundingClientRect();
      const currentEdges=JSON.parse(serializedEdges) as GraphEdge[];
      const nodeRects=[...root.querySelectorAll<HTMLElement>("[data-graph-id]")].map(node=>node.getBoundingClientRect());
      const obstacleBounds={
        left:Math.min(...nodeRects.map(rect=>rect.left-rootRect.left)),
        right:Math.max(...nodeRects.map(rect=>rect.right-rootRect.left)),
      };
      const endpointCounts=new Map<string,number>();
      currentEdges.forEach(edge=>{
        const fromPort=edge.fromPort??"bottom"; const toPort=edge.toPort??"top";
        const fromKey=`from:${edge.from}:${fromPort}`; const toKey=`to:${edge.to}:${toPort}`;
        endpointCounts.set(fromKey,(endpointCounts.get(fromKey)??0)+1);
        endpointCounts.set(toKey,(endpointCounts.get(toKey)??0)+1);
      });
      const mergeLanes=new Map<string,{count:number;approach:number;maxApproach:number}>();
      for(const edge of currentEdges){
        if(edge.route||edge.fanout||edge.departure!==undefined)continue;
        if(!(edge.fromPort??"bottom").startsWith("bottom")||!(edge.toPort??"top").startsWith("top"))continue;
        const source=root.querySelector<HTMLElement>(`[data-graph-id="${edge.from}"]`);
        const target=root.querySelector<HTMLElement>(`[data-graph-id="${edge.to}"]`);
        if(!source||!target)continue;
        const gap=target.getBoundingClientRect().top-source.getBoundingClientRect().bottom;
        if(gap<=0)continue;
        const lane=mergeLanes.get(edge.to)??{count:0,approach:0,maxApproach:Infinity};
        lane.count++;
        lane.approach=Math.max(lane.approach,edge.approach??30);
        lane.maxApproach=Math.min(lane.maxApproach,gap/2);
        mergeLanes.set(edge.to,lane);
      }
      const handledFanouts=new Set<string>();
      const next=currentEdges.flatMap(edge=>{
        const source=root.querySelector<HTMLElement>(`[data-graph-id="${edge.from}"]`);
        const target=root.querySelector<HTMLElement>(`[data-graph-id="${edge.to}"]`);
        if(!source||!target)return [];
        const fromPort=edge.fromPort??"bottom"; const toPort=edge.toPort??"top";
        const sourceRect=source.getBoundingClientRect(); const targetRect=target.getBoundingClientRect();
        const [sx,sy]=point(sourceRect,fromPort,rootRect);
        const [tx,ty]=point(targetRect,toPort,rootRect);
        if(edge.fanout){
          if(handledFanouts.has(edge.fanout))return [];
          handledFanouts.add(edge.fanout);
          const grouped=currentEdges.filter(candidate=>candidate.fanout===edge.fanout);
          const targets=grouped.flatMap(candidate=>{
            const groupedTarget=root.querySelector<HTMLElement>(`[data-graph-id="${candidate.to}"]`);
            if(!groupedTarget)return [];
            const [x,y]=point(groupedTarget.getBoundingClientRect(),candidate.toPort??"top",rootRect);
            return [{x,y}];
          });
          const tone:EdgeTone=source.classList.contains("tensor-weight")?"weight":source.classList.contains("tensor-side")?"external":"data";
          return routeGraphFanout({source:{x:sx,y:sy},targets,departure:edge.departure??72}).map(route=>({d:route.path,tone,marker:route.arrow}));
        }
        const direction=edge.route??(fromPort==="right"||fromPort==="left"||toPort==="right"||toPort==="left"?"horizontal":"vertical");
        const safeClearance=direction==="side-left"||direction==="bus-left"
          ?Math.min(GRAPH_GEOMETRY.clearance,Math.max(4,obstacleBounds.left-GRAPH_GEOMETRY.arrowClearance))
          :direction==="side-right"||direction==="bus-right"
            ?Math.min(GRAPH_GEOMETRY.clearance,Math.max(4,rootRect.width-obstacleBounds.right-GRAPH_GEOMETRY.arrowClearance))
            :GRAPH_GEOMETRY.clearance;
        const targetConnections=endpointCounts.get(`to:${edge.to}:${toPort}`)??1;
        const sourceConnections=endpointCounts.get(`from:${edge.from}:${fromPort}`)??1;
        const mergeLane=direction==="vertical"?mergeLanes.get(edge.to):undefined;
        const approach=mergeLane&&mergeLane.count>1
          ?Math.min(mergeLane.approach,mergeLane.maxApproach)
          :edge.approach??(targetConnections>1?48:sourceConnections>1?38:toPort==="top-left"||toPort==="top-right"?30:18);
        const tone:EdgeTone=edge.route?.startsWith("side-")
          ?"residual"
          :source.classList.contains("tensor-weight")
            ?"weight"
            :source.classList.contains("tensor-side")
              ?"external"
              :"data";
        return [{d:routeGraphEdge({source:{x:sx,y:sy},target:{x:tx,y:ty},direction,obstacleBounds,clearance:safeClearance,approach,departure:edge.departure}).path,tone,marker:true}];
      });
      setPaths(next);
    };
    const observer=new ResizeObserver(()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(measure)});
    observer.observe(root);
    root.querySelectorAll<HTMLElement>("[data-graph-id]").forEach(node=>observer.observe(node));
    measure();
    frame=requestAnimationFrame(measure);
    return()=>{cancelAnimationFrame(frame);observer.disconnect()};
  },[serializedEdges,serializedAlignments]);
  const tones:EdgeTone[]=["data","weight","external","residual"];
  return <div ref={rootRef} className={`graph-surface ${className}`}>{children}<svg className="graph-connectors" aria-hidden="true"><defs>{tones.map(tone=><marker key={tone} id={`${markerId}-${tone}`} className={`edge-marker edge-marker-${tone}`} markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0.5 0.8 L 8.5 5 L 0.5 9.2 Z"/></marker>)}</defs><g className="edge-halos">{paths.map((path,index)=><path key={`${edgeKey}-halo-${index}`} className="edge-halo" d={path.d}/>)}</g><g className="edge-lines">{paths.map((path,index)=><path key={`${edgeKey}-line-${index}`} className={`edge-line edge-${path.tone}`} d={path.d} markerEnd={path.marker===false?undefined:`url(#${markerId}-${path.tone})`}/>)}</g></svg></div>;
}
