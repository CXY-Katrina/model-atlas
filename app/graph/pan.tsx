import { useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { GRAPH_GEOMETRY } from "./routing";

export function GraphPan({children}:{children:ReactNode}){
  const viewportRef=useRef<HTMLDivElement>(null);
  const contentRef=useRef<HTMLDivElement>(null);
  const dragRef=useRef({pointerId:-1,x:0,y:0,offsetX:0,offsetY:0});
  const [offset,setOffset]=useState({x:0,y:0});
  const [dragging,setDragging]=useState(false);
  const [canPan,setCanPan]=useState(false);
  useLayoutEffect(()=>{
    const viewport=viewportRef.current; const content=contentRef.current;
    if(!viewport||!content)return;
    const measure=()=>{
      const next=content.scrollWidth>viewport.clientWidth+1||content.scrollHeight>viewport.clientHeight+1;
      setCanPan(next);
      if(!next)setOffset({x:0,y:0});
    };
    const observer=new ResizeObserver(measure);
    observer.observe(viewport); observer.observe(content); measure();
    return()=>observer.disconnect();
  },[]);
  const clampOffset=(x:number,y:number)=>{
    const viewport=viewportRef.current; const content=contentRef.current;
    if(!viewport||!content)return {x,y};
    const padding=GRAPH_GEOMETRY.panPadding;
    const minX=Math.min(0,viewport.clientWidth-content.scrollWidth-padding);
    const minY=Math.min(0,viewport.clientHeight-content.scrollHeight-padding);
    return {x:Math.max(minX,Math.min(padding,x)),y:Math.max(minY,Math.min(0,y))};
  };
  const onPointerDown=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(!canPan)return;
    if((event.target as HTMLElement).closest("button,a"))return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current={pointerId:event.pointerId,x:event.clientX,y:event.clientY,offsetX:offset.x,offsetY:offset.y};
    setDragging(true);
  };
  const onPointerMove=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(dragRef.current.pointerId!==event.pointerId)return;
    setOffset(clampOffset(dragRef.current.offsetX+event.clientX-dragRef.current.x,dragRef.current.offsetY+event.clientY-dragRef.current.y));
  };
  const stopDragging=(event:ReactPointerEvent<HTMLDivElement>)=>{
    if(dragRef.current.pointerId!==event.pointerId)return;
    dragRef.current.pointerId=-1;
    setDragging(false);
  };
  return <div ref={viewportRef} className={`graph-pan-viewport ${canPan?"can-pan":"fits-page"} ${dragging?"is-dragging":""}`} aria-label={canPan?"可拖动的算子流程图":"完整显示的算子流程图"} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={stopDragging} onPointerCancel={stopDragging}>
    <div ref={contentRef} className="graph-pan-content" style={{transform:canPan?`translate3d(${offset.x}px,${offset.y}px,0)`:undefined}}>{children}</div>
    {canPan&&<span className="graph-pan-hint">抓住空白处拖动画布</span>}
  </div>;
}
