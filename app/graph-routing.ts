export type RoutePoint = {x:number;y:number};

type RouteDirection = "vertical" | "horizontal" | "side-left" | "side-right";

type RouteGraphEdgeOptions = {
  source:RoutePoint;
  target:RoutePoint;
  direction:RouteDirection;
  obstacleBounds:{left:number;right:number};
  clearance:number;
};

const n=(value:number)=>Number(value.toFixed(2));

export function routeGraphEdge({source,target,direction,obstacleBounds,clearance}:RouteGraphEdgeOptions){
  const sx=n(source.x); const sy=n(source.y); const tx=n(target.x); const ty=n(target.y);
  if(direction==="vertical"){
    const sign=ty>=sy?1:-1;
    const terminalY=n(ty-sign*Math.min(14,Math.abs(ty-sy)/3));
    const midY=n((sy+terminalY)/2);
    return {path:`M ${sx} ${sy} C ${sx} ${midY}, ${tx} ${midY}, ${tx} ${terminalY} L ${tx} ${ty}`,rail:null};
  }
  if(direction==="horizontal"){
    const sign=tx>=sx?1:-1;
    const terminalX=n(tx-sign*Math.min(14,Math.abs(tx-sx)/3));
    const midX=n((sx+terminalX)/2);
    return {path:`M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${ty}, ${terminalX} ${ty} L ${tx} ${ty}`,rail:null};
  }

  const right=direction==="side-right";
  const rail=n(right?obstacleBounds.right+clearance:obstacleBounds.left-clearance);
  const horizontalDirection=right?1:-1;
  const verticalDirection=ty>=sy?1:-1;
  const radius=n(Math.min(12,Math.abs(rail-sx)/2,Math.abs(ty-sy)/2));
  const firstCornerX=n(rail-horizontalDirection*radius);
  const firstCornerY=n(sy+verticalDirection*radius);
  const secondCornerY=n(ty-verticalDirection*radius);
  const lastCornerX=n(rail-horizontalDirection*radius);
  const path=[
    `M ${sx} ${sy}`,
    `L ${firstCornerX} ${sy}`,
    `Q ${rail} ${sy}, ${rail} ${firstCornerY}`,
    `L ${rail} ${secondCornerY}`,
    `Q ${rail} ${ty}, ${lastCornerX} ${ty}`,
    `L ${tx} ${ty}`,
  ].join(" ");
  return {path,rail};
}
