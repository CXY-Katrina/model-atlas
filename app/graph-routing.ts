export type RoutePoint = {x:number;y:number};

type RouteDirection = "vertical" | "horizontal" | "side-left" | "side-right" | "bus-left" | "bus-right";

type RouteGraphEdgeOptions = {
  source:RoutePoint;
  target:RoutePoint;
  direction:RouteDirection;
  obstacleBounds:{left:number;right:number};
  clearance:number;
  approach?:number;
  departure?:number;
};

const n=(value:number)=>Number(value.toFixed(2));

export function routeGraphEdge({source,target,direction,obstacleBounds,clearance,approach=14,departure=24}:RouteGraphEdgeOptions){
  const sx=n(source.x); const sy=n(source.y); const tx=n(target.x); const ty=n(target.y);
  if(direction==="vertical"){
    const sign=ty>=sy?1:-1;
    if(sx===tx)return {path:`M ${sx} ${sy} L ${tx} ${ty}`,rail:null};
    const gap=Math.abs(ty-sy);
    const targetLane=Math.min(approach,gap/2);
    const terminalY=n(ty-sign*targetLane);
    const horizontalSign=tx>=sx?1:-1;
    const finalStraight=Math.min(14,targetLane*.45);
    const radius=n(Math.min(24,approach*.55,Math.abs(tx-sx)/2,Math.abs(terminalY-sy),Math.max(0,targetLane-finalStraight)));
    const path=[
      `M ${sx} ${sy}`,
      `L ${sx} ${n(terminalY-sign*radius)}`,
      `Q ${sx} ${terminalY}, ${n(sx+horizontalSign*radius)} ${terminalY}`,
      `L ${n(tx-horizontalSign*radius)} ${terminalY}`,
      `Q ${tx} ${terminalY}, ${tx} ${n(terminalY+sign*radius)}`,
      `L ${tx} ${ty}`,
    ].join(" ");
    return {path,rail:null};
  }
  if(direction==="bus-left"||direction==="bus-right"){
    const right=direction==="bus-right";
    const sign=ty>=sy?1:-1;
    const rail=n(right?obstacleBounds.right+clearance:obstacleBounds.left-clearance);
    const horizontalSign=right?1:-1;
    const terminalY=n(ty-sign*Math.min(approach,Math.abs(ty-sy)/3));
    const junctionY=n(sy+sign*Math.min(departure,Math.abs(terminalY-sy)/3));
    const radius=n(Math.min(14,Math.abs(rail-sx)/2,Math.abs(terminalY-junctionY)/3,Math.abs(junctionY-sy)/2));
    const path=[
      `M ${sx} ${sy}`,
      `L ${sx} ${n(junctionY-sign*radius)}`,
      `Q ${sx} ${junctionY}, ${n(sx+horizontalSign*radius)} ${junctionY}`,
      `L ${n(rail-horizontalSign*radius)} ${junctionY}`,
      `Q ${rail} ${junctionY}, ${rail} ${n(junctionY+sign*radius)}`,
      `L ${rail} ${n(terminalY-sign*radius)}`,
      `Q ${rail} ${terminalY}, ${n(rail-horizontalSign*radius)} ${terminalY}`,
      `L ${n(tx+horizontalSign*radius)} ${terminalY}`,
      `Q ${tx} ${terminalY}, ${tx} ${n(terminalY+sign*radius)}`,
      `L ${tx} ${ty}`,
    ].join(" ");
    return {path,rail};
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
