/** Model-independent routing policy; each model chooses its approach/departure lanes. */
export const GRAPH_GEOMETRY = {
  portNear: 0.34,
  portFar: 0.66,
  clearance: 24,
  cornerRadius: 24,
  arrowClearance: 8,
  straightTolerance: 0.05,
  panPadding: 28,
} as const;

export type RoutePoint = {x:number;y:number};

export type FanoutRoute = {
  path:string;
  arrow:boolean;
  role:"trunk"|"rail"|"drop";
};

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

export function routeGraphFanout({source,targets,departure=48,radius=GRAPH_GEOMETRY.cornerRadius}:{
  source:RoutePoint;
  targets:RoutePoint[];
  departure?:number;
  radius?:number;
}):FanoutRoute[]{
  if(targets.length===0)return [];
  const sx=n(source.x); const sy=n(source.y);
  const direction=targets.reduce((sum,target)=>sum+target.y,0)/targets.length>=sy?1:-1;
  const nearestGap=Math.min(...targets.map(target=>Math.abs(target.y-sy)));
  const railDistance=Math.min(Math.max(8,departure),Math.max(8,nearestGap-8));
  const railY=n(sy+direction*railDistance);
  const xs=targets.map(target=>n(target.x));
  const minX=Math.min(...xs); const maxX=Math.max(...xs);
  const leftRadius=n(Math.min(radius,Math.abs(sx-minX),railDistance/2));
  const rightRadius=n(Math.min(radius,Math.abs(maxX-sx),railDistance/2));
  const trunkRadius=Math.max(leftRadius,rightRadius);
  const routes:FanoutRoute[]=[{
    path:`M ${sx} ${sy} L ${sx} ${n(railY-direction*trunkRadius)}`,
    arrow:false,
    role:"trunk",
  }];

  if(minX<sx){
    routes.push({
      path:`M ${sx} ${n(railY-direction*leftRadius)} Q ${sx} ${railY}, ${n(sx-leftRadius)} ${railY} L ${n(minX+leftRadius)} ${railY} Q ${minX} ${railY}, ${minX} ${n(railY+direction*leftRadius)}`,
      arrow:false,
      role:"rail",
    });
  }
  if(maxX>sx){
    routes.push({
      path:`M ${sx} ${n(railY-direction*rightRadius)} Q ${sx} ${railY}, ${n(sx+rightRadius)} ${railY} L ${n(maxX-rightRadius)} ${railY} Q ${maxX} ${railY}, ${maxX} ${n(railY+direction*rightRadius)}`,
      arrow:false,
      role:"rail",
    });
  }

  for(const target of targets){
    const tx=n(target.x); const ty=n(target.y);
    const isLeftEnd=tx===minX&&tx<sx;
    const isRightEnd=tx===maxX&&tx>sx;
    const startY=isLeftEnd?n(railY+direction*leftRadius):isRightEnd?n(railY+direction*rightRadius):railY;
    routes.push({path:`M ${tx} ${startY} L ${tx} ${ty}`,arrow:true,role:"drop"});
  }
  return routes;
}

export function routeGraphEdge({source,target,direction,obstacleBounds,clearance,approach=14,departure}:RouteGraphEdgeOptions){
  const sx=n(source.x); const sy=n(source.y); const tx=n(target.x); const ty=n(target.y);
  if(direction==="vertical"){
    const sign=ty>=sy?1:-1;
    // DOM layout and center alignment can differ by a few hundredths of a pixel.
    if(Math.abs(sx-tx)<GRAPH_GEOMETRY.straightTolerance)return {path:`M ${sx} ${sy} L ${sx} ${ty}`,rail:null};
    const gap=Math.abs(ty-sy);
    if(departure!==undefined){
      const horizontalSign=tx>=sx?1:-1;
      const departureDistance=Math.min(Math.max(0,departure),Math.max(0,gap-8));
      const junctionY=n(sy+sign*departureDistance);
      const radius=n(Math.min(GRAPH_GEOMETRY.cornerRadius,departureDistance/2,Math.abs(tx-sx)/2,Math.abs(ty-junctionY)/2));
      const path=[
        `M ${sx} ${sy}`,
        `L ${sx} ${n(junctionY-sign*radius)}`,
        `Q ${sx} ${junctionY}, ${n(sx+horizontalSign*radius)} ${junctionY}`,
        `L ${n(tx-horizontalSign*radius)} ${junctionY}`,
        `Q ${tx} ${junctionY}, ${tx} ${n(junctionY+sign*radius)}`,
        `L ${tx} ${ty}`,
      ].join(" ");
      return {path,rail:null};
    }
    const targetLane=Math.min(approach,gap/2);
    const terminalY=n(ty-sign*targetLane);
    const horizontalSign=tx>=sx?1:-1;
    const finalStraight=Math.min(14,targetLane*.45);
    const radius=n(Math.min(GRAPH_GEOMETRY.cornerRadius,approach*.55,Math.abs(tx-sx)/2,Math.abs(terminalY-sy),Math.max(0,targetLane-finalStraight)));
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
    const junctionY=n(sy+sign*Math.min(departure??24,Math.abs(terminalY-sy)/3));
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
