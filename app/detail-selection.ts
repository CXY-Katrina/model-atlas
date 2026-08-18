export type DetailState<T> = { hovered: T | null; pinned: T | null };

export type DetailEvent<T> =
  | { type: "hover"; node: T }
  | { type: "pin"; node: T }
  | { type: "leave" }
  | { type: "clear" };

export function nextDetailState<T>(state: DetailState<T>, event: DetailEvent<T>): DetailState<T> {
  if (event.type === "hover") return { ...state, hovered: event.node };
  if (event.type === "pin") return { hovered: null, pinned: event.node };
  if (event.type === "leave") return { ...state, hovered: null };
  return { hovered: null, pinned: null };
}
