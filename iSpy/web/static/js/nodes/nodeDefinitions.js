// -------- static node definitions, no DOM --------
export const DEFAULT_PIPELINE = 'object_detection';

export const TEMPLATES = [
  { pipe: 'object_detection', label: 'video + object detection' },
  { pipe: 'april_tag', label: 'video + april tag' },
];

export function pipeNodeId(name, count) { return 'pipe:' + name + ':' + count; }