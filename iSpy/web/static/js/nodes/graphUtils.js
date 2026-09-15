// -------- pure graph helpers (wires/validation), no DOM --------
export function wirePath(x1, y1, x2, y2) {
  const dx = Math.max(30, Math.abs(x2 - x1) / 2);
  return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
}

// port anchors sit at node edge mid-height. anchor off the node box, not the
// translateY(-50%) port element, so pan/zoom and drags never desync the svg
export function nodeEdgePos(layout, key, edge, w, h) {
  const pos = layout[key] || { x: 0, y: 0 };
  return edge === 'in'
    ? { x: pos.x, y: pos.y + h / 2 }
    : { x: pos.x + w, y: pos.y + h / 2 };
}

export function validateGraph(g) {
  const issues = [];
  Object.keys(g.cameras).forEach(key => {
    if (!g.wires[key]) {
      issues.push({ level: 'info', key, message: 'camera is not wired to a pipeline' });
    }
  });
  Object.values(g.wires).forEach(pipeKey => {
    if (!g.pipes[pipeKey]) {
      issues.push({ level: 'error', key: pipeKey, message: 'wire points at a missing pipeline node' });
    }
  });
  Object.keys(g.pipes).forEach(key => {
    if (!g.pipelines[g.pipes[key].name]) {
      issues.push({ level: 'warn', key, message: `pipeline '${g.pipes[key].name}' is not on the server` });
    }
  });
  return issues;
}