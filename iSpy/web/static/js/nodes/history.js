// -------- undo/redo for positions + pipeline node presence. wires are not
// snapshotted - they re-derive from camera config on restore. no DOM. --------
export function createHistory() {
  const stack = [];
  let idx = -1;

  function snapshot(g) {
    return {
      layout: JSON.parse(JSON.stringify(g.layout)),
      pipes: JSON.parse(JSON.stringify(g.pipes)),
    };
  }

  return {
    push(g) {
      stack.splice(idx + 1);
      stack.push(snapshot(g));
      if (stack.length > 60) stack.shift();
      idx = stack.length - 1;
    },
    canUndo() { return idx > 0; },
    canRedo() { return idx < stack.length - 1; },
    undo() { if (!this.canUndo()) return null; idx--; return stack[idx]; },
    redo() { if (!this.canRedo()) return null; idx++; return stack[idx]; },
    get length() { return stack.length; },
    get index() { return idx; },
  };
}