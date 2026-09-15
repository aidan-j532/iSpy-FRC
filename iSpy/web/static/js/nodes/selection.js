// -------- selection + drag + keyboard shortcuts. uses ctx actions for cross-module moves. --------
import { nodeEl } from './dom.js';
import { saveLayout } from './storage.js';

export function paintSelection(ctx) {
  document.querySelectorAll('.node').forEach(n => n.classList.toggle('selected', ctx.selected.has(n.dataset.key)));
}

export function makeDraggable(ctx, el, oid) {
  let dragging = false, moved = false, startMouse = null;
  el.addEventListener('mousedown', ev => {
    if (ev.button !== 0) return;
    // don't drag when the click started on a button or port (those do their own thing)
    if (ev.target.closest('.node-btn') || ev.target.closest('.port')) return;
    ev.preventDefault();
    if (!ctx.selected.has(oid)) {
      if (ev.ctrlKey || ev.metaKey) ctx.selected.add(oid);
      else { ctx.selected.clear(); ctx.selected.add(oid); }
    }
    paintSelection(ctx);
    dragging = true; moved = false;
    startMouse = { x: ev.clientX, y: ev.clientY };
  });
  window.addEventListener('mousemove', ev => {
    if (!dragging) return;
    const dx = (ev.clientX - startMouse.x) / ctx.view.zoom;
    const dy = (ev.clientY - startMouse.y) / ctx.view.zoom;
    if (dx || dy) moved = true;
    ctx.selected.forEach(k => {
      const pos = ctx.graph.layout[k];
      if (!pos) return;
      pos.x += dx; pos.y += dy;
      const n = nodeEl(k);
      if (n) { n.style.left = pos.x + 'px'; n.style.top = pos.y + 'px'; }
    });
    startMouse = { x: ev.clientX, y: ev.clientY };
    ctx.renderWires();
  });
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    if (moved) { ctx.pushUndo(); saveLayout(ctx.graph.layout); }
  };
  window.addEventListener('mouseup', stop);
  el.addEventListener('dblclick', ev => {
    if (ev.target.closest('.port') || ev.target.closest('.node-btn')) return;
    ctx.editNode(oid);
  });
}

export function duplicateNode(ctx, key) {
  if (!ctx.graph.pipes[key]) return;
  const pos = ctx.graph.layout[key] || { x: 60, y: 60 };
  ctx.pushUndo();
  ctx.addPipelineNode(ctx.graph.pipes[key].name, { x: pos.x + 36, y: pos.y + 36 });
}

export function duplicateSelection(ctx) {
  if (!ctx.selected.size) return;
  ctx.pushUndo();
  let duped = false;
  ctx.selected.forEach(key => { if (ctx.graph.pipes[key]) { duplicateNode(ctx, key); duped = true; } });
  if (!duped) showToast('only pipeline nodes can be duplicated here', 'error');
}

export function deleteSelection(ctx) {
  if (!ctx.selected.size) return;
  ctx.pushUndo();
  [...ctx.selected].forEach(key => {
    if (ctx.graph.cameras[key]) ctx.deleteCameraNode(key);
    else if (ctx.graph.pipes[key]) ctx.removePipelineNode(key);
  });
  saveLayout(ctx.graph.layout);
}

export function initKeyboard(ctx) {
  document.addEventListener('keydown', ev => {
    const tag = (ev.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (ev.ctrlKey || ev.metaKey) {
      const k = ev.key.toLowerCase();
      if (k === 'z') { ev.preventDefault(); const s = ev.shiftKey ? ctx.history.redo() : ctx.history.undo(); if (s) ctx.restore(s); return; }
      if (k === 'y') { ev.preventDefault(); const s = ctx.history.redo(); if (s) ctx.restore(s); return; }
      if (k === 'd') { ev.preventDefault(); duplicateSelection(ctx); return; }
      return;
    }
    if (ev.key === 'Delete' || ev.key === 'Backspace') { ev.preventDefault(); deleteSelection(ctx); }
    else if (ev.key === 'Escape') { ctx.selected.clear(); paintSelection(ctx); }
    else if (ev.key === '+' || ev.key === '=') { ctx.zoomAroundCenter(1.2); }
    else if (ev.key === '-') { ctx.zoomAroundCenter(0.83); }
  });
}