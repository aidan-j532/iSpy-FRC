// -------- canvas transform: zoom, pan, marquee. expects ctx.renderWires/applyView/paintSelection --------
import { $, nodeEl } from './dom.js';
import { saveView } from './storage.js';

export function applyView(ctx) {
  $('#graph-layer').style.transform = `translate(${ctx.view.x}px, ${ctx.view.y}px) scale(${ctx.view.zoom})`;
  $('#node-canvas').style.backgroundSize = `${Math.max(8, 22 * ctx.view.zoom)}px ${Math.max(8, 22 * ctx.view.zoom)}px`;
  $('#zoom-label').textContent = Math.round(ctx.view.zoom * 100) + '%';
}

export function screenToWorld(ctx, cx, cy) {
  const r = $('#node-canvas').getBoundingClientRect();
  return { x: (cx - r.left - ctx.view.x) / ctx.view.zoom, y: (cy - r.top - ctx.view.y) / ctx.view.zoom };
}

// zoom keeping the point under the cursor fixed
export function zoomAt(ctx, factor, cx, cy) {
  const r = $('#node-canvas').getBoundingClientRect();
  const wx = (cx - r.left - ctx.view.x) / ctx.view.zoom;
  const wy = (cy - r.top - ctx.view.y) / ctx.view.zoom;
  ctx.view.zoom = Math.min(2.5, Math.max(0.2, ctx.view.zoom * factor));
  ctx.view.x = cx - r.left - wx * ctx.view.zoom;
  ctx.view.y = cy - r.top - wy * ctx.view.zoom;
  saveView(ctx.view);
  ctx.applyView();
  ctx.renderWires();
}

export function zoomAroundCenter(ctx, factor) {
  const r = $('#node-canvas').getBoundingClientRect();
  zoomAt(ctx, factor, r.left + r.width / 2, r.top + r.height / 2);
}

export function setZoom(ctx, z) { zoomAroundCenter(ctx, z / ctx.view.zoom); }

export function fitToScreen(ctx) {
  const keys = Object.keys(ctx.graph.layout);
  if (!keys.length) { setZoom(ctx, 1); return; }
  const r = $('#node-canvas').getBoundingClientRect();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  keys.forEach(k => {
    const p = ctx.graph.layout[k];
    const el = nodeEl(k);
    const w = el ? el.offsetWidth : 170, h = el ? el.offsetHeight : 70;
    if (p) {
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + w); maxY = Math.max(maxY, p.y + h);
    }
  });
  if (!isFinite(minX)) { setZoom(ctx, 1); return; }
  const pad = 60;
  const zoom = Math.min(1.25, Math.max(0.2, Math.min((r.width - pad) / (maxX - minX + pad), (r.height - pad) / (maxY - minY + pad))));
  ctx.view.zoom = zoom;
  ctx.view.x = pad / 2 - minX * zoom + (r.width / 2 - ((minX + maxX) / 2) * zoom);
  ctx.view.y = pad / 2 - minY * zoom + (r.height / 2 - ((minY + maxY) / 2) * zoom);
  saveView(ctx.view);
  ctx.applyView();
  ctx.renderWires();
}

// -------- canvas pan + marquee selection --------
export function initPan(ctx) {
  const wrap = $('#node-canvas');
  let panning = null, marquee = null, moved = false, startClient = null;

  wrap.addEventListener('mousedown', ev => {
    if (ev.target.closest('.node') || ev.target.closest('.port') || ev.target.closest('.canvas-controls')) return;
    if (ev.button !== 0 && ev.button !== 1) return;
    ev.preventDefault();
    startClient = { x: ev.clientX, y: ev.clientY };
    moved = false;
    if (ev.button === 1 || ev.shiftKey) {
      // middle mouse (or shift-drag) pans the viewport
      panning = { sx: ev.clientX, sy: ev.clientY, x: ctx.view.x, y: ctx.view.y };
      wrap.classList.add('panning');
    } else {
      // left-drag on empty canvas = marquee select
      const w = screenToWorld(ctx, ev.clientX, ev.clientY);
      marquee = { x: w.x, y: w.y };
      const m = $('#marquee');
      m.style.display = 'block';
      m.style.left = w.x + 'px'; m.style.top = w.y + 'px';
      m.style.width = '0px'; m.style.height = '0px';
    }
  });

  const move = ev => {
    if (!panning && !marquee) return;
    if (Math.abs(ev.clientX - startClient.x) + Math.abs(ev.clientY - startClient.y) > 3) moved = true;
    if (panning) {
      ctx.view.x = panning.x + (ev.clientX - panning.sx);
      ctx.view.y = panning.y + (ev.clientY - panning.sy);
      saveView(ctx.view);
      ctx.applyView();
      ctx.renderWires();
    } else if (marquee) {
      const w = screenToWorld(ctx, ev.clientX, ev.clientY);
      const m = $('#marquee');
      const x = Math.min(marquee.x, w.x), y = Math.min(marquee.y, w.y);
      m.style.left = x + 'px'; m.style.top = y + 'px';
      m.style.width = (Math.max(marquee.x, w.x) - x) + 'px';
      m.style.height = (Math.max(marquee.y, w.y) - y) + 'px';
    }
  };

  const up = ev => {
    if (!panning && !marquee) return;
    if (marquee) {
      const w = screenToWorld(ctx, ev.clientX, ev.clientY);
      const a = { x: Math.min(marquee.x, w.x), y: Math.min(marquee.y, w.y) };
      const b = { x: Math.max(marquee.x, w.x), y: Math.max(marquee.y, w.y) };
      if (moved && (b.x - a.x > 4 || b.y - a.y > 4)) {
        if (!ev.shiftKey) ctx.selected.clear();
        document.querySelectorAll('#graph-layer .node').forEach(el => {
          const k = el.dataset.key, p = ctx.graph.layout[k];
          if (!p) return;
          const wd = el.offsetWidth, h = el.offsetHeight;
          if (p.x < b.x && p.x + wd > a.x && p.y < b.y && p.y + h > a.y) ctx.selected.add(k);
        });
        ctx.paintSelection();
      } else if (!ev.shiftKey) {
        ctx.selected.clear();
        ctx.paintSelection();
      }
      $('#marquee').style.display = 'none';
    }
    panning = null; marquee = null;
    wrap.classList.remove('panning');
  };

  wrap.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
}

export function initWheel(ctx) {
  $('#node-canvas').addEventListener('wheel', ev => {
    ev.preventDefault();
    zoomAt(ctx, ev.deltaY < 0 ? 1.1 : 0.9, ev.clientX, ev.clientY);
  }, { passive: false });
}