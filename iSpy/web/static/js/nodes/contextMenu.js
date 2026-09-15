// -------- right-click menus (context menu on nodes / empty canvas) --------
import { $, jsStr } from './dom.js';

export function showMenu(x, y, items) {
  const menu = $('#ctx-menu');
  menu.innerHTML = items;
  menu.style.display = 'block';
  menu.style.left = x + 'px';
  menu.style.top = y + 'px';
  const r = menu.getBoundingClientRect();
  if (r.right > window.innerWidth - 8) menu.style.left = Math.max(8, x - r.width) + 'px';
  if (r.bottom > window.innerHeight - 8) menu.style.top = Math.max(8, y - r.height) + 'px';
}

export function init(ctx) {
  const wrap = $('#node-canvas');
  wrap.addEventListener('contextmenu', ev => {
    ev.preventDefault();
    const node = ev.target.closest('.node');
    if (node) {
      const key = node.dataset.key;
      ctx.selected.clear(); ctx.selected.add(key); ctx.paintSelection();
      let items = '';
      if (node.dataset.kind === 'pipeline') {
        items += `<button onclick="editNode('${jsStr(key)}')">Settings</button>`;
        items += `<button onclick="duplicateNode('${jsStr(key)}')">Duplicate</button>`;
        items += `<button class="danger" onclick="removePipelineNode('${jsStr(key)}')">Remove node</button>`;
      } else {
        items += `<button onclick="editNode('${jsStr(key)}')">Settings</button>`;
        items += `<button class="danger" onclick="deleteCameraNode('${jsStr(key)}')">Delete camera&hellip;</button>`;
      }
      showMenu(ev.clientX, ev.clientY, items);
    } else {
      showMenu(ev.clientX, ev.clientY,
        '<button onclick="addPipelineNodeFromPicker()">Add pipeline&hellip;</button>' +
        '<button onclick="addNewCamera()">Add camera&hellip;</button>' +
        '<div class="sep"></div>' +
        '<button onclick="fitToScreen()">Fit to view</button>' +
        '<button onclick="setZoom(1)">Zoom 100%</button>' +
        '<button onclick="resetLayout()">Reset layout</button>');
    }
  });
  document.addEventListener('click', () => { $('#ctx-menu').style.display = 'none'; });
  document.addEventListener('mousedown', ev => { if (ev.button === 0) $('#ctx-menu').style.display = 'none'; });
}