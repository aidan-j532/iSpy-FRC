// -------- layout + view persist in localStorage so a refresh keeps the same picture --------
const LAYOUT_KEY = 'ispy_node_layout_v1';
const VIEW_KEY = LAYOUT_KEY + ':view';

export function loadLayout() { try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; } catch { return {}; } }
export function saveLayout(layout) { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch { } }
export function loadView() { try { return JSON.parse(localStorage.getItem(VIEW_KEY)) || { x: 48, y: 40, zoom: 1 }; } catch { return { x: 48, y: 40, zoom: 1 }; } }
export function saveView(view) { try { localStorage.setItem(VIEW_KEY, JSON.stringify(view)); } catch { } }