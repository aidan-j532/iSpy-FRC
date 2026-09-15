// -------- tiny dom/escape helpers shared across modules --------
export const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
export const jsStr = s => String(s ?? '').replace(/\\/g,'\\\\').replace(/'/g,"\\'");
export const $ = id => document.getElementById(id);
export function nodeEl(key) { return $('#node-' + key); }