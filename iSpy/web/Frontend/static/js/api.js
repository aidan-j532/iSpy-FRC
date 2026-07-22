// Shared fetch helpers. Every page should use these instead of raw fetch()
// so error handling / JSON parsing / headers stay identical everywhere.
// Usage:
//   const data = await apiGet('/api/status');
//   const result = await apiPost('/api/cameras/0/settings', {fps_cap: 30});

const ISPY = (() => {
    async function apiGet(path) {
        const res = await fetch(path, { headers: { "Accept": "application/json" } });
        if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
        return res.json();
    }

    async function apiPost(path, body) {
        const res = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
        });
        if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
        return res.json();
    }

    async function apiDelete(path) {
        const res = await fetch(path, { method: "DELETE" });
        if (!res.ok) throw new Error(`DELETE ${path} -> ${res.status}`);
        return res.json();
    }

    // Small helper for status pills: statusPillHTML("ok", "Camera 1") ->
    // <span class="ispy-pill ok">Camera 1</span>
    function statusPillHTML(level, label) {
        const safe = String(label).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
        return `<span class="ispy-pill ${level}">${safe}</span>`;
    }

    return { apiGet, apiPost, apiDelete, statusPillHTML };
})();