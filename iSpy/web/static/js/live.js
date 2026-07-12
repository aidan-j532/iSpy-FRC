// Reusable "keep this data fresh" helper. Plain polling (not SSE) to start -
// simplest thing that works with Flask's dev server and threaded=True, and
// every page that needs live numbers (status, camera fps, conversion
// progress) can use the exact same pattern instead of each rolling its own
// setInterval loop like HealthReporter's old inline HTML did.
//
// Usage:
//   const stop = ISPY_LIVE.poll('/api/status', 500, (data) => {
//       document.getElementById('fps').textContent = data.fps;
//   });
//   // stop() to cancel when navigating away / tearing down a component

const ISPY_LIVE = (() => {
    function poll(path, intervalMs, onData, onError) {
        let stopped = false;

        async function tick() {
            if (stopped) return;
            try {
                const data = await ISPY.apiGet(path);
                onData(data);
            } catch (e) {
                if (onError) onError(e);
                else console.warn(`live poll failed for ${path}:`, e);
            } finally {
                if (!stopped) setTimeout(tick, intervalMs);
            }
        }

        tick();
        return () => { stopped = true; };
    }

    return { poll };
})();