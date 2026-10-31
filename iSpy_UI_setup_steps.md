## iSpy Web UI Setup Steps for JSON-Based ObjectVision

Follow these steps in order. All URLs assume the iSpy instance is at `http://<board-ip>:5000`.

### 1. Enable the NetworkTables utility add-on
1. Open `http://<board-ip>:5000` in a browser.
2. Click **Add-ons** in the top navigation bar (the page says "Trackers, utilities, and frame processors that extend vision. Changes take effect after the vision service restarts (Dashboard → Restart).").
3. Find **`FRC/network_table_handler`** in the utilities group (it may be listed under a "Utilities" or "Built-in" section).
4. Toggle the **enable switch** to turn it **on**.
   - *Note*: If the add-on is already on, skip this step. If it was off, the publish list will now include the default entries.

### 2. Verify/fix the `vision_data` publish row
While on the **Add-ons** page, locate the publish list (the utility shows a table of publish entries). You should see a row named **`vision_data`** with these fields:

| Field         | Should be set to |
|---------------|------------------|
| **name**      | `vision_data`    |
| **data_type** | `json`           |
| **source**    | `detections`     |
| **nt_topic**  | `vision_data`    |

- If `data_type` is currently `struct[]`, change it to `json` (or remove the `struct[]` row and keep only the `json` one).
- The default `DEFAULT_PUBLISH` from iSpy's `NetworkHandler.py` already has this set correctly as JSON, so you may not need to change anything — but double-check there is no duplicate `struct[]` entry for `vision_data`.

### 3. Set Units to `frc`
1. Click **Settings** in the top navigation bar.
2. Go to the **Units** tab (default tab on load).
3. Verify the **Unit** selector is set to **`frc`** (FRC — meters).
   - Options: `frc` (meters), `meter`, `foot`, `inch`, `centimeter`.
   - The config.json in this checkout already has `"unit": "frc"`, but verify it on the UI.
4. If you changed it, click **Save Settings** at the bottom — the UI will warn: "Unit changed - you must update how your code processes measurements".
5. Note: Changes take effect after the vision restart.

### 4. Restart the vision service
1. Click **Dashboard** in the top navigation bar.
2. Click the **Restart** button (or call the service command `restart` via the endpoint `http://<board-ip>:5050/service/restart`).
3. The UI will show "iSpy is restarting" and reconnect automatically.
4. Wait until the vision status shows as running and the detection count updates.

### 5. Verify in Glass/OutlineViewer
After the restart:
- **Topic**: `VisionData/vision_data`
- **Type**: Should be a **string** (not `struct:Object[]`).
- **Value**: Should be a JSON array starting with `[`, e.g.:
  ```json
  [{"name":"fuel","x":1.23,"y":-0.45,"z":0.0,"roll":0.0,"pitch":0.0,"yaw":0.2,"confidence":0.96,"id":1,"depth_source":"monocular"}]
  ```
- **FPS** and **num_detections** values should be updating while iSpy is running.
- If you stop iSpy, the `visionTimestamp` in your Java code should stop changing, and the data should go stale after 500 ms (this is handled by the timestamp-based staleness check you added to `ObjectVision.periodic()`).

### 6. (Optional) Confirm no duplicate `struct[]` entry
If you see **two** `vision_data` rows — one with `data_type: "json"` and one with `data_type: "struct[]"` — remove or disable the `struct[]` row. Only the `json` row should publish to `vision_data`. The Java code you're switching to reads a string topic, not a struct array.

### Summary Checklist
- [ ] `FRC/network_table_handler` add-on is **enabled**
- [ ] `vision_data` publish row has `data_type: json`, `source: detections`, `nt_topic: vision_data`
- [ ] **Settings → Unit** is `frc`
- [ ] Vision restarted via **Dashboard → Restart**
- [ ] `VisionData/vision_data` in Glass shows a **string JSON array**
- [ ] `fps`/`num_detections` update in the iSpy Dashboard