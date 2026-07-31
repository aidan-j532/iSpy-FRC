def get_structured_recommendations(config: dict) -> list[dict]:
    out = []
    def add(severity, key, message):
        out.append({"severity": severity, "key": key, "message": message})

    calib_issues = False
    for cam_name, cam_cfg in config.get("camera_configs", {}).items():
        calib = cam_cfg.get("calibration", {})
        if calib.get("size", 0) == 0 and calib.get("distance", 0) == 0:
            add("critical", f"calib.{cam_name}",
                f"Camera '{cam_name}' is uncalibrated (size/distance are 0) - distance estimates will be wrong.")
            calib_issues = True
        if cam_cfg.get("height", 0) == 0:
            add("normal", f"height.{cam_name}", f"Camera '{cam_name}' height is 0 - verify this is intentional.")
        if not cam_cfg.get("device_id"):
            add("normal", f"deviceid.{cam_name}",
                f"Camera '{cam_name}' has no saved device_id - it may not survive a USB replug/reindex.")

    dbscan = config.get("dbscan", {})
    if dbscan.get("epsilon", 0) == 0:
        add("normal", "dbscan_eps", "DBSCAN epsilon is 0 - clustering is disabled.")

    if not config.get("camera_configs"):
        add("critical", "no_cameras", "No cameras configured - vision cannot run.")

    dist_thresh = config.get("distance_threshold", -1)
    if dist_thresh is not None and dist_thresh > 1.5:
        add("normal", "dist_thresh", f"distance_threshold is large ({dist_thresh}m) - different objects may merge.")

    if config.get("use_network_tables") and not config.get("network_tables_ip"):
        add("critical", "nt_no_ip", "NetworkTables is enabled but no IP is set.")

    return out