def _addon_setting(config: dict, addon_type: str, addon_name: str,
                   key: str, default=None):
    """One setting from an add-on's config entry. Add-ons live at
    plugins.<type>.<name> as dicts; presence == enabled, so settings only
    exist for enabled add-ons."""
    try:
        entry = config["plugins"][addon_type][addon_name]
    except (KeyError, TypeError):
        return default
    if not isinstance(entry, dict):
        return default
    return entry.get(key, default)


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

    # DBSCAN / tracking settings belong to their add-ons now. Absent add-on
    # (not enabled) = no recommendation, since the add-on is not running.
    if _addon_setting(config, "trackers", "path_planner", "epsilon", 0) == 0:
        add("normal", "dbscan_eps",
            "PathPlanner DBSCAN epsilon is 0 - clustering is disabled.")

    if not config.get("camera_configs"):
        add("critical", "no_cameras", "No cameras configured - vision cannot run.")

    dist_thresh = _addon_setting(config, "trackers", "object_tracker",
                                 "distance_threshold", None)
    if dist_thresh is not None and dist_thresh > 1.5:
        add("normal", "dist_thresh",
            f"object_tracker distance_threshold is large ({dist_thresh}m) - "
            "different objects may merge.")

    if _addon_setting(config, "utilities", "network_table_handler",
                      "network_tables_ip", "") == "":
        add("critical", "nt_no_ip",
            "network_table_handler is enabled but no network_tables_ip is set.")

    return out