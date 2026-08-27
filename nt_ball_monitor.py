#!/usr/bin/env python3
"""Live NetworkTables monitor for iSpy vision output (any pipeline).

iSpy's network_table_handler runs as an NT *client* that connects to a server
at a configurable IP. With no robot around, this script becomes that server so
you can watch iSpy's output as if you were the robot's drive code.

Setup (once):
  1. Enable the "network_table_handler" addon in iSpy.
  2. Set its Robot IP to the machine running THIS script (e.g. 127.0.0.1 if
     both are on the same box, or the server's LAN IP).

Then:
  python nt_ball_monitor.py          # runs an NT server on 0.0.0.0:5810

iSpy publishes the full list of detected objects as a single JSON string topic
called VisionData/vision_data. Because every pipeline (object_detection,
april_tag, qr_code, optical_flow, depth, ...) flattens to the same schema, this
monitor works for all of them - it just prints whatever iSpy is detecting.

Driving hint: each object has x / y in metres relative to the camera (x = right
+, y = forward +). Steer to bring x -> 0, drive forward proportional to y.
"""

import json
import ntcore


def main():
    inst = ntcore.NetworkTableInstance.getDefault()
    # Act as the NT server so iSpy's client can connect to us. Signature:
    # startServer(persist_filename, listen_address, port3=1735, port4=5810)
    inst.startServer("networktables.json", "", 1735, 5810)
    print("NT server started on 0.0.0.0:5810 (iSpy Robot IP = this host)")
    print("Press Ctrl+C to stop.\n")

    table = inst.getTable("VisionData")

    vision_sub = table.getStringTopic("vision_data").subscribe(None)
    count_sub = table.getDoubleTopic("num_detections").subscribe(0.0)
    fps_sub = table.getDoubleTopic("fps").subscribe(0.0)
    lag_sub = table.getDoubleTopic("camera_lag").subscribe(0.0)

    last_line = ""
    try:
        while True:
            raw = vision_sub.get()
            n = count_sub.get()
            fps = fps_sub.get()
            lag = lag_sub.get()

            objects = []
            if raw is not None and raw:
                try:
                    objects = json.loads(raw)
                except (ValueError, TypeError):
                    objects = []

            if objects:
                # pick the closest object (smallest y = nearest, +Y is forward)
                closest = min(objects, key=lambda o: o.get("y", 0.0))
                name = closest.get("name", "?")
                steer = closest.get("x", 0.0)
                dist = closest.get("y", 0.0)
                conf = closest.get("confidence")
                conf_s = f", conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
                line = (
                    f"[{n:.0f} object(s)] closest='{name}'{conf_s} -> "
                    f"x={steer:+.3f} m, y={dist:.3f} m ahead   "
                    f"(fps={fps:.1f}, lag={lag*1000:.0f}ms)"
                )
                if line != last_line:
                    print("\033[32m" + line + "\033[0m")
                    last_line = line
            else:
                if last_line:
                    print("\033[90mno objects in view\033[0m")
                    last_line = ""
            inst.waitForListenerQueue(0.1)
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    finally:
        inst.stopServer()


if __name__ == "__main__":
    main()
