#!/usr/bin/env python3
"""Live NetworkTables ball-position monitor for the default v26 Fuel model.

iSpy's network_table_handler runs as an NT *client* that connects to a server
at a configurable IP (default 10.0.0.2). With no robot around, this script
becomes that server on 127.0.0.1, so you can watch iSpy's output as if you
were the robot's drive code.

Setup (once):
  1. In the web UI -> Addons, enable "network_table_handler" and set
     Robot IP = 127.0.0.1
  2. Make sure your camera uses the object_detection pipeline with the
     _default_v26_detect_for_fuel model (detects class "Fuel").

Then:
  python nt_ball_monitor.py          # runs an NT server on 127.0.0.1

This subscribes to the same topics the robot would read:
  VisionData/vision_data     -> FuelStruct array: x (right+), y (forward+), z (up)
  VisionData/num_detections  -> how many balls seen
  VisionData/fps / camera_lag

Ball-driving hint: steer to make x -> 0, and use y (distance ahead, metres)
to decide how far to drive forward. +Y is straight ahead of the camera.
"""

import dataclasses
import time
import ntcore
import wpiutil.wpistruct


@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def main():
    inst = ntcore.NetworkTableInstance.getDefault()
    # Act as the NT server so iSpy's client can connect to us. Signature:
    # startServer(persist_filename, listen_address, port3=1735, port4=5810)
    # Defaults already use 5810 for the main port, but pass it explicitly so
    # it stays correct regardless of the ntcore default.
    inst.startServer("networktables.json", "", 1735, 5810)
    print("NT server started on 127.0.0.1:5810 (robot IP for iSpy = 127.0.0.1)")
    print("Press Ctrl+C to stop.\n")

    table = inst.getTable("VisionData")

    ball_sub = table.getStructArrayTopic("vision_data", FuelStruct).subscribe([])
    count_sub = table.getDoubleTopic("num_detections").subscribe(0.0)
    fps_sub = table.getDoubleTopic("fps").subscribe(0.0)
    lag_sub = table.getDoubleTopic("camera_lag").subscribe(0.0)

    last_msg = ""
    try:
        while True:
            balls = ball_sub.get()
            n = count_sub.get()
            fps = fps_sub.get()
            lag = lag_sub.get()

            if balls:
                # pick the closest ball (smallest y = nearest, since +Y is forward)
                closest = min(balls, key=lambda b: b.y)
                steer = closest.x  # metres right(+) / left(-)
                dist = closest.y   # metres ahead
                msg = (
                    f"[{n:.0f} ball(s)] closest -> x={steer:+.3f} m, "
                    f"y={dist:.3f} m ahead   (fps={fps:.1f}, lag={lag*1000:.0f}ms)"
                )
                if msg != last_msg:
                    print("\033[32m" + msg + "\033[0m")
                    last_msg = msg
            else:
                if last_msg:
                    print("\033[90mno ball in view\033[0m")
                    last_msg = ""
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    finally:
        inst.stopServer()


if __name__ == "__main__":
    main()
