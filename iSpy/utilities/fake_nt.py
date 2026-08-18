import argparse
import dataclasses
import math
import time

import ntcore
import wpiutil.wpistruct
from wpimath.geometry import Pose2d, Rotation2d


# mirrors NetworkHandler.py's FuelStruct so the subscription can decode what iSpy publishes
@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def make_fake_pose(t: float, radius: float = 2.0, period_s: float = 20.0) -> Pose2d:
    omega = 2 * math.pi / period_s
    x = radius * math.cos(omega * t)
    y = radius * math.sin(omega * t)
    heading = omega * t  # face the direction of travel
    return Pose2d(x, y, Rotation2d(heading))


def main():
    parser = argparse.ArgumentParser(description="Fake NT4 robot server for testing iSpy")
    parser.add_argument("--port", type=int, default=ntcore.NetworkTableInstance.kDefaultPort4,
                         help="NT4 server port (default: ntcore's standard 5810)")
    parser.add_argument("--radius", type=float, default=2.0, help="Fake driving-circle radius (m)")
    parser.add_argument("--period", type=float, default=20.0, help="Seconds per full lap")
    parser.add_argument("--static", action="store_true",
                         help="Publish a fixed pose at the origin instead of moving in a circle")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Pose publish rate")
    args = parser.parse_args()

    inst = ntcore.NetworkTableInstance.getDefault()

    # pyntcore API differs across versions - try known startServer/startServer4 signatures in order
    started = False
    for attempt in (
        lambda: inst.startServer(persist_filename="", listen_address="", port3=1735, port4=args.port),
        lambda: inst.startServer("", "", 1735, args.port),
        lambda: inst.startServer4("iSpy-fake-robot", listen_address="", port4=args.port),
        lambda: inst.startServer(),
    ):
        try:
            attempt()
            started = True
            break
        except (TypeError, AttributeError):
            continue
    if not started:
        raise RuntimeError(
            "Could not start NT server with any known startServer/startServer4 "
            "signature - check your pyntcore version's API with: "
            "python -c \"import ntcore; help(ntcore.NetworkTableInstance.startServer)\""
        )
    print(f"NT server started (requested port {args.port}; falls back to pyntcore default if unsupported by this version)")
    print("Point iSpy's network_tables_ip at 127.0.0.1 (or this machine's IP if iSpy runs elsewhere).")

    # --- Publisher: robot pose, same topic NetworkTableHandler.get_robot_pose() reads ---
    odom_table = inst.getTable("AdvantageKit/RealOutputs/Odometry")
    pose_pub = odom_table.getStructTopic("Robot", Pose2d).publish()

    # --- Subscribers: everything NetworkTableHandler.update() publishes ---
    vision_table = inst.getTable("VisionData")
    fuel_sub = vision_table.getStructArrayTopic("vision_data", FuelStruct).subscribe([])
    fps_sub = vision_table.getDoubleTopic("fps").subscribe(0.0)
    det_sub = vision_table.getDoubleTopic("num_detections").subscribe(0.0)
    lag_sub = vision_table.getDoubleTopic("camera_lag").subscribe(0.0)
    hopper_sub = vision_table.getBooleanTopic("hopper_sees_object").subscribe(False)

    print("Waiting for iSpy to connect and start publishing VisionData...\n")

    start = time.perf_counter()
    interval = 1.0 / args.rate_hz
    last_print = 0.0
    last_fuel_count = -1

    try:
        while True:
            t = time.perf_counter() - start

            pose = Pose2d() if args.static else make_fake_pose(t, args.radius, args.period)
            pose_pub.set(pose)

            # print vision data on change, else once/sec so you know its still alive
            fuels = fuel_sub.get()
            now = time.time()
            if len(fuels) != last_fuel_count or now - last_print > 1.0:
                last_fuel_count = len(fuels)
                last_print = now
                print(
                    f"[t={t:6.1f}s] robot=({pose.X():+.2f}, {pose.Y():+.2f}, "
                    f"{math.degrees(pose.rotation().radians()):+6.1f}deg)  "
                    f"fps={fps_sub.get():.1f}  detections={int(det_sub.get())}  "
                    f"lag={lag_sub.get()*1000:.0f}ms  hopper={hopper_sub.get()}"
                )
                for f in fuels:
                    print(f"    fuel: x={f.x:+.2f} y={f.y:+.2f} z={f.z:+.2f} "
                          f"yaw={math.degrees(f.yaw):+.1f}deg")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping fake robot server.")


if __name__ == "__main__":
    main()