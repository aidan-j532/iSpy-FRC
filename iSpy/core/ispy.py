from iSpy.boot.boot import on_boot as boot_run
from iSpy.core.game_loop import main as game_loop_main
import argparse

# So bascially this is the new command, where you just run it and it boots and starts vision. It shuold by defualt not wait for pipelines to be ready

def main():
    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser.add_argument("-s", "--service", action="store_true",
                         help="Install and start the watchdog service")
    parser.add_argument("-f", "--fresh", action="store_true",
                         help="Wipe all iSpy generated stuff, "
                              "resets to default setup")
    parser.add_argument("-w", "--wait", action="store_true",
                        help="Wait for all pipelines to be ready before running vision")
    args = parser.parse_args()
    boot_run(install_service=args.service, fresh=args.fresh, wait=args.wait)

    game_loop_main()

if __name__ == "__main__":
    main()