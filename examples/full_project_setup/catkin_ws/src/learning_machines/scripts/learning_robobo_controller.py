#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions


if __name__ == "__main__":
<<<<<<< HEAD
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
=======
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reallife', action='store_true', default=False, help='whether to run IRL')
    parser.add_argument('-s', '--simulation', action='store_true', default=False, help='whether to run in simulation')


    args = parser.parse_args()

    if args.reallife:
        rob = HardwareRobobo(camera=False)
    elif args.simulation:
        rob = SimulationRobobo()
        pass
>>>>>>> 9ea6f29dbf826b5535fd29cfc3b13d48a017d381
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    run_all_actions(rob)
