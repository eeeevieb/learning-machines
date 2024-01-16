#!/usr/bin/env python3
import sys, argparse

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_task_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reallife', action='store_true', default=False, help='whether to run IRL')
    parser.add_argument('-s', '--simulation', action='store_true', default=False, help='whether to run in simulation')


    args = parser.parse_args()

    if args.reallife:
        rob = HardwareRobobo(camera=True)
    elif args.simulation:
        rob = SimulationRobobo()
        
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    run_task_1(rob)
