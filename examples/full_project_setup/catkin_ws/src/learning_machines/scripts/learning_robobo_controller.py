#!/usr/bin/env python3
import argparse

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_task_0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-h', '--hardware', action='store_true', default=False, help='whether to run IRL')
    parser.add_argument('-s', '--simulator', action='store_true', default=False, help='whether to run in simulation')


    args = parser.parse_args()

    if args['hardware']:
        rob = HardwareRobobo(camera=False)
    elif args['simulator']:
        rob = SimulationRobobo(camera=False)
        pass
    else:
        raise ValueError('should provide a flag to run on robot or simulator')
    run_task_0(rob)
