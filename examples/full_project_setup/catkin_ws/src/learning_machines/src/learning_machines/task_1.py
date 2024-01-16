from robobo_interface import SimulationRobobo
from .rl_policygrad import PolicyGradientModel
import numpy as np
from .irobobo_extensions import get_observation, get_reward, do_action


def do_stuff(rob):
    model = PolicyGradientModel(rob, 16)
    model.train(num_episodes=1000, max_t=100, gamma=0.99)

    reward = 0
    max_iter = 1000
    iter = 0
    while iter < max_iter:
        observation = get_observation(rob)

        action = model.predict(observation)
        do_action(rob, action)
        reward += get_reward(rob)
        iter += 1




def run_task_1(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print(" ")
    
    do_stuff(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

