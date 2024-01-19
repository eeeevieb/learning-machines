from robobo_interface import SimulationRobobo
from .RL_policygrad import PolicyGradientModel
import numpy as np
from .irobobo_extensions import get_observation, get_reward, do_action, POSSIBLE_ACTIONS
import torch


def do_stuff(rob):
    model = PolicyGradientModel(rob, 16)
    print('INFO setup model')

    print('INFO started training model')
    model.train(num_episodes=100, max_t=100, gamma=0.99)
    model.save_model('polgrad.pth')

    reward = 0
    max_iter = 10
    iter = 0
    while iter < max_iter:
        observation = get_observation(rob)

        action, i = model.predict(observation[0])
        
        do_action(rob, action)
        reward += get_reward(rob,iter,action)
        iter += 1




def run_task_1(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print(" ")

    rob.set_phone_tilt(90, 50)
    
    do_stuff(rob)
    # model = PolicyGradientModel(rob, 16)
    # # model.policy.load_state_dict(torch.load("examples/full_project_setup/results/intermediate_cp3.pth"))
    # model.policy.load_state_dict(torch.load("./results/intermediate_cp3.pth"))
    # reward = 0
    # max_iter = 100
    # iter = 0
    # while iter < max_iter:
    #     observation = get_observation(rob)

    #     action, i = model.predict(observation[0])
        
    #     do_action(rob, action)
    #     reward += get_reward(rob,iter,action)
    #     iter += 1
    # print(reward)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

