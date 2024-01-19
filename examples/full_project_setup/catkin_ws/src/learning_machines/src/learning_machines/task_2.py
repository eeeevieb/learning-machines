from robobo_interface import SimulationRobobo, IRobobo
from .RL_policygrad import PolicyGradientModel
import numpy as np
from .irobobo_extensions import get_observation, get_reward, do_action, POSSIBLE_ACTIONS
import torch


def train(rob:IRobobo):
    model = PolicyGradientModel(rob, 16)
    model.train(100, max_t=100, gamma=0.7, print_every=10)
    model.save_model('100_epochs.pth')


def run(rob:IRobobo):
    model = PolicyGradientModel(rob, 16)
    model.policy.load_state_dict(torch.load('./results/CHECKPOINT_NAME.pth'))
    print('INFO loaded model from checkpoint')

    reward = 0
    max_iter = 100
    iter = 0
    while iter < max_iter:
        observation = get_observation(rob)

        action, prob = model.predict(observation[0])
        print(f'INFO action: {action}, probability: {prob}')
        do_action(rob, action)
        reward += get_reward(rob,iter,action)
        iter += 1
        if iter % 50 == 0:
            print('reward:', reward)

def run_task_2(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('INFO started simulation')

    rob.set_phone_tilt(90, 50)
    
    train(rob)
    # run(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
