from robobo_interface import SimulationRobobo, IRobobo
import numpy as np

THRESHOLD = 100
POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']

def get_number_of_target_pixels(img):
    return 1


def get_reward(rob:IRobobo):
    image = rob.get_image_front()
    return get_number_of_target_pixels(image)


def get_observation(rob:IRobobo):
    return rob.read_irs()


def do_action(rob:IRobobo, action):
    
    if action not in POSSIBLE_ACTIONS:
        print('do_action(): action unknown:', action)
        return
    
    if action == 'move_forward':
        rob.move(50, 50, 100)
    elif action == 'turn_right':
        rob.move(50, 0, 50)
    elif action == 'turn_left':
        rob.move(0, 50, 50)
    else:
        rob.move(-20, -20, 100)
        

def do_stuff(rob):
    reward = get_reward(rob)
    while reward < THRESHOLD:
        observation = get_observation(rob)
        action = model.pick_action(observation)
        do_action(rob, action)




def run_task_0(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print(" ")
    
    do_stuff(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

