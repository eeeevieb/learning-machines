from robobo_interface import IRobobo
import numpy as np
import cv2

THRESHOLD = 100
POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']

def get_number_of_target_pixels(img):
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (blue > green) & (blue > red)
    count = np.count_nonzero(mask)
    return (count / px_num) + 0.001


def get_reward_for_food(rob:IRobobo, action):
    return rob.nr_food_collected()*100

def get_reward(rob:IRobobo, t, action):
    image = rob.get_image_front()
    pixels = get_number_of_target_pixels(image)

    obstacles = (np.clip(max(rob.read_irs()), 0, 1000) / 1000) - 0.001

    orient = rob.read_wheels()
    ori = (abs(orient.wheel_pos_l - orient.wheel_pos_r) / (50*t +1))

    turns = (0.9 if action == 0 or action == 3  else 0)
    
    #reward = pixels * (1-obstacles) * (1 - ori) # check 
    reward = pixels * (1-obstacles) * (1 - turns)
    #print(f"pixels: {pixels}, obs: {obstacles}, orient: {ori}, reward: {reward}")
    
    return reward


def get_observation(rob:IRobobo):
    # return [*rob.read_irs(), get_number_of_target_pixels(rob.get_image_front)], rob.get_image_front()
    return rob.read_irs(), rob.get_image_front()


def get_simulation_done(rob:IRobobo):
    image = rob.get_image_front()
    pixels = get_number_of_target_pixels(image)

    return pixels == 1.001
    # return any(np.array(rob.read_irs()) > 150)


def do_action(rob:IRobobo, action):
    
    if action not in POSSIBLE_ACTIONS:
        print('do_action(): action unknown:', action)
        return 0
    block = 0
    if action == 'move_forward':
        block = rob.move(100, 100, 100)
    elif action == 'turn_right':
        block = rob.move(50, -50, 50)
    elif action == 'turn_left':
        block = rob.move(-50, 50, 50)
    else:
        block = rob.move(-50, -50, 100)
    return block
    
