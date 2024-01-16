from robobo_interface import IRobobo
import numpy as np
import cv2

THRESHOLD = 100
POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']

def get_number_of_target_pixels(img):
    blue, green, red = cv2.split(img)
    mask = (blue > green) & (blue > red)
    count = np.count_nonzero(mask)
    return count


def get_reward(rob:IRobobo):
    image = rob.get_image_front()
    pixels = get_number_of_target_pixels(image)
    obs=max(rob.read_irs())
    obstacles= np.clip(obs, 0, 100) / 100
    orient = rob.read_wheels()

    reward = pixels * (1-obstacles) # * (1-abs(orient.wheel_pos_l - orient.wheel_speed_r)) # check 
    return reward


def get_observation(rob:IRobobo):
    return rob.read_irs(), rob.get_image_front()


def get_simulation_done(rob:IRobobo):
    return False


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