from robobo_interface import IRobobo
import numpy as np
import cv2
from collections import deque

POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']
LAST_FOOD_COLLECTED:int = 0
LAST_REWARD:int = 0
MAX_REWARD:int = 0
LAST_MOVES:deque= deque(maxlen=20)

def get_number_of_target_pixels(img):
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (green > blue+10) & (green > red+10)
    count = np.count_nonzero(mask)
    return (count / px_num)

def get_red(img):
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (red > blue+10) & (red > green+10)
    count = np.count_nonzero(mask)
    return (count / px_num)

def get_number_of_tgt_px_left_right(img):
    left_img, right_img = img[:, :img.shape[1]//2, :], img[:, img.shape[1]//2:, :]
    
    return get_number_of_target_pixels(left_img), get_number_of_target_pixels(right_img)

def get_number_of_tgt_px_quad(img):
    top_left = get_number_of_target_pixels(img[:img.shape[0]//2, :img.shape[1]//2, :])
    top_right = get_number_of_target_pixels(img[:img.shape[0]//2, img.shape[1]//2:, :])
    bottom_left = get_number_of_target_pixels(img[img.shape[0]//2:, :img.shape[1]//2, :])
    bottom_right = get_number_of_target_pixels(img[img.shape[0]//2:, img.shape[1]//2:, :])
    return top_left, top_right, bottom_left, bottom_right

def get_red_quad(img):
    top_left = get_red(img[:img.shape[0]//2, :img.shape[1]//2, :])
    top_right = get_red(img[:img.shape[0]//2, img.shape[1]//2:, :])
    bottom_left = get_red(img[img.shape[0]//2:, :img.shape[1]//2, :])
    bottom_right = get_red(img[img.shape[0]//2:, img.shape[1]//2:, :])
    return [top_left, top_right, bottom_left, bottom_right]

def get_4x4_grid(rob):
    img = rob.get_image_front()
    rows, cols, _ = img.shape

    square_height = rows // 4
    square_width = cols // 4

    grid = []
    for i in range(4):
        for j in range(4):
            square = img[i * square_height: (i + 1) * square_height, j * square_width: (j + 1) * square_width, :]
            grid.append(get_number_of_target_pixels(square))
    return grid

def get_reward_for_food(rob:IRobobo, action):
    global LAST_FOOD_COLLECTED
    food_collected = LAST_FOOD_COLLECTED
    if rob.nr_food_collected() > food_collected:
        LAST_FOOD_COLLECTED = LAST_FOOD_COLLECTED + 1
        # print("i'm here now, collected food:", LAST_FOOD_COLLECTED)
        return 100
    else:
        # print("got here for some reason")
        return 0

def is_stuck():
    global LAST_MOVES
    count = 0
    for i in LAST_MOVES:
        if i > 0.7: count+=1
    return True if count == len(LAST_MOVES) else False

def get_reward(rob, action, t):
    #reset values
    global LAST_REWARD, MAX_REWARD, LAST_MOVES
    reward = 0
    temp = 0
    pixels = 0
    image = rob.get_image_front()
    cv2.imwrite("/root/results/picture.jpeg", image) 

    #components of reward
    top_half = get_number_of_target_pixels(image[:image.shape[0]//2, :, :])
    bottom_half = get_number_of_target_pixels(image[image.shape[0]//2:, :, :])
    red= get_red_quad(image)
    ifr=rob.read_irs()
    f_obs= ((np.clip(max([ifr[2],ifr[3],ifr[4],ifr[5],ifr[7]]), 60, 300)-60) / 240)
    b_obs= (np.clip(max([ifr[0],ifr[1],ifr[6]]), 0, 1000) / 1000)
    obstacles = max(f_obs,b_obs)
    food = get_reward_for_food(rob, action)
    orient = rob.read_wheels()
    ori = (abs(orient.wheel_pos_l - orient.wheel_pos_r) / (100*(t+1)))
    food=rob.robot_got_food()
    LAST_MOVES.append(obstacles)
    if is_stuck(): return -100
    #reward function
    if obstacles > 0.7:
        reward -= obstacles
    elif food:
        reward += 2
        pixels= 100*top_half + 200*bottom_half
        if pixels < 0.1:
            pixels = -1
    else:
        pixels= red[0]+red[1]+ 5*red[2]+ 5*red[3]
        if pixels < 0.1:
            pixels = -1
    reward+=pixels#-t

    if reward < LAST_REWARD:
        temp = -0.1
    if reward > MAX_REWARD:
        MAX_REWARD = reward
        temp = 1

    LAST_REWARD = reward
    reward+=temp

    print(t,POSSIBLE_ACTIONS[action],"pixels:", round(pixels,3), "food:", food, "obstacles:", round(obstacles,3), "temp:",temp, "reward:", round(reward,3))
    return reward


def reset_food(rob):
    global LAST_FOOD_COLLECTED, LAST_REWARD, MAX_REWARD, LAST_MOVES
    LAST_FOOD_COLLECTED = 0
    LAST_REWARD = 0
    MAX_REWARD = 0
    LAST_MOVES = deque(maxlen=20)


def get_observation(rob:IRobobo):
    observation = rob.read_irs() + get_4x4_grid(rob) + get_red_quad(rob.get_image_front())
    return observation, rob.get_image_front()


def get_simulation_done(rob:IRobobo):
    return rob.base_got_food() or is_stuck()


def do_action(rob:IRobobo, action):
    
    if action not in POSSIBLE_ACTIONS:
        print('do_action(): action unknown:', action)
        return 0
    block = 0
    if action == 'move_forward':
        block = rob.move(50, 50, 500)
    elif action == 'turn_right':
        block = rob.move(20, -20, 250)
    elif action == 'turn_left':
        block = rob.move(-20, 20, 250)
    else:
        block = rob.move(-20, -20, 500)
    return block
    
