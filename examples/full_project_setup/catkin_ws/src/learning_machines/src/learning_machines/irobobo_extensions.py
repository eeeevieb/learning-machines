from robobo_interface import IRobobo
import numpy as np
import cv2

POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']
LAST_FOOD_COLLECTED:int = 0

def get_number_of_target_pixels(img):
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (green > blue+10) & (green > red+10)
    count = np.count_nonzero(mask)
    return (count / px_num)


def get_number_of_tgt_px_left_right(img):
    left_img, right_img = img[:, :img.shape[1]//2, :], img[:, img.shape[1]//2:, :]
    return get_number_of_target_pixels(left_img), get_number_of_target_pixels(right_img)


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


def get_reward(rob, action):
    image = rob.get_image_front()
    cv2.imwrite("/root/results/picture.jpeg", image) 

    pixels = get_number_of_target_pixels(image)
    obstacles = (np.clip(max(rob.read_irs()), 0, 1000) / 2000)
    food = get_reward_for_food(rob, action)

    reward = food if food > 0 else (pixels - obstacles)

    # print("last food collected:", LAST_FOOD_COLLECTED, "pixels:", pixels, "food:", food, "reward:", reward)

    return reward

def reset_food(rob):
    global LAST_FOOD_COLLECTED
    LAST_FOOD_COLLECTED = 0



# def get_reward(rob:IRobobo, t, action):
#     image = rob.get_image_front()
#     pixels = get_number_of_target_pixels(image)

#     obstacles = (np.clip(max(rob.read_irs()), 0, 1000) / 1000) - 0.001

#     orient = rob.read_wheels()
#     ori = (abs(orient.wheel_pos_l - orient.wheel_pos_r) / (50*t +1))

#     turns = (0.9 if (action == 'turn_right' or action == 'turn_left' or action == 1 or action == 2) else 0)
    
#     #reward = pixels * (1-obstacles) * (1 - ori) # check 
#     reward = pixels * (1-obstacles) * (1 - turns)
#     #print(f"pixels: {pixels}, obs: {obstacles}, orient: {ori}, reward: {reward}")
    
#     return reward


def get_observation(rob:IRobobo):
    observation = rob.read_irs()
    observation += get_number_of_tgt_px_left_right(rob.get_image_front())

    return observation, rob.get_image_front()
    # return rob.read_irs(), rob.get_image_front()


def get_simulation_done(rob:IRobobo):
    global LAST_FOOD_COLLECTED
    return LAST_FOOD_COLLECTED == 7
    
    
    # image = rob.get_image_front()
    # pixels = get_number_of_target_pixels(image)
    # return pixels == 1.001
    
    
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
    
