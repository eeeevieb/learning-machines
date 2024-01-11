import cv2
import csv

from data_files import FIGRURES_DIR, RESULT_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.position())


def run_the_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    test_emotions(rob)
    test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)
        rob.set_realtime()

    test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def wheel_and_turn(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    irs_data = []

    print("IRS data: ", rob.read_irs())
    while all([r < 80 for r in  rob.read_irs()[3:]] ):
        rob.move(51, 50, millis=100)
        print("IRS data: ", rob.read_irs())
        irs_data.append(rob.read_irs())
    print("IRS data: ", rob.read_irs())
    irs_data.append(rob.read_irs())
    rob.move_blocking(-50, -50, 1000)
    rob.move_blocking(20, -20, 5100)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return irs_data

def run_task_0(rob):
    loops = 10
    irs_dict = {}
    
    for i in range(loops):
        data = wheel_and_turn(rob)
        irs_dict[i] = data
    
    print(irs_dict)

        # # irs_dict[key] = irs_dict[key].append(key)
        # with open(str(RESULT_DIR / "results.csv") , 'w') as f:
     
        #     # using csv.writer method from CSV package
        #     write = csv.writer(f)

        #     for key in irs_dict:
        #         f.write('row')
        #         print(irs_dict[key])

        # f.close()
    




# rob.move_blocking(10, 100, 1000) --> l, r, time
    
