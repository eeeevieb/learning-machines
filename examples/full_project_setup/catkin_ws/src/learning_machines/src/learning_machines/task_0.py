from robobo_interface import (
    SimulationRobobo,
)
        

def wheel_and_turn(rob):
    irs_data = []

    print("IRS data: ", rob.read_irs())
    while all([r < 80 for r in  rob.read_irs()[3:]] ):
        rob.move(51, 50, millis=100)
        print("IRS data: ", rob.read_irs())
        irs_data.append(rob.read_irs())

        # Make sure it stops when it falls of the grid
        if len(irs_data) > 1000:
            if all([r == 0 for r in irs_data[-4:-1][0]]):
                rob.reset_wheels()
                return irs_data
            
    print("IRS data : ", rob.read_irs())
    print("I'm running this function")  
    irs_data.append(rob.read_irs())
    rob.move_blocking(-50, -50, 1000)
    rob.move_blocking(20, -20, 5100) 
    rob.reset_wheels() 
    

    return irs_data


def run_task_0(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print(" ")
    data = wheel_and_turn(rob)
    print(data)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

