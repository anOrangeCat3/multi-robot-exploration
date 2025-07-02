from env import Env
import numpy as np
from utils import *

env = Env(episode_index=0)

env.plot_env(step=0)
print(env.robot_cell)
print(env.robot_location)
print(len(env.global_frontiers))

def test1():
    env.step(next_waypoint=np.array([0, -1]))
    env.plot_env(step=1)
    print(env.robot_cell)
    print(env.robot_location)

    env.step(next_waypoint=np.array([-3, 1]))
    env.plot_env(step=2)
    print(env.robot_cell)
    print(env.robot_location)

def test2():
    frontier_coords=get_frontier_in_map(env.belief_info)
    print(frontier_coords)
    print(len(frontier_coords))

def test3():
    print(env.global_frontiers)
    env.step(next_waypoint=np.array([-3, 1]))
    env.plot_env(step=2)
    print(env.robot_cell)
    print(env.robot_location)
    print(len(env.global_frontiers))



# test1()
# test2()
test3()
