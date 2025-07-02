from env import Env
import numpy as np
from utils import *
import matplotlib.pyplot as plt

done=False
env = Env(episode_index=0)
plt.ion()  # 开启交互模式
fig = plt.figure()

for i in range(MAX_EPISODE_STEP):
    plt.clf()
    reward,done=env.step()
    env.plot_env(step=i)
    
    if done:
        break
    plt.pause(1)

plt.ioff()  # 关闭交互模式
plt.show()
