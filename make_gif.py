import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np

from utils.env import Env
from utils.utils import *

def plot_env_for_gif(env, step, ax):
    '''
    基于原始plot_env方法修改的GIF绘制函数
    '''
    ax.clear()
    
    # 绘制地图
    ax.imshow(env.robot_belief, cmap='gray')
    ax.axis('off')
    
    # 绘制机器人位置
    ax.plot((env.robot_location[0] - env.belief_origin_x) / env.cell_size,
            (env.robot_location[1] - env.belief_origin_y) / env.cell_size, 
            'mo', markersize=10, zorder=5)
    
    # 绘制轨迹
    if len(env.trajectory) > 0:
        # 先画所有历史路径（较淡）
        for i, path in enumerate(env.trajectory[:-1]):  # 除了最后一条
            if path is not None and len(path) > 1:
                path_array = np.array(path)
                path_x = (path_array[:, 0] - env.belief_origin_x) / env.cell_size
                path_y = (path_array[:, 1] - env.belief_origin_y) / env.cell_size
                ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, zorder=3)
        
        # 再画当前路径（突出显示）
        if len(env.trajectory) > 0:
            current_path = env.trajectory[-1]
            if current_path is not None and len(current_path) > 1:
                path_array = np.array(current_path)
                path_x = (path_array[:, 0] - env.belief_origin_x) / env.cell_size
                path_y = (path_array[:, 1] - env.belief_origin_y) / env.cell_size
                ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, 
                       label='Current Path', zorder=5)

    # 将前沿点(机器人坐标系)转换为地图坐标系
    if len(env.frontier_cluster_centers) > 0:
        frontier_array = np.array(list(env.frontier_cluster_centers))
        frontier_x = (frontier_array[:, 0] - env.belief_origin_x) / env.cell_size
        frontier_y = (frontier_array[:, 1] - env.belief_origin_y) / env.cell_size
        # 绘制前沿点
        ax.scatter(frontier_x, frontier_y, c='red', s=30, marker='o', 
                  alpha=0.8, zorder=4, label=f'Frontiers ({len(env.global_frontiers)})')
    
    # 添加标题
    ax.set_title(f"Step {step}  Explored ratio: {env.explored_rate*100:.4g}%  Travel distance: {env.travel_dist:.4g}m",
                 fontsize=20)

def create_gif_with_original_style():
    # 初始化环境
    episode_index = 10  # change episode_index to change map
    env = Env(episode_index=episode_index)
    
    frames = []
    print("开始生成GIF帧...")
    
    for i in range(MAX_EPISODE_STEP):
        reward, done = env.step()
        
        # 创建新的图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用修改后的绘制函数
        plot_env_for_gif(env, i, ax)
        
        # 设置紧凑布局
        plt.tight_layout()
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # 转换为PIL图像
        img = Image.open(buf)
        frames.append(img.copy())
        
        buf.close()
        plt.close(fig)
        
        print(f"已生成第 {i+1} 帧 - 探索率: {env.explored_rate*100:.2f}% - 距离: {env.travel_dist:.2f}m")
        
        if done:
            print(f"探索完成！总共 {i+1} 步")
            break
    
    # 创建GIF
    print("正在保存GIF...")
    if frames:
        frames[0].save(
            'gif/robot_exploration_'+str(episode_index)+'.gif',
            save_all=True,
            append_images=frames[1:],
            duration=300,  # 每帧300ms，稍慢一些便于观察
            loop=0,
            optimize=True
        )
        print("GIF已保存为 robot_exploration.gif")
        print(f"总帧数: {len(frames)}")

if __name__ == "__main__":
    create_gif_with_original_style()
