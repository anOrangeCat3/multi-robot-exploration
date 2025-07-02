import numpy as np
import os
from skimage import io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


from utils.parameter import *
from utils.utils import *
from utils.sensor import sensor_work

class Env:
    def __init__(self, episode_index:int)->None:
        '''
        robot_location(meter): 机器人坐标系，初始位置为原点
        robot_cell(像素点坐标): 地图坐标系，
        '''
        self.sensor_range = SENSOR_RANGE  # meter
        self.cell_size = CELL_SIZE  # meter

        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        # 坐标系转换：地图坐标系 -> 机器人坐标系(机器人位置为原点)
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter
        self.robot_belief = np.ones(self.ground_truth_size) * 127

        self.global_frontiers = set()
        self.frontier_cluster_centers = set()
        self.trajectory = []

        self.explored_rate = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        
        # 初始化地图信息
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.global_frontiers = get_frontier_in_map(self.belief_info)
        self.frontier_cluster_centers = cluster_frontiers(self.global_frontiers, n_clusters=N_CLUSTERS)
        self.travel_dist = 0  # meter


    def get_waypoint(self):
        '''
        获取目标点
        '''
        best_waypoint = None
        min_distance = float('inf')
        # TODO: 对所有的frontier_cluster_centers进行A*搜索
        for center in self.frontier_cluster_centers:
            path = A_star(self.robot_location, center, self.belief_info)
            # TODO：选择路径长度最短的点
            if path:
                path_length = get_path_length(path)
                if path_length < min_distance:
                    min_distance = path_length
                    best_waypoint = center
                    best_path = np.array(path)
        self.trajectory.append(best_path)
        return best_waypoint, min_distance


    def step(self):
        '''
        Args:
            next_waypoint: 目标点坐标
        Returns:
            next_obs:
            reward: 
            done: 
        '''
        # dist = np.linalg.norm(self.robot_location - next_waypoint)
        # TODO: 使用A*算法计算距离
        next_waypoint, dist = self.get_waypoint()
        self.update_robot_location(next_waypoint)
        self.update_robot_belief()
        self.update_global_frontiers()
        self.update_frontier_cluster_centers()
        self.travel_dist += dist
        self.evaluate_exploration_rate()

        if self.explored_rate >= EXPLORATION_RATE_THRESHOLD:
            done = True
        else:
            done = False

        # TODO: 计算奖励
        reward = self.calculate_reward(dist)

        return reward, done
    
    def calculate_reward(self, dist):
        reward = 0
        return reward

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
    

    def update_frontier_cluster_centers(self):
        '''
        更新前沿点聚类中心, 判断前沿点是否在可通行区域, 如果不在，则找到最近的可达前沿点
        Returns:
            valid_centers: 可达前沿点聚类中心 ndarray
        '''
        frontier_cluster_centers = cluster_frontiers(self.global_frontiers, n_clusters=N_CLUSTERS)
        valid_centers = []
        for center in frontier_cluster_centers:
            if not is_position_accessible(center, self.belief_info):
                nearest_accessible = find_nearest_accessible_position_spiral(center, self.belief_info)
                if nearest_accessible is not None:
                    valid_centers.append(nearest_accessible)
            else:
                valid_centers.append(center)
        self.frontier_cluster_centers = np.array(valid_centers)


    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)
    

    def update_global_frontiers(self):
        self.global_frontiers = get_frontier_in_map(self.belief_info)
    
    def evaluate_exploration_rate(self):
        '''
        更新探索率
        Return: 本次探索率变化
        '''
        old_explored_rate = self.explored_rate
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return self.explored_rate - old_explored_rate


    def import_ground_truth(self, episode_index:int)->tuple[np.ndarray,np.ndarray]:
        '''
        根据训练轮次索引导入对应的地图数据
        Args:
            episode_index: 训练轮次
        Returns:
            ground_truth: 地图数据
            robot_cell: 机器人初始位置
        '''
        map_dir = f'maps'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)  # 图像降采样

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell
    

    def plot_env(self, step):
        '''
        可视化效果
        '''
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        # 绘制机器人位置
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        if len(self.trajectory) > 0:
            # 先画所有历史路径（较淡）
            for i, path in enumerate(self.trajectory[:-1]):  # 除了最后一条
                if path is not None and len(path) > 1:
                    path_array = np.array(path)
                    path_x = (path_array[:, 0] - self.belief_origin_x) / self.cell_size
                    path_y = (path_array[:, 1] - self.belief_origin_y) / self.cell_size
                    plt.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5, zorder=3)
            
            # 再画当前路径（突出显示）
            if len(self.trajectory) > 0:
                current_path = self.trajectory[-1]
                if current_path is not None and len(current_path) > 1:
                    path_array = np.array(current_path)
                    path_x = (path_array[:, 0] - self.belief_origin_x) / self.cell_size
                    path_y = (path_array[:, 1] - self.belief_origin_y) / self.cell_size
                    plt.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.5, 
                            label='Current Path', zorder=5)

        # 将前沿点(机器人坐标系)转换为地图坐标系
        if len(self.frontier_cluster_centers) > 0:
            # print(self.frontier_cluster_centers)
            frontier_array = np.array(list(self.frontier_cluster_centers))
            frontier_x = (frontier_array[:, 0] - self.belief_origin_x) / self.cell_size
            frontier_y = (frontier_array[:, 1] - self.belief_origin_y) / self.cell_size
            # 绘制前沿点
            plt.scatter(frontier_x, frontier_y, c='red', s=10, marker='o', 
                   alpha=0.8, zorder=4, label=f'Frontiers ({len(self.global_frontiers)})')
        plt.title(f"Step {step}  Explored ratio: {self.explored_rate*100:.4g}%  Travel distance: {self.travel_dist:.4g}m")
        plt.tight_layout()
        plt.show()
        

    