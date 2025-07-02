import numpy as np
import os
from skimage import io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


from parameter import *
from utils import *
from sensor import sensor_work

class Env:
    def __init__(self, episode_index:int)->None:
        '''
        robot_location(meter): 机器人坐标系，初始位置为原点
        robot_cell(像素点坐标): 地图坐标系，
        '''
        self.sensor_range = SENSOR_RANGE
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
            if not self.is_position_accessible(center):
                nearest_accessible = self.find_nearest_accessible_position_spiral(center)
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
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)


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

        # plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        # 绘制机器人位置
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        # plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
        #          (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        # plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        # print(self.global_frontiers)
        if len(self.trajectory) > 0:
            # 先画所有历史路径（较淡）
            for i, path in enumerate(self.trajectory[:-1]):  # 除了最后一条
                if path is not None and len(path) > 1:
                    path_array = np.array(path)
                    path_x = (path_array[:, 0] - self.belief_origin_x) / self.cell_size
                    path_y = (path_array[:, 1] - self.belief_origin_y) / self.cell_size
                    plt.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.3, zorder=3)
            
            # 再画当前路径（突出显示）
            if len(self.trajectory) > 0:
                current_path = self.trajectory[-1]
                if current_path is not None and len(current_path) > 1:
                    path_array = np.array(current_path)
                    path_x = (path_array[:, 0] - self.belief_origin_x) / self.cell_size
                    path_y = (path_array[:, 1] - self.belief_origin_y) / self.cell_size
                    plt.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.8, 
                            label='Current Path', zorder=5)

        # 将前沿点(机器人坐标系)转换为地图坐标系
        frontier_array = np.array(list(self.frontier_cluster_centers))
        frontier_x = (frontier_array[:, 0] - self.belief_origin_x) / self.cell_size
        frontier_y = (frontier_array[:, 1] - self.belief_origin_y) / self.cell_size
        # 绘制前沿点
        plt.scatter(frontier_x, frontier_y, c='red', s=10, marker='o', 
                   alpha=0.8, zorder=4, label=f'Frontiers ({len(self.global_frontiers)})')
        plt.title(f"Step {step}  Explored ratio: {self.explored_rate:.4g}  Travel distance: {self.travel_dist:.4g}")
        plt.tight_layout()
        plt.show()
        # plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        # frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        # plt.close()
        # self.frame_files.append(frame)

    def is_position_accessible(self, position):
        """
        检查位置是否可通行
        Args:
            position: np.array 或 tuple, 位置坐标 (机器人坐标系)
        Returns:
            bool: True如果可通行，False如果不可通行
        """
        try:
            # 确保输入是numpy数组
            if isinstance(position, (list, tuple)):
                position = np.array(position)
            
            # 转换为网格坐标
            cell_position_raw = get_cell_position_from_coords(
                position.reshape(1, -1), self.belief_info
            )
            
            # 处理可能的标量返回值
            if isinstance(cell_position_raw, np.ndarray):
                if cell_position_raw.ndim == 0:
                    # 0维数组，无法索引
                    print(f"位置 {position} 转换结果是标量: {cell_position_raw}")
                    return False
                elif cell_position_raw.ndim == 1:
                    cell_position = cell_position_raw
                else:
                    cell_position = cell_position_raw[0]
            else:
                cell_position = np.array(cell_position_raw)
            
            # 确保是1维数组且有2个元素
            cell_position = np.array(cell_position).flatten()
            if len(cell_position) != 2:
                print(f"位置 {position} 转换结果格式错误: {cell_position}")
                return False
            
            # 检查是否在地图范围内
            map_height, map_width = self.robot_belief.shape
            if not (0 <= cell_position[0] < map_width and 0 <= cell_position[1] < map_height):
                return False
            
            # 检查是否是自由空间
            cell_value = self.robot_belief[cell_position[1], cell_position[0]]
            return cell_value == FREE
            
        except Exception as e:
            print(f"位置 {position} 不在可通行区域: {e}")
            return False
        
    def find_nearest_accessible_position_spiral(self, center_coords, max_search_radius=3.0):
        """
        使用螺旋搜索找到距离指定中心点最近的可通行位置
        Args:
            center_coords: np.array 或 tuple, 聚类中心坐标
            max_search_radius: float, 最大搜索半径 (米)
        Returns:
            np.array: 最近的可通行位置坐标，如果没有则返回None
        """
        # 确保输入是numpy数组
        if isinstance(center_coords, (list, tuple)):
            center_coords = np.array(center_coords)
        
        # 转换中心点为网格坐标
        try:
            center_cell_raw = get_cell_position_from_coords(
                center_coords.reshape(1, -1), self.belief_info
            )
            
            # 处理可能的标量返回值
            if isinstance(center_cell_raw, np.ndarray):
                if center_cell_raw.ndim == 0:
                    print(f"中心点 {center_coords} 转换结果是标量")
                    return None
                elif center_cell_raw.ndim == 1:
                    center_cell = center_cell_raw
                else:
                    center_cell = center_cell_raw[0]
            else:
                center_cell = np.array(center_cell_raw)
            
            # 确保是正确格式
            center_cell = np.array(center_cell).flatten()
            if len(center_cell) != 2:
                print(f"中心点 {center_coords} 转换格式错误: {center_cell}")
                return None
            
        except Exception as e:
            print(f"中心点坐标转换失败: {e}")
            return None
        
        # 检查是否在地图范围内
        map_height, map_width = self.robot_belief.shape
        if not (0 <= center_cell[0] < map_width and 0 <= center_cell[1] < map_height):
            print(f"中心点超出地图范围: {center_cell}, 地图大小: {map_width}x{map_height}")
            return None
        
        # 最大搜索半径对应的网格数
        max_radius_cells = int(max_search_radius / self.cell_size) + 1
        
        # 螺旋搜索：从中心开始，逐渐扩大搜索半径
        for radius in range(max_radius_cells + 1):
            # 在当前半径的所有位置中搜索
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 只搜索当前半径边界上的点（避免重复搜索内部点）
                    if abs(dx) != radius and abs(dy) != radius and radius > 0:
                        continue
                    
                    candidate_cell = np.array([center_cell[0] + dx, center_cell[1] + dy])
                    
                    # 检查边界
                    if not (0 <= candidate_cell[0] < map_width and 0 <= candidate_cell[1] < map_height):
                        continue
                    
                    # 检查是否可通行
                    if self.robot_belief[candidate_cell[1], candidate_cell[0]] == FREE:
                        # 转换回实际坐标
                        candidate_coords = get_coords_from_cell_position(
                            candidate_cell.reshape(1, -1), self.belief_info
                        )
                        
                        # 处理返回值
                        if isinstance(candidate_coords, np.ndarray):
                            if candidate_coords.ndim == 1:
                                result_coords = candidate_coords
                            else:
                                result_coords = candidate_coords.flatten()
                        else:
                            result_coords = np.array(candidate_coords)
                        
                        # 检查实际距离是否在搜索半径内
                        distance = np.linalg.norm(center_coords - result_coords)
                        if distance <= max_search_radius:
                            return result_coords
        return None