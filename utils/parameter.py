# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# sensor and utility range
SENSOR_RANGE = 16  # meter

# 探索参数
MAX_EPISODE_STEP = 128
EXPLORATION_RATE_THRESHOLD = 0.99  # 探索率阈值

# agent parameters
N_CLUSTERS = 10
