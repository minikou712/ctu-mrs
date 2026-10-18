"""
A-star path planner
@author: F. Nekovar
@maintainer: P. Petracek
"""

import itertools, time
from numpy import sqrt, argmin
import numpy as np
from queue import PriorityQueue

# # #{ class Node
class Node:

    def __init__(self, pos, route=0, parent=None, goal=None):
        self.__route  = route
        self.__pos    = pos
        self.__parent = parent
        if goal is not None:
            self.__goal = goal
        elif parent is not None:
            self.__goal = self.parent.goal
        else:
            raise Exception("Goal was not specified and the node does not have any parent!")
        self.__heuristic = self.__heuristicFunction()
        self.__value     = self.route + self.heuristic

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    @property
    def value(self):
        return self.__value

    @property
    def route(self):
        return self.__route

    @property
    def heuristic(self):
        return self.__heuristic

    @property
    def parent(self):
        return self.__parent

    @property
    def goal(self):
        return self.__goal

    @property
    def pos(self):
        return self.__pos

    # [v1.1 原代码保留 BEGIN] 第9步：补全不高估的欧氏启发函数。
    #     def __heuristicFunction(self):
    #         a = self.pos[0] - self.goal[0]
    #         b = self.pos[1] - self.goal[1]
    #         c = self.pos[2] - self.goal[2]
    #
    #         raise NotImplementedError('[STUDENTS TODO] Heuristic function guiding the state space exploration not implemented. You have to finish it on your own.')
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第9步：补全不高估的欧氏启发函数。
    def __heuristicFunction(self):
        # 欧氏距离与26邻域边代价使用同一网格单位，是一致且可采纳的下界。
        return float(np.linalg.norm(np.asarray(self.pos) - np.asarray(self.goal)))
# # #}

# # #{ class AStar
class AStar():

    def __init__(self, grid, safety_distance, timeout, straighten=True):
        self.grid            = grid
        self.safety_distance = safety_distance
        self.neighborhood    = [p for p in itertools.product([0, 1, -1], repeat=3) if not (p[0] == 0 and p[1] == 0 and p[2] == 0)] # 26-neighborhood
        self.straighten      = straighten
        self.timeout         = timeout

    def dist(self, first, second):
        a = first[0] - second[0]
        b = first[1] - second[1]
        c = first[2] - second[2]
        return sqrt(a**2 + b**2 + c**2)

    # [v1.1 原代码保留 BEGIN] 第7—8步：仅在整条连接线安全时拉直路径。
    #     def halveAndTest(self, path):
    #         pt1 = path[0]
    #         pt2 = path[-1]
    #
    #         if len(path) <= 2:
    #             return path
    #
    #         raise NotImplementedError('[STUDENTS TODO] A*: path straightening is not finished. Finish it on your own.')
    #         # Tips:
    #         #  - divide the given path by a certain ratio and use this method recursively
    #
    #         if self.grid.obstacleBetween(pt1, pt2):
    #
    #             # [STUDENTS TODO] Replace seg1 and seg2 variables effectively
    #             seg1 = path[:1]
    #             seg2 = path[1:]
    #
    #             seg1.extend(seg2)
    #             return seg1
    #
    #         return [pt1, pt2]
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第7—8步：仅在整条连接线安全时拉直路径。
    def halveAndTest(self, path):
        # 只处理单对巡检位姿之间的路径；不跨段删除规定的观察点。
        # 从当前点尝试最远的可见点，比递归切分避免深递归与重复节点。
        if len(path) <= 2:
            return path
        result, index = [path[0]], 0
        while index < len(path) - 1:
            target = len(path) - 1
            while target > index + 1 and self.grid.obstacleBetween(path[index], path[target]):
                target -= 1
            result.append(path[target])
            index = target
        return result

    # [v1.1 原代码保留 BEGIN] 第8—10步：保留精确观察位姿，验证网格连接。
    #     def generatePath(self, m_start, m_goal):
    #
    #         print("[INFO] A*: Searching for path from [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}].".format(m_start[0], m_start[1], m_start[2], m_goal[0], m_goal[1], m_goal[2]))
    #
    #         start = self.grid.metricToIndex(m_start)
    #         goal  = self.grid.metricToIndex(m_goal)
    #
    #         node = self.searchPath(start, goal)
    #         path = []
    #         path_m = []
    #         if node is None:
    #             print("[ERROR] A* did not find any path!")
    #             return None, None
    #         else:
    #             while node.parent:
    #                 path.append(node.pos)
    #                 node = node.parent
    #             path.append(start)
    #             if self.straighten:
    #                 path = self.halveAndTest(path)
    #             path.reverse()
    #             for node in path:
    #                 path_m.append(self.grid.indexToMetric(node))
    #             # keep goal as last point in path independently on resolution
    #             # path_m.append(m_goal)
    #
    #         distance = 0.0
    #         for i in range(1, len(path_m)):
    #             distance += self.dist(path_m[i - 1], path_m[i])
    #
    #         # keep heading of first point
    #         path_m[0] = (path_m[0][0], path_m[0][1], path_m[0][2], m_start[3])
    #
    #         return path_m, distance
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第8—10步：保留精确观察位姿，验证网格连接。
    def generatePath(self, m_start, m_goal):
        start, goal = self.grid.metricToIndex(m_start), self.grid.metricToIndex(m_goal)
        # 先验证真实端点；不能让量化或负索引把越界点变成合法点。
        if not self.grid.metricSegmentValid(m_start, m_start) or not self.grid.metricSegmentValid(m_goal, m_goal):
            return None, None
        node = self.searchPath(start, goal)
        if node is None:
            return None, None
        path = []
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        if self.straighten:
            path = self.halveAndTest(path)
        # 保留真实起终点及其heading，额外检查真实点到网格中心的连接。
        middle = [self.grid.indexToMetric(p) for p in path]
        path_m = [tuple(m_start)] + middle + [tuple(m_goal)]
        if any(not self.grid.metricSegmentValid(a, b) for a, b in zip(path_m, path_m[1:])):
            return None, None
        # 去掉重复网格中心，但真实起终点永远保留，包括原地转向。
        result = [path_m[0]]
        for point in path_m[1:-1]:
            if self.dist(result[-1], point) > 1e-10 and self.dist(point, path_m[-1]) > 1e-10:
                result.append(point)
        result.append(path_m[-1])
        return result, sum(self.dist(a, b) for a, b in zip(result, result[1:]))

    # [v1.1 原代码保留 BEGIN] 第10步：更新更小g及父节点，跳过过期候选。
    #     def searchPath(self, start, goal):
    #         start_node    = Node(start, goal=goal)
    #         # Initialize queue to hold open nodes
    #         open_queue    = PriorityQueue()
    #         # Initialize grid for fast lookup of best node value to positive infinity
    #         priority_grid = np.full(self.grid.dim,np.inf)
    #
    #         open_queue.put(start_node)
    #
    #         start_time = time.time()
    #
    #         while True:
    #             if open_queue.empty():
    #                 print("[ERROR] A*: open node queue is empty, could not find path!")
    #                 return None
    #
    #             best_node = open_queue.get()
    #
    #             if best_node.heuristic <= 0:
    #                 break
    #
    #             best_node_pos = best_node.pos
    #             # Throw away node if it was already opened with better value
    #             if priority_grid[best_node_pos] <= best_node.value:
    #                 continue
    #             # Assign best found node value to the lookup grid
    #             priority_grid[best_node_pos] = best_node.value
    #
    #             # Find all neighbors of the node
    #             neighbors = self.neighbors(best_node_pos)
    #             nodes     = [Node(n, best_node.route + self.dist(best_node.pos, n), best_node, goal) for n in neighbors]
    #
    #             for n in nodes:
    #                 # Add neighbors to queue if their value is better than previously opened
    #                 if priority_grid[n.pos] <= n.value:
    #                     continue
    #                 open_queue.put(n)
    #
    #             if time.time() - start_time > self.timeout:
    #                 print("[ERROR] A*: Timeout limit in searchPath() exceeded ({:.1f} s > {:.1f} s). Ending.".format(time.time() - start_time, self.timeout))
    #                 return None
    #
    #         return best_node
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第10步：更新更小g及父节点，跳过过期候选。
    def searchPath(self, start, goal):
        if self.grid.idxIsOccupied(start) or self.grid.idxIsOccupied(goal):
            return None
        # g_score在入队时更新，减少同一位置的劣质候选；出队时跳过过期记录。
        best_g = np.full(self.grid.dim, np.inf)
        best_g[start] = 0.0
        queue = PriorityQueue()
        queue.put(Node(start, goal=goal))
        deadline = time.monotonic() + self.timeout
        while not queue.empty():
            if time.monotonic() > deadline:
                return None
            current = queue.get()
            if current.route > best_g[current.pos] + 1e-12:
                continue
            if current.pos == goal:
                return current
            for neighbor in self.neighbors(current.pos):
                candidate = current.route + self.dist(current.pos, neighbor)
                if candidate + 1e-12 < best_g[neighbor]:
                    best_g[neighbor] = candidate
                    queue.put(Node(neighbor, candidate, current, goal))
        return None

    # [v1.1 原代码保留 BEGIN] 第5、8步：检查每条搜索边的安全性。
    #     def neighbors(self, pos):
    #         # Check grid bounds and obstacles for all neighbors of node
    #         neigh = []
    #         for n in self.neighborhood:
    #             idx = (pos[0] + n[0], pos[1] + n[1], pos[2] + n[2])
    #             if 0 <= idx[0] < self.grid.dim[0] and\
    #                0 <= idx[1] < self.grid.dim[1] and\
    #                0 <= idx[2] < self.grid.dim[2] and not\
    #                self.grid.idxIsOccupied(idx):
    #                     neigh.append(idx)
    #         return map(tuple, neigh)
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5、8步：检查每条搜索边的安全性。
    def neighbors(self, pos):
        # 对角移动也检查连接线，避免只检查终点而穿过障碍。
        return [idx for offset in self.neighborhood
                for idx in [tuple(pos[i] + offset[i] for i in range(3))]
                if not self.grid.idxIsOccupied(idx) and not self.grid.obstacleBetween(pos, idx)]
# # #}
