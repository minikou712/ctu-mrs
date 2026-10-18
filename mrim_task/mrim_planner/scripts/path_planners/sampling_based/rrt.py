import numpy as np
import random, time
from utils import distEuclidean
from data_types import Point

# # #{ class Tree
class Tree:
    def __init__(self, root_coordinates):
        self.nodes = {}
        self.add_node(root_coordinates, None, 0)
        self.valid = False

    def add_node(self, coordinates, parent, cost):
        self.nodes[coordinates] = {'parent': parent, 'cost': cost}

    def get_parent(self, node_coordinates):
        return self.nodes[node_coordinates]['parent']

    def rewire(self, node_coordinates, parent, cost):
        self.add_node(node_coordinates, parent, cost)

    def get_cost(self, node_coordinates):
        return self.nodes[node_coordinates]['cost']

    def find_path(self, end_node, start_node_hdg):

        path = []
        current_parent = end_node

        stop = False
        while not stop:
            coords = (current_parent[0], current_parent[1], current_parent[2], None)
            current_parent = self.get_parent(current_parent)

            if current_parent is None:
                coords = (coords[0], coords[1], coords[2], start_node_hdg) 
                stop   = True

            path.append(coords)

        # Return path from start point to end point.
        path.reverse()

        return path
# # #}

# # #{ class RRT
class RRT:

    def __init__(self):
        pass

    # # #{ generatePath()
    # [v1.1 原代码保留 BEGIN] 第5—7步：验证端点、直连捷径、有限超时、保留终点朝向。
    #     def generatePath(self, g_start, g_end, path_planner, rrtstar=False, straighten=False):
    #
    #         print("[INFO] {:s}: Searching for path from [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}].".format('RRT*' if rrtstar else 'RRT', g_start[0], g_start[1], g_start[2], g_end[0], g_end[1], g_end[2]))
    #         self.start = tuple(g_start[0:3])
    #         self.end   = tuple(g_end[0:3])
    #
    #         self.tree              = Tree(self.start)
    #         self.bounds            = path_planner['bounds']
    #         self.kdtree            = path_planner['obstacles_kdtree']
    #         self.safety_distance   = path_planner['safety_distance']
    #         self.timeout           = path_planner['timeout']
    #
    #         self.gaussian_sampling = path_planner['rrt/sampling/method'] == 'gaussian'
    #         if self.gaussian_sampling:
    #             self.gaussian_sampling_sigma_inflation = path_planner['rrt/sampling/gaussian/stddev_inflation']
    #
    #         rrtstar_neighborhood = path_planner['rrtstar/neighborhood'] if rrtstar else None
    #
    #         # build tree
    #         self.buildTree(branch_size=path_planner['rrt/branch_size'], rrtstar_neighborhood=rrtstar_neighborhood)
    #
    #         if not self.tree.valid:
    #             return None, None
    #
    #         # find path
    #         path = self.tree.find_path(self.end, g_start[3])
    #
    #         # smooth the path
    #         if straighten:
    #             for i in range(2):
    #                 path = self.halveAndTest(path)
    #
    #         distance = 0.0
    #         for i in range(1, len(path)):
    #             distance += distEuclidean(path[i - 1], path[i])
    #
    #         # print('rrt path:', path)
    #
    #         return path, distance
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5—7步：验证端点、直连捷径、有限超时、保留终点朝向。
    def generatePath(self, g_start, g_end, path_planner, rrtstar=False, straighten=False):
        self.start, self.end = tuple(g_start[:3]), tuple(g_end[:3])
        self.tree = Tree(self.start)
        self.bounds = path_planner['bounds']
        self.kdtree = path_planner['obstacles_kdtree']
        self.spatial_validator = path_planner.get('spatial_validator')
        self.safety_distance, self.timeout = path_planner['safety_distance'], path_planner['timeout']
        self.deadline = time.monotonic() + self.timeout
        self.goal_bias = path_planner.get('rrt/goal_bias', 0.1)
        self.gaussian_sampling = path_planner['rrt/sampling/method'] == 'gaussian'
        self.gaussian_sampling_sigma_inflation = path_planner.get('rrt/sampling/gaussian/stddev_inflation', 0.2)
        if path_planner['rrt/branch_size'] <= 0 or not 0 <= self.goal_bias <= 1:
            raise ValueError('RRT步长必须为正，目标偏置必须在[0,1]内')
        if not self.pointValid(Point(*self.start)) or not self.pointValid(Point(*self.end)):
            return None, None
        # 无障碍直连时已达到欧氏距离下界，无需随机搜索。
        if self.validateLinePath(self.start, self.end):
            return [tuple(g_start), tuple(g_end)], distEuclidean(self.start, self.end)
        neighborhood = path_planner['rrtstar/neighborhood'] if rrtstar else None
        if rrtstar and (neighborhood is None or neighborhood <= 0):
            raise ValueError('RRT*邻域半径必须为正')
        try:
            self.buildTree(path_planner['rrt/branch_size'], neighborhood)
        except TimeoutError:
            return None, None
        if not self.tree.valid:
            return None, None
        path = self.tree.find_path(self.end, g_start[3])
        path[-1] = tuple(g_end)
        if straighten:
            path = self.halveAndTest(path)
        return path, sum(distEuclidean(a, b) for a, b in zip(path, path[1:]))
    # # #}

    # # #{ pointValid()
    # [v1.1 原代码保留 BEGIN] 第5—6步：拒绝非有限坐标，并统一点与线段的安全区语义。
    #     def pointValid(self, point, check_bounds=True):
    #
    #         if check_bounds and not self.bounds.valid(point):
    #             return False
    #
    #         # check if point is at least safety_distance away from the nearest obstacle
    #         nn_dist, _  = self.kdtree.query(point.asList(), k=1)
    #         return nn_dist > self.safety_distance
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5—6步：拒绝非有限坐标，并统一点与线段的安全区语义。
    def pointValid(self, point, check_bounds=True):
        if not np.all(np.isfinite(point.asList())):
            return False
        if check_bounds and not self.bounds.valid(point):
            return False
        # 与线段复查一致，真实多边形之外的样本不会进入搜索树。
        if getattr(self, 'spatial_validator', None) is not None:
            return self.spatial_validator(point.asList(), point.asList(), 0.)
        distance, _ = self.kdtree.query(point.asList(), k=1)
        return distance > self.safety_distance
    # # #}

    # # #{ getRandomPoint()
    # [v1.1 原代码保留 BEGIN] 第6步：给安全点采样增加截止时间。
    #     def getRandomPoint(self):
    #         # Select a random point which is
    #         #  1) within world bounds and
    #         #  2) is collision-free
    #
    #         point       = None
    #         point_valid = False
    #         while not point_valid:
    #             x = random.uniform(self.bounds.point_min.x, self.bounds.point_max.x)
    #             y = random.uniform(self.bounds.point_min.y, self.bounds.point_max.y)
    #             z = random.uniform(self.bounds.point_min.z, self.bounds.point_max.z)
    #             point = Point(x, y, z)
    #
    #             point_valid = self.pointValid(point)
    #
    #         return point.asTuple()
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第6步：给安全点采样增加截止时间。
    def getRandomPoint(self):
        # 采样循环也受统一截止时间约束，狭窄或全占用环境不会无限等待。
        while time.monotonic() < self.deadline:
            point = Point(*(random.uniform(a, b) for a, b in zip(self.bounds.point_min.asList(), self.bounds.point_max.asList())))
            if self.pointValid(point):
                return point.asTuple()
        raise TimeoutError('RRT安全采样超时')
    # # #}

    # # #{ getRandomPointGaussian()
    # [v1.1 原代码保留 BEGIN] 第6步：补全高斯采样并提供均匀采样回退。
    #     def getRandomPointGaussian(self, sigma_offset=0.0):
    #
    #         # Compute mean and standard deviation
    #         st, en = [self.start[i] for i in range(3)], [self.end[i] for i in range(3)]
    #         mean   = np.mean([st, en], axis=0)
    #         sigma  = np.std([st, en], axis=0)
    #
    #         # Inflate zero stddev
    #         for i in range(3):
    #             if sigma[i] < 1e-3:
    #                 sigma[i] = 0.1
    #
    #         point       = None
    #         point_valid = False
    #         while not point_valid:
    #
    #             raise NotImplementedError('[STUDENTS TODO] Implement Gaussian sampling in RRT (Informed RRT) to speed up the process and narrow the paths.')
    #             # Tips:
    #             #  - sample from Normal distribution: use numpy.random.normal (https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html)
    #             #  - to prevent deadlocks when sampling continuously, increase the sampling space by inflating the standard deviation of the gaussian sampling
    #
    #             # STUDENTS TODO: Sample XYZ in the state space
    #             x = 0
    #             y = 0
    #             z = 0
    #
    #             point = Point(x, y, z)
    #             point_valid = self.pointValid(point)
    #
    #         return point.asTuple()
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第6步：补全高斯采样并提供均匀采样回退。
    def getRandomPointGaussian(self, sigma_offset=0.0):
        # 高斯偏置只是一种启发式，并非Informed RRT*的椭球采样。
        center = (np.array(self.start) + np.array(self.end)) / 2
        sigma = np.maximum(np.abs(np.array(self.end) - self.start) / 2, 0.1)
        for attempt in range(64):
            if time.monotonic() >= self.deadline:
                raise TimeoutError('RRT高斯采样超时')
            spread = sigma + max(0., sigma_offset) + attempt * max(0., self.gaussian_sampling_sigma_inflation)
            point = Point(*np.random.normal(center, spread))
            if self.pointValid(point):
                return point.asTuple()
        # 失败后恢复全空间均匀采样，保留探索远离起终点区域的能力。
        return self.getRandomPoint()
    # # #}

    # # #{ getClosestPoint()
    def getClosestPoint(self, point):

        # Go through all points in tree and return the closest one to point. Uses euclidian metric.
        min_distance = np.finfo(np.float32).max
        cl_point     = self.start

        for p in self.tree.nodes:
            distance = distEuclidean(point, p)
            if distance < min_distance:
                min_distance = distance
                cl_point = p

        return cl_point
    # # #}

    # # #{ setDistance()
    def setDistance(self, p_from, p_to, length):
        vec      = np.array([p_to[0] - p_from[0], p_to[1] - p_from[1], p_to[2] - p_from[2]])
        vec_norm = np.linalg.norm(vec)
        if vec_norm < length:
            return p_to

        vec = length * vec / vec_norm
        return (p_from[0] + vec[0], p_from[1] + vec[1], p_from[2] + vec[2])
    # # #}

    # # #{ validateLinePath()
    # [v1.1 原代码保留 BEGIN] 第5步：用闭线段到障碍点的精确距离替代固定间隔采样。
    #     def validateLinePath(self, p_from, p_to, discretization_factor=0.1, check_bounds=True):
    #
    #         v_from      = np.array([p_from[0], p_from[1], p_from[2]])
    #         v_to        = np.array([p_to[0], p_to[1], p_to[2]])
    #         v_from_to   = v_to - v_from
    #         len_from_to = np.linalg.norm(v_from_to)
    #
    #         if len_from_to > 1e-2:
    #             v_from_to = v_from_to / len_from_to # normalize the direction vector
    #
    #             len_ptr = 0.0
    #             while len_ptr < len_from_to:
    #                 p_ptr = v_from + len_ptr * v_from_to
    #                 if not self.pointValid(Point(p_ptr[0], p_ptr[1], p_ptr[2]), check_bounds):
    #                     return False
    #                 len_ptr += discretization_factor
    #
    #         return self.pointValid(Point(p_to[0], p_to[1], p_to[2]))
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5步：用闭线段到障碍点的精确距离替代固定间隔采样。
    def validateLinePath(self, p_from, p_to, discretization_factor=0.1, check_bounds=True):
        from path_planners.geometry import segment_clear
        # 保留原调用签名；不再依赖离散步长。矩形边界是凸集，验证端点即可。
        if check_bounds and (not self.bounds.valid(Point(*p_from[:3])) or not self.bounds.valid(Point(*p_to[:3]))):
            return False
        if getattr(self, 'spatial_validator', None) is not None:
            return self.spatial_validator(p_from, p_to, 0.)
        return segment_clear(p_from, p_to, self.kdtree, self.safety_distance)

    # # #}

    # # #{ getPointsInNeighborhood()
    def getPointsInNeighborhood(self, point, neighborhood):
        points = []
        for p in self.tree.nodes:
            if distEuclidean(point, p) < neighborhood:
                if self.validateLinePath(point, p):
                    points.append(p)
        return points
    # # #}

    # # #{ rewire()
    # [v1.1 原代码保留 BEGIN] 第6步关联选项：重连后同步后代代价，防止树环。
    #     def rewire(self, point, neighborhood):
    #         # Rewiring - if cost through given point is lower than its own, rewire it to go through that point.
    #         point_cost = self.tree.get_cost(point)
    #         for neighbor in self.getPointsInNeighborhood(point, neighborhood):
    #             rewired_cost = point_cost + distEuclidean(neighbor, point)
    #             if rewired_cost < self.tree.get_cost(neighbor):
    #                 self.tree.rewire(neighbor, point, rewired_cost)
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第6步关联选项：重连后同步后代代价，防止树环。
    def rewire(self, point, neighborhood):
        # 改父节点后递推更新整个子树代价；防止祖先被接到后代形成环。
        ancestors, node = set(), point
        while node is not None:
            ancestors.add(node)
            node = self.tree.get_parent(node)
        for neighbor in self.getPointsInNeighborhood(point, neighborhood):
            if neighbor in ancestors:
                continue
            candidate = self.tree.get_cost(point) + distEuclidean(neighbor, point)
            if candidate + 1e-12 < self.tree.get_cost(neighbor):
                self.tree.rewire(neighbor, point, candidate)
                pending = [neighbor]
                while pending:
                    parent = pending.pop()
                    for child, data in self.tree.nodes.items():
                        if data['parent'] == parent:
                            data['cost'] = self.tree.get_cost(parent) + distEuclidean(parent, child)
                            pending.append(child)
    # # #}

    # # #{ getParentWithOptimalCost()
    # [v1.1 原代码保留 BEGIN] 第6步关联选项：补全RRT*局部最低累计代价父节点。
    #     def getParentWithOptimalCost(self, point, closest_point, neighborhood):
    #
    #         parent = closest_point
    #         cost   = self.tree.get_cost(closest_point) + distEuclidean(closest_point, point)
    #
    #         neighborhood_points = self.getPointsInNeighborhood(point, neighborhood)
    #         for neighbor in neighborhood_points:
    #
    #             raise NotImplementedError('[STUDENTS TODO] Getting node parents in RRT* not implemented. You have to finish it.')
    #             # Tips:
    #             #  - look for neighbor which when connected yields minimal path cost all the way back to the start
    #             #  - you might need functions 'self.tree.get_cost()' or 'distEuclidean()'
    #
    #             # [STUDENTS TODO]: find a parent with optimal cost and fill these two variables
    #             cost = float('inf')
    #             parent = closest_point
    #
    #         return parent, cost
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第6步关联选项：补全RRT*局部最低累计代价父节点。
    def getParentWithOptimalCost(self, point, closest_point, neighborhood):
        parent = closest_point
        cost = self.tree.get_cost(parent) + distEuclidean(parent, point)
        for neighbor in self.getPointsInNeighborhood(point, neighborhood):
            candidate = self.tree.get_cost(neighbor) + distEuclidean(neighbor, point)
            if candidate < cost:
                parent, cost = neighbor, candidate
        return parent, cost
    # # #}

    # # #{ buildTree()
    # [v1.1 原代码保留 BEGIN] 第6步：目标偏置、累计代价、重复节点保护与全流程超时。
    #     def buildTree(self, branch_size, rrtstar_neighborhood=None):
    #
    #         rrtstar                      = rrtstar_neighborhood is not None
    #         start_time                   = time.time()
    #         rrt_gaussian_sigma_inflation = 0.0
    #
    #         while not self.tree.valid:
    #
    #             point         = self.getRandomPoint() if not self.gaussian_sampling else self.getRandomPointGaussian(rrt_gaussian_sigma_inflation)
    #             closest_point = self.getClosestPoint(point)
    #
    #             # normalize vector closest_point->point to length of branch_size
    #             point = self.setDistance(closest_point, point, branch_size)
    #
    #             if self.validateLinePath(point, closest_point):
    #
    #                 if not rrtstar:
    #                     parent, cost = closest_point, distEuclidean(point, closest_point)
    #                 else:
    #                     # RRT*: Choose neighbor which will provide the best cost.
    #                     parent, cost = self.getParentWithOptimalCost(point, closest_point, rrtstar_neighborhood)
    #
    #                 self.tree.add_node(point, parent, cost)
    #
    #                 if rrtstar:
    #                     # RRT*: Rewire all neighbors
    #                     self.rewire(point, rrtstar_neighborhood)
    #
    #                 # Check, whether end is reachable. If yes, stop the tree generation.
    #                 if distEuclidean(point, self.end) < branch_size and self.validateLinePath(point, self.end):
    #                     self.tree.add_node(self.end, point, self.tree.get_cost(point) + distEuclidean(self.end, point))
    #                     self.tree.valid = True
    #
    #             # Gaussian sampling: increase standard deviation of the sampling Normal distribution
    #             elif self.gaussian_sampling:
    #                 rrt_gaussian_sigma_inflation += self.gaussian_sampling_sigma_inflation
    #
    #             if time.time() - start_time > self.timeout:
    #                 print("[ERROR] {:s}: Timeout limit in buildTree() exceeded ({:.1f} s > {:.1f} s). Ending.".format('RRT*' if rrtstar else 'RRT', time.time() - start_time, self.timeout))
    #                 return
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第6步：目标偏置、累计代价、重复节点保护与全流程超时。
    def buildTree(self, branch_size, rrtstar_neighborhood=None):
        # 目标偏置使树更常朝目标扩展；剩余采样仍探索环境。
        while time.monotonic() < self.deadline:
            if random.random() < self.goal_bias:
                sample = self.end
            else:
                sample = self.getRandomPointGaussian() if self.gaussian_sampling else self.getRandomPoint()
            closest = self.getClosestPoint(sample)
            point = self.setDistance(closest, sample, branch_size)
            # 跳过重复节点，避免零距离扩展覆盖父节点导致回溯成环。
            if point in self.tree.nodes or not self.validateLinePath(closest, point):
                continue
            if rrtstar_neighborhood is None:
                parent, cost = closest, self.tree.get_cost(closest) + distEuclidean(closest, point)
            else:
                parent, cost = self.getParentWithOptimalCost(point, closest, rrtstar_neighborhood)
            self.tree.add_node(point, parent, cost)
            if rrtstar_neighborhood is not None:
                self.rewire(point, rrtstar_neighborhood)
            if distEuclidean(point, self.end) <= branch_size and self.validateLinePath(point, self.end):
                if point != self.end:
                    self.tree.add_node(self.end, point, cost + distEuclidean(point, self.end))
                self.tree.valid = True
                # 维持原接口的首次可行解停止行为；不声称此RRT*变体渐近最优。
                return
    # # #}

    # # #{ halveAndTest()
    # [v1.1 原代码保留 BEGIN] 第7步：逐次选取安全的最远捷径，保持单段起终点。
    #     def halveAndTest(self, path):
    #         pt1 = path[0][0:3]
    #         pt2 = path[-1][0:3]
    #
    #         if len(path) <= 2:
    #             return path
    #
    #         raise NotImplementedError('[STUDENTS TODO] RRT: path straightening is not finished. Finish it on your own.')
    #         # Tips:
    #         #  - divide the given path by a certain ratio and use this method recursively
    #         #  - validateLinePath() returns true if there are no obstacles between two points and vice-versa
    #         # note that path straightening does not have an effect when using correctly implemented rrtstar
    #
    #         if not self.validateLinePath(pt1, pt2, check_bounds=False):
    #
    #             # [STUDENTS TODO] Replace seg1 and seg2 variables effectively
    #             seg1 = path[:1]
    #             seg2 = path[1:]
    #
    #             seg1.extend(seg2)
    #             return seg1
    #
    #         return [path[0], path[-1]]
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第7步：逐次选取安全的最远捷径，保持单段起终点。
    def halveAndTest(self, path):
        if len(path) <= 2:
            return path
        shortened, index = [path[0]], 0
        while index < len(path) - 1:
            target = len(path) - 1
            while target > index + 1 and not self.validateLinePath(path[index], path[target]):
                target -= 1
            shortened.append(path[target])
            index = target
        return shortened
    # # #}

# # #}
