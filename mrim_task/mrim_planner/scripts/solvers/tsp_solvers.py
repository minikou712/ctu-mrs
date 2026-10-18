"""
Various types of TSP utilizing local planners for distance estimation and path planning
@author: P. Petracek & V. Kratky & P.Vana & P.Cizek & R.Penicka
"""

import numpy as np

from random import randint

from sklearn.cluster import KMeans
from scipy.spatial.kdtree import KDTree

from utils import *
from path_planners.grid_based.grid_3d import Grid3D
from path_planners.grid_based.astar   import AStar
from path_planners.sampling_based.rrt import RRT

from solvers.LKHInvoker import LKHInvoker

class TSPSolver3D():

    ALLOWED_PATH_PLANNERS               = ('euclidean', 'astar', 'rrt', 'rrtstar')
    ALLOWED_DISTANCE_ESTIMATION_METHODS = ('euclidean', 'astar', 'rrt', 'rrtstar')
    GRID_PLANNERS                       = ('astar')

    def __init__(self):
        self.lkh = LKHInvoker()

    # # #{ setup()
    def setup(self, problem, path_planner, viewpoints):
        """setup objects required in path planning methods"""

        if path_planner is None:
            return

        assert path_planner['path_planning_method'] in self.ALLOWED_PATH_PLANNERS, 'Given method to compute path (%s) is not allowed. Allowed methods: %s' % (path_planner, self.ALLOWED_PATH_PLANNERS)
        assert path_planner['distance_estimation_method'] in self.ALLOWED_DISTANCE_ESTIMATION_METHODS, 'Given method for distance estimation (%s) is not allowed. Allowed methods: %s' % (path_planner, self.ALLOWED_DISTANCE_ESTIMATION_METHODS)

        # [v1.1 新增] 路径搜索与轨迹复查使用相同的多边形安全区。
        from path_planners.geometry import EnvironmentValidator
        path_planner['spatial_validator'] = EnvironmentValidator(problem, path_planner['safety_distance'])

        # Setup environment
        if path_planner['path_planning_method'] != 'euclidean' or path_planner['distance_estimation_method'] != 'euclidean':

            # setup KD tree for collision queries
            # [v1.1 原代码保留] 允许空障碍点云。
            # obstacles_array = np.array([[opt.x, opt.y, opt.z] for opt in problem.obstacle_points])
            # [v1.1 新实现] 允许空障碍点云。
            obstacles_array = np.array([[opt.x, opt.y, opt.z] for opt in problem.obstacle_points]).reshape(-1, 3)
            path_planner['obstacles_kdtree'] = KDTree(obstacles_array)

            # setup environment bounds
            xs = [p.x for p in problem.safety_area]
            ys = [p.y for p in problem.safety_area]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            path_planner['bounds'] = Bounds(Point(x_min, y_min, problem.min_height), Point(x_max, y_max, problem.max_height))

        # Setup 3D grid for grid-based planners
        if path_planner['path_planning_method'] in self.GRID_PLANNERS or path_planner['distance_estimation_method'] in self.GRID_PLANNERS:

            # construct grid
            x_list = [opt.x for opt in problem.obstacle_points]
            x_list.extend([vp.pose.point.x for vp in viewpoints])
            y_list = [opt.y for opt in problem.obstacle_points]
            y_list.extend([vp.pose.point.y for vp in viewpoints])
            z_list = [opt.z for opt in problem.obstacle_points]
            z_list.extend([vp.pose.point.z for vp in viewpoints])

            min_x = np.min(x_list) - path_planner['safety_distance']
            max_x = np.max(x_list) + path_planner['safety_distance']
            min_y = np.min(y_list) - path_planner['safety_distance']
            max_y = np.max(y_list) + path_planner['safety_distance']
            min_z = problem.min_height
            max_z = problem.max_height

            dim_x = int(np.floor((max_x - min_x) / path_planner['astar/grid_resolution']))+1
            dim_y = int(np.floor((max_y - min_y) / path_planner['astar/grid_resolution']))+1
            dim_z = int(np.floor((max_z - min_z) / path_planner['astar/grid_resolution']))+1

            path_planner['grid'] = Grid3D(idx_zero = (min_x, min_y,min_z), dimensions=(dim_x,dim_y,dim_z), resolution_xyz=path_planner['astar/grid_resolution'])
            # [v1.1 原代码保留] A*也使用规划器定义的世界边界。
            # path_planner['grid'].setObstacles(problem.obstacle_points, path_planner['safety_distance'])
            # [v1.1 新实现] A*也使用规划器定义的世界边界。
            path_planner['grid'].setObstacles(problem.obstacle_points, path_planner['safety_distance'])
            path_planner['grid'].bounds = path_planner['bounds']
            path_planner['grid'].spatial_validator = path_planner['spatial_validator']

    # # #}

    # #{ plan_tour()

    # [v1.1 原代码保留 BEGIN] 第4步：对称代价与路径复用，点对求解次数减半。
    #     def plan_tour(self, problem, viewpoints, path_planner=None):
    #         '''
    #         Solve TSP on viewpoints with given goals and starts
    #
    #         Parameters:
    #             problem (InspectionProblem): task problem
    #             viewpoints (list[Viewpoint]): list of Viewpoint objects
    #             path_planner (dict): dictionary of parameters
    #
    #         Returns:
    #             path (list): sequence of points with start equaling the end
    #         '''
    #
    #         # Setup 3D grid for grid-based planners and KDtree for sampling-based planners
    #         self.setup(problem, path_planner, viewpoints)
    #
    #         n              = len(viewpoints)
    #         self.distances = np.zeros((n, n))
    #         self.paths = {}
    #
    #         # find path between each pair of goals (a, b)
    #         for a in range(n):
    #             for b in range(n):
    #                 if a == b:
    #                     continue
    #
    #                 # [STUDENTS TODO]
    #                 #   - Play with distance estimates in TSP (tsp/distance_estimates parameter in config) and see how it influences the solution
    #                 #   - You will probably see that computing for all poses from both sets takes a long time.
    #                 #   - Think if you can reduce the number of computations.
    #
    #                 # get poses of the viewpoints
    #                 g1 = viewpoints[a].pose
    #                 g2 = viewpoints[b].pose
    #
    #                 # estimate distances between the viewpoints
    #                 path, distance = self.compute_path(g1, g2, path_planner, path_planner['distance_estimation_method'])
    #
    #                 # store paths/distances in matrices
    #                 self.paths[(a, b)]   = path
    #                 self.distances[a][b] = distance
    #
    #         # compute TSP tour
    #         path = self.compute_tsp_tour(viewpoints, path_planner)
    #
    #         return path
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第4步：对称代价与路径复用，点对求解次数减半。
    def plan_tour(self, problem, viewpoints, path_planner=None):
        self.setup(problem, path_planner, viewpoints)
        n = len(viewpoints)
        if n == 0:
            raise ValueError('每架无人机至少需要一个起点')
        if n == 1:
            return [Pose(*viewpoints[0].pose.asList()), Pose(*viewpoints[0].pose.asList())]
        self.distances, self.paths = np.zeros((n, n)), {}
        # 当前静态环境、无方向性位置代价具有对称性；每个无序点对只求解一次。
        # 反向路径必须复制Pose，避免后续朝向插值修改缓存和另一条路径。
        estimate = path_planner['distance_estimation_method'] if path_planner else 'euclidean'
        for a in range(n):
            for b in range(a + 1, n):
                path, distance = self.compute_path(viewpoints[a].pose, viewpoints[b].pose, path_planner, estimate)
                self.distances[a, b] = self.distances[b, a] = distance
                for first, last, points in [(a, b, path), (b, a, list(reversed(path)))]:
                    copied = [Pose(*p.asList()) for p in points]
                    copied[0] = Pose(*viewpoints[first].pose.asList())
                    copied[-1] = Pose(*viewpoints[last].pose.asList())
                    self.paths[first, last] = copied
        return self.compute_tsp_tour(viewpoints, path_planner)

    # #}

    # # #{ compute_path()

    def compute_path(self, p_from, p_to, path_planner, path_planner_method):
        '''
        Computes collision-free path (if feasible) between two points

        Parameters:
            p_from (Pose): start
            p_to (Pose): to
            path_planner (dict): dictionary of parameters
            path_planner_method (string): method of path planning

        Returns:
            path (list[Pose]): sequence of points
            distance (float): length of path
        '''
        path, distance = [], float('inf')

        # Use Euclidean metric
        if path_planner is None or path_planner_method == 'euclidean':

            path, distance = [p_from, p_to], distEuclidean(p_from, p_to)

        # Plan with A*
        elif path_planner_method == 'astar':

            astar = AStar(path_planner['grid'], path_planner['safety_distance'], path_planner['timeout'], path_planner['straighten'])
            path, distance = astar.generatePath(p_from.asList(), p_to.asList())
            if path:
                path = [Pose(p[0], p[1], p[2], p[3]) for p in path]

        # Plan with RRT/RRT*
        elif path_planner_method.startswith('rrt'):

            rrt = RRT()
            path, distance = rrt.generatePath(p_from.asList(), p_to.asList(), path_planner, rrtstar=(path_planner_method == 'rrtstar'), straighten=path_planner['straighten'])
            if path:
                path = [Pose(p[0], p[1], p[2], p[3]) for p in path]

        if path is None or len(path) == 0:
            rospy.logerr('No path found. Shutting down.')
            rospy.signal_shutdown('No path found. Shutting down.');
            exit(-2)

        return path, distance

    # # #}

    # #{ compute_tsp_tour()

    # [v1.1 原代码保留 BEGIN] 第4步：修正两点往返闭环，保留观察点朝向并固定起点。
    #     def compute_tsp_tour(self, viewpoints, path_planner):
    #         '''
    #         Compute the shortest tour based on the distance matrix (self.distances) and connect the path throught waypoints
    #
    #         Parameters:
    #             viewpoints (list[Viewpoint]): list of VPs
    #             path_planner (dict): dictionary of parameters
    #
    #         Returns:
    #             path (list[Poses]): sequence of points with start equaling the end
    #         '''
    #
    #         # compute the shortest sequence given the distance matrix
    #         sequence = self.compute_tsp_sequence()
    #
    #         path = []
    #         n    = len(self.distances)
    #
    #         for a in range(n):
    #             b = (a + 1) % n
    #             a_idx       = sequence[a]
    #             b_idx       = sequence[b]
    #             if n == 2:
    #                 a_idx = 0
    #                 b_idx = 1
    #
    #             # if the paths are already computed
    #             if path_planner['distance_estimation_method'] == path_planner['path_planning_method']:
    #                 actual_path = self.paths[(a_idx, b_idx)]
    #             # if the path planning and distance estimation methods differ, we need to compute the path
    #             else:
    #                 actual_path, _ = self.compute_path(viewpoints[a_idx].pose, viewpoints[b_idx].pose, path_planner, path_planner['path_planning_method'])
    #
    #             # join paths
    #             path = path + actual_path[:-1]
    #
    #             # force flight to end point
    #             if a == (n - 1):
    #                 path = path + [viewpoints[b_idx].pose]
    #         return path
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第4步：修正两点往返闭环，保留观察点朝向并固定起点。
    def compute_tsp_tour(self, viewpoints, path_planner):
        n = len(viewpoints)
        # 小规模任务直接确定顺序，避免LKH对1、2节点问题的限制。
        sequence = list(range(n)) if n <= 2 else self.compute_tsp_sequence()
        if sorted(sequence) != list(range(n)):
            raise ValueError('LKH返回的访问序列不是合法排列')
        # 旋转闭环，让每条轨迹严格从本机起点开始。
        index = sequence.index(0)
        sequence = sequence[index:] + sequence[:index]
        result = []
        for a, b in zip(sequence, sequence[1:] + sequence[:1]):
            if path_planner is None or path_planner['distance_estimation_method'] == path_planner['path_planning_method']:
                points = self.paths[a, b]
            else:
                points, _ = self.compute_path(viewpoints[a].pose, viewpoints[b].pose, path_planner, path_planner['path_planning_method'])
            points = [Pose(*p.asList()) for p in points]
            points[0], points[-1] = Pose(*viewpoints[a].pose.asList()), Pose(*viewpoints[b].pose.asList())
            result.extend(points[:-1])
        result.append(Pose(*viewpoints[0].pose.asList()))
        return result

    # #}

    # # #{ compute_tsp_sequence()

    # [v1.1 原代码保留 BEGIN] 第4步：小任务精确求解，大任务保留LKH。
    #     def compute_tsp_sequence(self):
    #         '''
    #         Compute the shortest sequence based on the distance matrix (self.distances) using LKH
    #
    #         Returns:
    #             sequence (list): sequence of viewpoints ordered optimally w.r.t the distance matrix
    #         '''
    #
    #         n = len(self.distances)
    #         fname_tsp = "problem"
    #         user_comment = "a comment by the user"
    #         self.lkh.writeTSPLIBfile_FE(fname_tsp, self.distances, user_comment)
    #         self.lkh.run_LKHsolver_cmd(fname_tsp, silent=True)
    #         sequence = self.lkh.read_LKHresult_cmd(fname_tsp)
    #
    #         if len(sequence) > 0 and sequence[0] is not None:
    #             for i in range(len(sequence)):
    #                 if sequence[i] is None:
    #                     new_sequence = sequence[i:len(sequence)] + sequence[:i]
    #                     sequence = new_sequence
    #                     break
    #
    #         return sequence
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第4步：小任务精确求解，大任务保留LKH。
    def compute_tsp_sequence(self):
        n = len(self.distances)
        if n <= 2:
            return list(range(n))
        if n <= 10:
            # 小任务使用Held–Karp动态规划：固定起点0，状态为(已访问集合,末点)。
            # 在小矩阵上直接得到最短闭环，并避免启动外部LKH的固定开销。
            # 复杂度O(n²2^n)，因此严格限制到10节点，规模更大仍交给LKH。
            costs, parents = {}, {}
            for last in range(1, n):
                costs[1 << (last-1), last] = self.distances[0, last]
            for mask in range(1, 1 << (n-1)):
                for last in range(1, n):
                    state = mask, last
                    if state not in costs:
                        continue
                    for following in range(1, n):
                        bit = 1 << (following-1)
                        if mask & bit:
                            continue
                        target = mask | bit, following
                        candidate = costs[state] + self.distances[last, following]
                        if candidate < costs.get(target, float('inf')):
                            costs[target], parents[target] = candidate, last
            mask = (1 << (n-1))-1
            last = min(range(1, n), key=lambda j: costs[mask, j] + self.distances[j, 0])
            reverse = []
            while mask:
                reverse.append(last)
                previous = parents.get((mask, last))
                mask ^= 1 << (last-1)
                last = previous
            return [0] + reverse[::-1]
        # 大规模任务保留LKH接口；版本1.1不声称大规模启发式是全局最优。
        fname_tsp = 'problem'
        self.lkh.writeTSPLIBfile_FE(fname_tsp, self.distances, 'MRIM v1.1')
        self.lkh.run_LKHsolver_cmd(fname_tsp, silent=True)
        return self.lkh.read_LKHresult_cmd(fname_tsp)

    # # #}

    # #{ clusterViewpoints()

    # [v1.1 原代码保留 BEGIN] 第3步：实现可复现KMeans及考虑既有任务、返航的预计完工时间均衡分配。
    #     def clusterViewpoints(self, problem, viewpoints, method):
    #         '''
    #         Clusters viewpoints into K (number of robots) clusters.
    #
    #         Parameters:
    #             problem (InspectionProblem): task problem
    #             viewpoints (list): list of Viewpoint objects
    #             method (string): method ('random', 'kmeans')
    #
    #         Returns:
    #             clusters (Kx list): clusters of points indexed for each robot:
    #         '''
    #         k = problem.number_of_robots
    #
    #         ## | ------------------- K-Means clustering ------------------- |
    #         if method == 'kmeans':
    #             # Prepare positions of the viewpoints in the world
    #             positions = np.array([vp.pose.point.asList() for vp in viewpoints])
    #
    #             raise NotImplementedError('[STUDENTS TODO] KMeans clustering of viewpoints not implemented. You have to finish it on your own')
    #             # Tips:
    #             #  - utilize sklearn.cluster.KMeans implementation (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    #             #  - after finding the labels, you may want to swap the classes (e.g., by looking at the distance of the UAVs from the cluster centers)
    #             #  - Find the start poses of the UAVs in problem.start_poses[r].position.{x,y,z}
    #
    #             # [STUDENTS TODO]: fill 1D list 'labels' of size len(viewpoints) with indices of the robots
    #             labels = [randint(0, k - 1) for vp in viewpoints]
    #
    #         ## | -------------------- Random clustering ------------------- |
    #         else:
    #             labels = [randint(0, k - 1) for vp in viewpoints]
    #
    #         # Store as clusters (2D array of viewpoints)
    #         clusters = []
    #         for r in range(k):
    #             clusters.append([])
    #
    #             for label in range(len(labels)):
    #                 if labels[label] == r:
    #                     clusters[r].append(viewpoints[label])
    #
    #         return clusters
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第3步：实现可复现KMeans及考虑既有任务、返航的预计完工时间均衡分配。
    def clusterViewpoints(self, problem, viewpoints, method, max_velocity=(1., 1., 1.), max_acceleration=(1., 1., 1.), viewpoint_distance=3.):
        from scipy.optimize import linear_sum_assignment
        k = problem.number_of_robots
        clusters = [[] for _ in range(k)]
        if not viewpoints:
            return clusters
        # 任务允许的执行者是硬约束，不能在均衡负载时破坏。
        inspection = {p.idx: p for p in problem.inspection_points}
        def eligible(vp):
            allowed = inspection[vp.idx].inspectability
            return [r for r in range(k) if problem.robot_ids[r] in allowed]
        if method == 'random':
            for vp in viewpoints:
                robots = eligible(vp)
                if not robots:
                    raise ValueError('目标没有可执行的无人机')
                clusters[robots[randint(0, len(robots)-1)]].append(vp)
            return clusters
        starts = [Pose(p.position.x, p.position.y, p.position.z, p.heading) for p in problem.start_poses]
        if method == 'kmeans':
            # 可复现聚类；簇编号没有无人机语义，使用最小成本匹配分配给起点。
            count = min(k, len(viewpoints))
            model = KMeans(n_clusters=count, random_state=0, n_init=10).fit([v.pose.point.asList() for v in viewpoints])
            costs = np.array([[np.linalg.norm(np.array(s.point.asList()) - center) for center in model.cluster_centers_] for s in starts])
            rows, cols = linear_sum_assignment(costs)
            mapping = dict(zip(cols, rows))
            for vp, label in zip(viewpoints, model.labels_):
                robots = eligible(vp)
                if not robots:
                    raise ValueError('目标没有可执行的无人机')
                preferred = mapping[label]
                robot = preferred if preferred in robots else min(robots, key=lambda r: distEuclidean(starts[r], vp.pose))
                clusters[robot].append(vp)
            return clusters
        if method != 'balanced':
            raise ValueError('未知任务分配方法: ' + method)
        # 以起止静止的直线段时间估计成本，包含既有专属目标和返航。
        # 此处不含绕障、转向或两机冲突时间，因此是启发式估计而非实际完工时间。
        speed, accel = min(max_velocity), min(max_acceleration)
        if speed <= 0 or accel <= 0:
            raise ValueError('速度、加速度上限必须为正')
        def edge(a, b):
            length = distEuclidean(a, b)
            return 2*np.sqrt(length/accel) if length < speed*speed/accel else length/speed + speed/accel
        routes = [[s, s] for s in starts]
        def insertion(route, pose):
            return min((edge(a, pose) + edge(pose, b) - edge(a, b), i+1)
                       for i, (a, b) in enumerate(zip(route, route[1:])))
        for ip in problem.inspection_points:
            if len(ip.inspectability) == 1 and ip.inspectability[0] in problem.robot_ids:
                r = list(problem.robot_ids).index(ip.inspectability[0])
                vp = inspectionPointToViewPoint(ip, viewpoint_distance)
                _, index = insertion(routes[r], vp.pose)
                routes[r].insert(index, vp.pose)
        loads = [sum(edge(a, b) for a, b in zip(route, route[1:])) for route in routes]
        # 先分配离所有起点较远的目标，降低最后才处理昂贵目标造成的失衡。
        ordered = sorted(viewpoints, key=lambda vp: (-min(distEuclidean(s, vp.pose) for s in starts), vp.idx))
        for vp in ordered:
            choices = []
            for r in eligible(vp):
                increment, index = insertion(routes[r], vp.pose)
                projected = loads.copy()
                projected[r] += increment
                choices.append((max(projected), sum(projected), r, index, increment))
            if not choices:
                raise ValueError('目标没有可执行的无人机')
            _, _, r, index, increment = min(choices)
            routes[r].insert(index, vp.pose)
            loads[r] += increment
            clusters[r].append(vp)
        return clusters

    # #}
