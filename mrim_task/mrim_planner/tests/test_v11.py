"""已学内容版v1.1：验证任务分配、欧氏启发函数及路径拉直。

运行：python -m unittest discover -s mrim_task/mrim_planner/tests -v
依赖NumPy、SciPy、scikit-learn、Matplotlib、PyYAML；不依赖TOPPRA。
ROS替身只提供消息和日志接口，运行的规划算法来自实际源码。
"""
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np
from scipy.spatial import KDTree

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def install_ros_shims():
    rospy = types.ModuleType('rospy')
    for name in ['logerr', 'logwarn', 'loginfo', 'signal_shutdown']:
        setattr(rospy, name, lambda *args, **kwargs: None)
    sys.modules['rospy'] = rospy
    sys.modules['rospkg'] = types.ModuleType('rospkg')

    class Message:
        pass

    for package, names in [('mrs_msgs', ['TrajectoryReference', 'Reference']),
                           ('geometry_msgs', ['Point']),
                           ('mrim_resources', ['InspectionPoint', 'InspectionProblem'])]:
        parent = types.ModuleType(package)
        module = types.ModuleType(package + '.msg')
        for name in names:
            setattr(module, name, Message)
        parent.msg = module
        sys.modules[package], sys.modules[package + '.msg'] = parent, module


install_ros_shims()
from data_types import Point, Pose, Viewpoint, Bounds
from utils import distEuclidean
from solvers.tsp_solvers import TSPSolver3D
from path_planners.grid_based.astar import AStar, Node
from path_planners.grid_based.grid_3d import Grid3D
from path_planners.sampling_based.rrt import RRT


def vp(index, x, y=0.):
    return Viewpoint(index, Pose(x, y, 2., 0.))


def task(points, allowed=None):
    return types.SimpleNamespace(
        number_of_robots=2, robot_ids=[1, 2],
        inspection_points=[types.SimpleNamespace(idx=p.idx, inspectability=allowed or [1, 2]) for p in points])


def length(path):
    return sum(distEuclidean(a, b) for a, b in zip(path, path[1:]))


class AssignmentTests(unittest.TestCase):
    def test_considers_existing_load_instead_of_only_nearest_robot(self):
        # 与学习例子对应：A原负载20，B为4；新目标分别增加2、5，选B。
        assigned = [[vp(0, 0), vp(1, 10)], [vp(0, 11, -4.5), vp(2, 11, -2.5)]]
        public = [vp(3, 11)]
        result = TSPSolver3D().clusterViewpoints(task(public), public, 'balanced', assigned)
        self.assertEqual(result, [[], public])
        # 分配函数不改动调用者已有的任务列表。
        self.assertEqual([len(group) for group in assigned], [2, 2])

    def test_prefers_nearby_robot_when_both_are_empty(self):
        assigned, public = [[vp(0, 0)], [vp(0, 10)]], [vp(1, 1)]
        result = TSPSolver3D().clusterViewpoints(task(public), public, 'balanced', assigned)
        self.assertEqual(result, [public, []])

    def test_respects_inspectability(self):
        assigned, public = [[vp(0, 0)], [vp(0, 10)]], [vp(1, 1)]
        result = TSPSolver3D().clusterViewpoints(task(public, [2]), public, 'balanced', assigned)
        self.assertEqual(result, [[], public])

    def test_assigns_all_public_points_exactly_once(self):
        assigned = [[vp(0, 0)], [vp(0, 10)]]
        public = [vp(i, float(i)) for i in range(1, 9)]
        solver = TSPSolver3D()
        result = solver.clusterViewpoints(task(public), public, 'balanced', assigned)
        self.assertEqual(sorted(p.idx for group in result for p in group), list(range(1, 9)))
        repeated = solver.clusterViewpoints(task(public), public, 'balanced', assigned)
        self.assertEqual(result, repeated)

    def test_no_public_points(self):
        result = TSPSolver3D().clusterViewpoints(task([]), [], 'balanced', [[vp(0, 0)], [vp(0, 10)]])
        self.assertEqual(result, [[], []])

    def test_missing_existing_tasks_or_eligible_robot_is_rejected(self):
        solver, public = TSPSolver3D(), [vp(1, 1)]
        with self.assertRaises(ValueError):
            solver.clusterViewpoints(task(public), public, 'balanced')
        with self.assertRaises(ValueError):
            solver.clusterViewpoints(task(public, [3]), public, 'balanced', [[vp(0, 0)], [vp(0, 10)]])


class HeuristicTests(unittest.TestCase):
    def test_three_four_five_and_g_plus_h(self):
        node = Node((0, 0, 0), route=6, goal=(3, 4, 0))
        self.assertEqual(node.heuristic, 5.)
        self.assertEqual(node.value, 11.)

    def test_height_and_goal(self):
        self.assertEqual(Node((0, 0, 2), goal=(0, 0, 5)).heuristic, 3.)
        self.assertEqual(Node((2, 3, 4), goal=(2, 3, 4)).heuristic, 0.)

    def test_existing_astar_search_uses_completed_heuristic(self):
        # 直接设置网格；不涉及原版依赖np.int的米制转换和障碍膨胀。
        grid = Grid3D((0, 0, 0), (6, 6, 3), 1.)
        node = AStar(grid, .1, 2., straighten=False).searchPath((0, 0, 1), (3, 4, 1))
        self.assertIsNotNone(node)
        self.assertAlmostEqual(node.route, 3*math.sqrt(2)+1)


class StraighteningTests(unittest.TestCase):
    def setUp(self):
        self.rrt = RRT()
        self.rrt.bounds = Bounds(Point(-1, -1, 0), Point(5, 4, 4))
        self.rrt.kdtree = KDTree([[2, 0, 2]])
        self.rrt.safety_distance = 1.
        self.path = [(0, 0, 2, 0), (0, 3, 2, None), (4, 3, 2, None), (4, 0, 2, 1)]

    def test_rrt_shortens_route_without_crossing_obstacle(self):
        result = self.rrt.halveAndTest(self.path)
        self.assertEqual(result[0], self.path[0])
        self.assertEqual(result[-1], self.path[-1])
        self.assertAlmostEqual(length(self.path), 10.)
        self.assertAlmostEqual(length(result), 8.)
        self.assertTrue(all(self.rrt.validateLinePath(a, b) for a, b in zip(result, result[1:])))

    def test_rrt_rejects_shortcut_with_new_obstacle(self):
        self.rrt.kdtree = KDTree([[2, 0, 2], [2, 1.5, 2]])
        self.assertFalse(self.rrt.validateLinePath(self.path[0], self.path[2]))
        result = self.rrt.halveAndTest(self.path)
        self.assertTrue(all(self.rrt.validateLinePath(a, b) for a, b in zip(result, result[1:])))

    def test_astar_shortcut_uses_existing_grid_obstacle_check(self):
        grid = Grid3D((0, 0, 0), (5, 5, 3), 1.)
        planner = AStar(grid, 0., 2.)
        path = [(0, 0, 1), (0, 3, 1), (4, 3, 1), (4, 0, 1)]
        self.assertEqual(planner.halveAndTest(path), [path[0], path[-1]])
        grid.array[2, 0, 1] = 1
        shortened = planner.halveAndTest(path)
        self.assertNotEqual(shortened, [path[0], path[-1]])
        self.assertTrue(all(not grid.obstacleBetween(a, b) for a, b in zip(shortened, shortened[1:])))
        self.assertLessEqual(length(shortened), length(path))

    def test_empty_and_short_paths(self):
        for path in [[], self.path[:1], self.path[:2]]:
            self.assertEqual(self.rrt.halveAndTest(path), path)
            planner = AStar(Grid3D((0, 0, 0), (2, 2, 2), 1.), 0., 2.)
            self.assertEqual(planner.halveAndTest(path), path)


if __name__ == '__main__':
    unittest.main()
