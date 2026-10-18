"""v1.1算法回归：数值算法用真实依赖，仅以轻量对象替代ROS消息/节点接口。

运行：python -m unittest discover -s mrim_task/mrim_planner/tests -v
依赖：NumPy、SciPy、scikit-learn、Matplotlib、PyYAML、TOPPRA。
这些测试不验证ROS传输、MRS控制跟踪或动力学仿真。
"""

import itertools
import math
from pathlib import Path
import random
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import KDTree

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
RESOURCES = SCRIPTS.parents[1] / 'mrim_resources'
sys.path.insert(0, str(SCRIPTS))


def install_ros_shims():
    # ROS1通常不在普通Python环境中；替身只提供消息字段，算法代码不替换。
    rospy = types.ModuleType('rospy')
    for name in ['logerr', 'logwarn', 'loginfo', 'signal_shutdown']:
        setattr(rospy, name, lambda *a, **kw: None)
    sys.modules['rospy'] = rospy
    rospkg = types.ModuleType('rospkg')
    rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda name: str(RESOURCES))
    sys.modules['rospkg'] = rospkg

    class Message:
        def __init__(self, *args):
            self.position = types.SimpleNamespace(x=0., y=0., z=0.)
            self.heading = 0.
            self.robot_ids, self.start_poses, self.inspection_points = [], [], []
            self.obstacle_points, self.safety_area = [], []
            self.x = self.y = self.z = 0.
            if len(args) == 3:
                self.x, self.y, self.z = args

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
from utils import ProblemLoader, inspectionPointToViewPoint, distEuclidean, wrapAngle
from path_planners.geometry import EnvironmentValidator, segment_clear
from path_planners.grid_based.grid_3d import Grid3D
from path_planners.grid_based.astar import AStar, Node
from path_planners.sampling_based.rrt import RRT, Tree
from solvers.tsp_solvers import TSPSolver3D
from trajectory import Trajectory, TrajectoryUtils


def problem(starts=((1, 1, 2), (9, 1, 2)), obstacles=(), polygon=None):
    return types.SimpleNamespace(
        number_of_robots=len(starts), robot_ids=list(range(1, len(starts)+1)),
        start_poses=[types.SimpleNamespace(position=Point(*s), heading=0.) for s in starts],
        inspection_points=[], obstacle_points=[Point(*p) for p in obstacles],
        safety_area=[Point(x, y, 0) for x, y in (polygon or [(0, 0), (10, 0), (10, 10), (0, 10)])],
        min_height=0., max_height=10.)


def config():
    return {'path_planning_method': 'rrt', 'distance_estimation_method': 'euclidean',
            'safety_distance': .3, 'timeout': 2., 'straighten': True,
            'rrt/branch_size': 1., 'rrt/sampling/method': 'uniform',
            'rrtstar/neighborhood': 2., 'astar/grid_resolution': .25}


def inspection(idx, xyz, allowed):
    return types.SimpleNamespace(idx=idx, position=Point(*xyz), inspectability=allowed,
                                 inspect_heading=0., inspect_tilt=-math.pi/2)


class GeometryTests(unittest.TestCase):
    def test_near_obstacle_between_sampling_points(self):
        tree = KDTree([[.05, .009, 0]])
        self.assertFalse(segment_clear([0, 0, 0], [1, 0, 0], tree, .01))
        self.assertFalse(segment_clear([0, 0, 0], [0, 0, 0], KDTree([[.1, 0, 0]]), .1))
        self.assertTrue(segment_clear([0, 0, 0], [1, 0, 0], KDTree(np.empty((0, 3))), .1))

    def test_concave_boundary_and_curve_margin(self):
        shape = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]
        valid = EnvironmentValidator(problem(polygon=shape), .1)
        self.assertFalse(valid([.5, 3, 2], [3.5, 3, 2]))
        self.assertTrue(valid([.5, .5, 2], [3.5, .5, 2]))
        self.assertFalse(valid([.5, .5, 2], [3.5, .5, 2], .6))
        self.assertFalse(valid([.5, .5, -1], [.5, .5, 2]))

    def test_shortcut_cannot_cross_obstacle(self):
        rrt = RRT()
        rrt.bounds = Bounds(Point(-1, -1, 0), Point(5, 4, 4))
        rrt.kdtree, rrt.safety_distance = KDTree([[2, 0, 2]]), 1.
        path = [(0, 0, 2, 0), (0, 3, 2, None), (4, 3, 2, None), (4, 0, 2, 1)]
        result = rrt.halveAndTest(path)
        self.assertEqual(result[0], path[0])
        self.assertEqual(result[-1], path[-1])
        self.assertLess(sum(distEuclidean(a, b) for a, b in zip(result, result[1:])), 10)
        self.assertTrue(all(rrt.validateLinePath(a, b) for a, b in zip(result, result[1:])))


class AStarTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid3D((0, 0, 0), (10, 10, 5), 1.)
        self.grid.setObstacles([], .1)

    def test_heuristic_and_optimal_distance(self):
        self.assertEqual(Node((0, 0, 0), goal=(3, 4, 0)).heuristic, 5.)
        search = AStar(self.grid, .1, 2., straighten=False)
        node = search.searchPath((0, 0, 1), (3, 4, 1))
        self.assertAlmostEqual(node.route, 3*math.sqrt(2)+1)

    def test_grid_bounds_and_diagonal_supercover(self):
        self.assertTrue(self.grid.idxIsOccupied((-1, 0, 0)))
        self.assertTrue(self.grid.idxIsOccupied((10, 0, 0)))
        self.grid.array[1, 0, 1] = True
        self.assertTrue(self.grid.obstacleBetween((0, 0, 1), (1, 1, 1)))
        self.assertTrue(self.grid.obstacleBetween((1, 1, 1), (0, 0, 1)))

    def test_exact_endpoints_and_occupied_start(self):
        search = AStar(self.grid, .1, 2., straighten=True)
        start, stop = (.2, .2, 1.1, .3), (4.2, 3.1, 1.2, 1.8)
        path, _ = search.generatePath(start, stop)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], stop)
        self.grid.array[self.grid.metricToIndex(start)] = True
        self.assertIsNone(search.generatePath(start, stop)[0])

    def test_shifted_origin_conversion(self):
        grid = Grid3D((-10, 4, -2), (5, 5, 5), .5)
        grid.setObstacles([], 0.)
        grid.array[2, 2, 2] = True
        self.assertTrue(grid.xyzIsOccupied((-9, 5, -1)))
        self.assertEqual(grid.metricToIndex((-10, 4, -3))[2], -2)


class TSPTests(unittest.TestCase):
    def test_balanced_uses_fixed_load_and_respects_eligibility(self):
        task = problem(starts=((1, 1, 2), (4, 1, 2)))
        fixed = inspection(1, (9, 9, 2), [1])
        public = inspection(2, (2, 2, 2), [1, 2])
        task.inspection_points = [fixed, public]
        vp = inspectionPointToViewPoint(public, 0.)
        solver = TSPSolver3D()
        clusters = solver.clusterViewpoints(task, [vp], 'balanced', viewpoint_distance=0.)
        self.assertEqual(clusters[1], [vp])
        public.inspectability = [1]
        self.assertEqual(solver.clusterViewpoints(task, [vp], 'balanced', viewpoint_distance=0.)[0], [vp])

    def test_kmeans_small_empty_and_deterministic(self):
        task = problem()
        task.inspection_points = [inspection(1, (2, 2, 2), [1, 2])]
        vp = Viewpoint(1, Pose(2, 2, 2, 0.))
        solver = TSPSolver3D()
        self.assertEqual(solver.clusterViewpoints(task, [], 'kmeans'), [[], []])
        first = solver.clusterViewpoints(task, [vp], 'kmeans')
        self.assertEqual(first, solver.clusterViewpoints(task, [vp], 'kmeans'))
        self.assertEqual(sum(map(len, first)), 1)

    def test_pair_cache_and_two_node_return(self):
        task, cfg, solver = problem(), config(), TSPSolver3D()
        cfg['path_planning_method'] = 'euclidean'
        vps = [Viewpoint(0, Pose(1, 1, 2, .2)), Viewpoint(1, Pose(4, 1, 2, 1.))]
        with patch.object(solver, 'compute_path', wraps=solver.compute_path) as calls:
            result = solver.plan_tour(task, vps, cfg)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual([p.asList() for p in result], [p.pose.asList() for p in [vps[0], vps[1], vps[0]]])
        solver.paths[0, 1][0].heading = 2.
        self.assertEqual(solver.paths[1, 0][-1].heading, .2)
        self.assertEqual(len(solver.plan_tour(task, vps[:1], cfg)), 2)

    def test_small_exact_solver_against_exhaustive_oracle(self):
        solver = TSPSolver3D()
        rng = np.random.default_rng(13)
        points = rng.normal(size=(7, 3))
        solver.distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
        route = solver.compute_tsp_sequence()
        def cost(order):
            return sum(solver.distances[a, b] for a, b in zip(order, order[1:]+order[:1]))
        optimum = min(cost([0]+list(p)) for p in itertools.permutations(range(1, 7)))
        self.assertAlmostEqual(cost(route), optimum)


class RRTTests(unittest.TestCase):
    def test_goal_bias_both_sampling_methods_and_rrtstar(self):
        for method, star in [('uniform', False), ('gaussian', False), ('uniform', True)]:
            random.seed(10)
            np.random.seed(10)
            task, cfg = problem(obstacles=[(5, 5, 2)]), config()
            cfg['rrt/sampling/method'] = method
            cfg['rrt/sampling/gaussian/stddev_inflation'] = .2
            TSPSolver3D().setup(task, cfg, [])
            rrt = RRT()
            path, _ = rrt.generatePath((1, 5, 2, .2), (9, 5, 2, 1.), cfg, rrtstar=star, straighten=True)
            self.assertIsNotNone(path)
            self.assertEqual(path[-1], (9, 5, 2, 1.))
            self.assertTrue(all(rrt.validateLinePath(a, b) for a, b in zip(path, path[1:])))
            for node, info in rrt.tree.nodes.items():
                if info['parent'] is not None:
                    self.assertAlmostEqual(info['cost'], rrt.tree.get_cost(info['parent'])+distEuclidean(node, info['parent']))

    def test_sampling_timeout_and_invalid_endpoint(self):
        rrt = RRT()
        rrt.deadline = time.monotonic()+.01
        rrt.bounds = Bounds(Point(0, 0, 0), Point(1, 1, 1))
        rrt.kdtree, rrt.safety_distance = KDTree([[.5, .5, .5]]), 10.
        with self.assertRaises(TimeoutError):
            rrt.getRandomPoint()
        task, cfg = problem(obstacles=[(1, 1, 2)]), config()
        TSPSolver3D().setup(task, cfg, [])
        self.assertIsNone(rrt.generatePath((1, 1, 2, 0), (8, 1, 2, 0), cfg)[0])


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.utility = TrajectoryUtils((2., 2., 1.), (1., 1., .5), .2)
        self.v, self.a = [2., 2., 1., 1.], [1., 1., .5, 1.]
        self.utility.spatial_validator = EnvironmentValidator(problem(polygon=[(-20, -20), (20, -20), (20, 20), (-20, 20)]), .1)

    def check_dynamics(self, poses):
        values = np.array([p.asList() for p in poses])
        values[:, 3] = np.unwrap(values[:, 3])
        values = np.vstack([values[0], values, values[-1]])
        velocity = np.diff(values, axis=0)/.2
        acceleration = np.diff(velocity, axis=0)/.2
        self.assertTrue(np.all(np.abs(velocity) <= np.array(self.v)+1e-7))
        self.assertTrue(np.all(np.abs(acceleration) <= np.array(self.a)+1e-7))

    def sample(self, waypoints, stops):
        return self.utility.sampleTrajectoryThroughWaypoints(Trajectory(.2, waypoints), stops, True, 1., .1, self.v, self.a)

    def test_time_and_length_include_segment_boundaries(self):
        points = [Pose(0, 0, 2, 0.), Pose(1, 0, 2, 0.), Pose(2, 0, 2, 0.)]
        trajectory = Trajectory(.2, points)
        trajectory.setSegment(0, points[:2])
        trajectory.setSegment(1, points[2:])
        self.assertAlmostEqual(trajectory.getTime(), .4)
        self.assertAlmostEqual(trajectory.getLength(), 2.)

    def test_distance_interpolation_angle_boundary_and_no_mutation(self):
        points = [Pose(0, 0, 2, math.radians(170)), Pose(1, 0, 2, None), Pose(4, 0, 2, math.radians(-170))]
        result = self.utility.interpolateHeading(points)
        self.assertAlmostEqual(result[1].heading, math.radians(175))
        self.assertIsNone(points[1].heading)
        result = self.utility.interpolateHeading([Pose(0, 0, 2, 0.), Pose(0, 0, 2, None), Pose(0, 0, 2, 1.)])
        self.assertAlmostEqual(result[1].heading, .5)

    def test_short_long_diagonal_stationary_and_pure_rotation(self):
        for stop in [Pose(1, 0, 2, 0.), Pose(8, 0, 2, 0.), Pose(4, 3, 4, 2.), Pose(0, 0, 2, 0.), Pose(0, 0, 2, math.pi)]:
            start = Pose(0, 0, 2, 0.)
            poses, duration = self.utility.sampleStraightSegmentWithStops(start, stop)
            self.assertEqual(poses[-1], stop)
            self.assertAlmostEqual(duration, len(poses)*.2)
            self.check_dynamics([start]+poses)
        _, duration = self.utility.sampleStraightSegmentWithStops(Pose(0, 0, 2, 0.), Pose(8, 0, 2, 0.))
        self.assertAlmostEqual(duration, 6.)

    def test_random_stopped_segments_respect_all_four_axes(self):
        rng = np.random.default_rng(27)
        self.utility.max_heading_rate = self.v[3]
        self.utility.max_heading_acceleration = self.a[3]
        for _ in range(60):
            start = Pose(*rng.uniform(-3, 3, size=4))
            stop = Pose(*rng.uniform(-3, 3, size=4))
            poses, _ = self.utility.sampleStraightSegmentWithStops(start, stop)
            self.check_dynamics([start]+poses)

    def test_continuous_saves_time_and_preserves_viewpoints(self):
        waypoints = [Pose(0, 0, 2, 0.), Pose(4, 0, 2, 0.), Pose(8, 0, 2, 0.)]
        stopped = self.sample(waypoints, True)
        smooth = self.sample(waypoints, False)
        self.assertEqual(self.utility.last_sampling_mode, 'continuous')
        self.assertLess(len(smooth.getPoses()), len(stopped.getPoses()))
        self.check_dynamics(smooth.getPoses())
        self.assertEqual(smooth.getPoses()[0], waypoints[0])
        self.assertEqual(smooth.getPoses()[-1], waypoints[-1])
        for knot, original in zip(self.utility._parametrization_knots, waypoints):
            np.testing.assert_allclose(self.utility._parametrization_path(knot), original.asList(), atol=1e-9)

    def test_turn_with_velocity_acceleration_and_heading_limits(self):
        waypoints = [Pose(0, 0, 2, 0.), Pose(3, 0, 2, .4), Pose(3, 3, 2, 1.2)]
        result = self.sample(waypoints, False)
        self.assertEqual(self.utility.last_sampling_mode, 'continuous')
        self.check_dynamics(result.getPoses())

    def test_continuous_rotation_and_wrapped_heading(self):
        for waypoints in [
            [Pose(0, 0, 2, 0.), Pose(0, 0, 2, 1.5)],
            [Pose(0, 0, 2, math.radians(170)), Pose(3, 0, 2, math.radians(-170))],
        ]:
            result = self.sample(waypoints, False)
            self.assertEqual(self.utility.last_sampling_mode, 'continuous')
            self.check_dynamics(result.getPoses())

    def test_sampled_viewpoints_respect_tight_inspection_tolerances(self):
        self.utility.inspection_position_tolerance = .03
        self.utility.inspection_heading_tolerance = .03
        waypoints = [Pose(0, 0, 2, 0.), Pose(3.19, 0, 2, .6), Pose(8, 0, 2, 1.)]
        result = self.sample(waypoints, False)
        poses = result.getPoses()
        for required in waypoints:
            self.assertTrue(any(distEuclidean(p, required) <= .03 and abs(wrapAngle(p.heading-required.heading)) <= .03 for p in poses))
        self.check_dynamics(poses)

    def test_spline_collision_falls_back_to_safe_polyline(self):
        # 样条在直角处越出折线，设置障碍仅阻挡平滑曲线。
        waypoints = [Pose(0, 0, 2, 0.), Pose(3, 0, 2, 0.), Pose(3, 3, 2, 0.)]
        task = problem(obstacles=[(1.875, -.375, 2)], polygon=[(-20, -20), (20, -20), (20, 20), (-20, 20)])
        self.utility.spatial_validator = EnvironmentValidator(task, .15)
        result = self.sample(waypoints, False)
        self.assertEqual(self.utility.last_sampling_mode, 'fallback_stops')
        self.check_dynamics(result.getPoses())
        self.assertTrue(all(self.utility.spatial_validator(a.point.asList(), b.point.asList()) for a, b in zip(result.getPoses(), result.getPoses()[1:])))

    def test_infeasible_parametrization_fallback_and_invalid_original_path(self):
        waypoints = [Pose(0, 0, 2, 0.), Pose(3, 0, 2, 0.)]
        with patch.object(self.utility, 'getParametrizedTrajectory', return_value=None):
            self.sample(waypoints, False)
        self.assertEqual(self.utility.last_sampling_mode, 'fallback_stops')
        self.utility.spatial_validator = lambda *args: False
        with self.assertRaises(ValueError):
            self.sample(waypoints, False)


if __name__ == '__main__':
    unittest.main()
