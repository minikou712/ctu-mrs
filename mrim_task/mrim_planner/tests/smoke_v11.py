"""v1.1自带小型任务算法串联验证；复用测试中的ROS消息替身。"""

import ast
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import yaml

from test_v11 import RESOURCES, SCRIPTS
from utils import ProblemLoader, wrapAngle
from planner import MrimPlanner
from path_planners.geometry import EnvironmentValidator


def run():
    random.seed(0)
    np.random.seed(0)
    task, message = ProblemLoader().loadProblem(str(RESOURCES / 'problems/apocalypse_small.problem'))
    if task is None:
        raise RuntimeError(message)
    package = SCRIPTS.parent
    defaults = yaml.safe_load((package.parent / 'mrim_manager/config/default_offline.yaml').read_text())
    custom = yaml.safe_load((package / 'config/virtual.yaml').read_text())
    planner = MrimPlanner.__new__(MrimPlanner)
    planner._plot = False
    planner._viewpoints_distance = defaults['viewpoints']['distance']
    planner._inspection_position_tolerance = defaults['viewpoints']['inspection_limits']['distance']
    planner._inspection_heading_tolerance = defaults['viewpoints']['inspection_limits']['heading']
    planner._trajectory_dt = defaults['trajectories']['dt']
    dynamics = defaults['dynamic_constraints']
    planner._max_velocity = tuple(dynamics['max_velocity'][a] for a in 'xyz')
    planner._max_acceleration = tuple(dynamics['max_acceleration'][a] for a in 'xyz')
    planner._max_heading_rate = dynamics['max_heading_rate']
    planner._max_heading_acceleration = dynamics['max_heading_rate_acceleration']
    planner._tsp_clustering_method = custom['tsp']['clustering']
    planner._sample_with_stops = custom['trajectory_sampling']['with_stops']
    planner._smoothing_distance = custom['path_smoothing']['lookahead_dist']
    planner._smoothing_sampling_step = custom['path_smoothing']['sampling_step']
    planner._collision_avoidance = custom['collision_avoidance']['method']
    planner._safety_distance_mutual = defaults['trajectories']['min_distance']['mutual']
    pp = custom['path_planner']
    planner._path_planner = {
        'timeout': pp['timeout'], 'path_planning_method': pp['method'],
        'safety_distance': defaults['trajectories']['min_distance']['obstacles'],
        'distance_estimation_method': custom['tsp']['distance_estimates'],
        'straighten': pp['straighten_paths'], 'rrt/branch_size': pp['rrt']['branch_size'],
        'rrt/sampling/method': pp['rrt']['sampling']['method'],
        'rrt/goal_bias': pp['rrt']['goal_bias'], 'rrtstar/neighborhood': None}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        trajectories, _ = planner.planTrajectories(task)
    valid = EnvironmentValidator(task, planner._path_planner['safety_distance'])
    velocity_limits = np.array(list(planner._max_velocity)+[planner._max_heading_rate])
    acceleration_limits = np.array(list(planner._max_acceleration)+[planner._max_heading_acceleration])
    results = []
    for index, trajectory in enumerate(trajectories):
        poses = trajectory.getPoses()
        values = np.array([p.asList() for p in poses])
        values[:, 3] = np.unwrap(values[:, 3])
        differences = np.diff(np.vstack([values[0], values, values[-1]]), axis=0)/planner._trajectory_dt
        accelerations = np.diff(differences, axis=0)/planner._trajectory_dt
        velocity_ratio = float(np.max(np.abs(differences)/velocity_limits))
        acceleration_ratio = float(np.max(np.abs(accelerations)/acceleration_limits))
        assert velocity_ratio <= 1.00001 and acceleration_ratio <= 1.00001
        assert all(valid(a[:3], b[:3]) for a, b in zip(values, values[1:]))
        start = task.start_poses[index]
        expected = np.array([start.position.x, start.position.y, start.position.z])
        np.testing.assert_allclose(values[0, :3], expected, atol=1e-8)
        np.testing.assert_allclose(values[-1, :3], expected, atol=1e-8)
        results.append({'robot_id': task.robot_ids[index], 'samples': len(poses),
                        'sampling_mode': trajectory.sampling_mode,
                        'sample_intervals_seconds': (len(poses)-1)*planner._trajectory_dt,
                        'length_meters': trajectory.getLength(),
                        'max_velocity_limit_ratio': velocity_ratio,
                        'max_acceleration_limit_ratio': acceleration_ratio})
    # 单独载入原有Evaluator类，真实调用原评分逻辑，避免引入其ROS节点依赖。
    manager_source = (package.parent/'mrim_manager/scripts/manager.py').read_text()
    evaluator_node = next(n for n in ast.parse(manager_source).body if isinstance(n, ast.ClassDef) and n.name == 'Evaluator')
    namespace = {'np': np, 'wrapAngle': wrapAngle}
    manager_utils = ast.parse((package.parent/'mrim_manager/scripts/mrim_manager/utils.py').read_text())
    for class_name in ['Vector4d', 'Viewpoint']:
        node = next(n for n in manager_utils.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<manager data types>', 'exec'), namespace)
    exec(compile(ast.Module(body=[evaluator_node], type_ignores=[]), '<original Evaluator>', 'exec'), namespace)
    evaluator = namespace['Evaluator'](task, planner._viewpoints_distance, planner._viewpoints_distance,
                                      planner._inspection_position_tolerance, planner._inspection_heading_tolerance,
                                      .2, [1.4, 1.4], [1.4, 1.4])
    pose_lists = [t.getPoses() for t in trajectories]
    count = max(map(len, pose_lists))
    minimum_mutual_distance = float('inf')
    for k in range(count):
        poses = [points[min(k, len(points)-1)] for points in pose_lists]
        vectors = [namespace['Vector4d'](*p.asList()) for p in poses]
        evaluator.updateInspectionStatus(*vectors)
        minimum_mutual_distance = min(minimum_mutual_distance, np.linalg.norm(poses[0].point.asArray()-poses[1].point.asArray()))
    assert evaluator.getScore() == len(task.inspection_points), '采样结果未覆盖全部巡检目标'
    report = {'problem': 'apocalypse_small.problem', 'seed': 0, 'inspection_score': int(evaluator.getScore()),
              'inspection_points': len(task.inspection_points), 'robots': results,
              'continuous_fallback_count': capture.getvalue().count('连续轨迹回退'),
              'minimum_mutual_sample_distance_meters': float(minimum_mutual_distance),
              'mutual_check_threshold_meters': planner._safety_distance_mutual,
              'mutual_samples_pass': bool(minimum_mutual_distance >= planner._safety_distance_mutual),
              'scope': '真实规划与原评分算法；ROS消息替身；未运行ROS/MRS仿真'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    run()
