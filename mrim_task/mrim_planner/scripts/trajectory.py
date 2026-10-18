"""
Custom TSP Loader
@author: P. Petracek, V. Kratky
"""

import math
# [v1.1 原代码保留] 移除未使用的强制依赖。
# import dubins
# [v1.1 新实现] 移除未使用的强制依赖。
# v1.1：当前文件未调用dubins，无需强制安装该扩展。

from utils import segmentPointDist, distEuclidean, lineSphereIntersections, simulateStep, wrapAngle, angleDiff, poseInDistance
from data_types import Pose, Viewpoint
import numpy as np

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
ta.setup_logging("INFO")

## | ------------------------- Classes ------------------------ |

# # #{ class Segment
class Segment:

    def __init__(self, idx_from, idx_to, poses=[]):
        self.idx_from = idx_from
        self.idx_to   = idx_to
        self.poses    = poses

    def getTime(self, dT):
        return len(self.poses) * dT
# # #}

# # #{ class Trajectory
class Trajectory:

    def __init__(self, dT, waypoints):
        self.dT                    = dT
        self.poses                 = []
        self.length                = 0.0
        self.len_changed           = True

        self.waypoints        = waypoints

        self.segments = [Segment(i-1, i) for i in range(1, len(self.waypoints))]

    # # #{ setSegment()
    def setSegment(self, seg_idx_from, poses):
        self.segments[seg_idx_from].poses = poses
        self.len_changed                  = True
    # # #}

    # # #{ getPoses()
    def getPoses(self):
        poses = []
        for seg in self.segments:
            poses = poses + seg.poses
        return poses
    # # #}

    # # #{ delaySegment()
    def delaySegment(self, seg_idx, t, at_start=False):
        '''
        Delays particular segment by t seconds either in start or end point.

        Parameters:
            seg_idx (int): segment index
            t (float): delay in seconds
            at_start (bool): delay is added at the start of the segment if True, at the end otherwise
        '''

        if seg_idx >= len(self.segments):
            return

        segment = self.segments[seg_idx]
        if len(segment.poses) == 0:
            return

        # Count number of poses to be added
        count = int(t / self.dT)

        # Copy start/end sample count-times
        index       = 0 if at_start else -1
        sequence    = count * [segment.poses[index]]

        # Replace poses at given segment
        self.segments[seg_idx].poses = segment.poses[:index] + sequence + segment.poses[index:]

    # # #}

    # # #{ delayStart()
    def delayStart(self, t):
        '''
        Delays trajectory at its beginning by t seconds.

        Parameters:
            t (float): delay in seconds
        '''

        self.delaySegment(0, t, at_start=True)

    # # #}

    # # #{ print()
    def print(self):
        print('Trajectory:')
        print(' - time: {:.1f} s'.format(self.getTime()))
        print(' - length: {:.1f} m'.format(self.getLength()))
        print(' - poses:')
        poses = self.getPoses()
        for i in range(len(poses)):
            p        = poses[i]
            appendix = ''
            print('  [{:d}]: ({:.4f}, {:.4f}, {:.4f}) | hdg: {:.2f}{:s}'.format(i, p[0], p[1], p[2], p[3], appendix))
    # # #}

    # # #{ GETTERS
    # [v1.1 原代码保留 BEGIN] 第11步：时间统计与dt样本的真实时间跨度一致。
    #     def getTime(self):
    #         samples_count = sum([len(seg.poses) for seg in self.segments])
    #         return samples_count * self.dT
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第11步：时间统计与dt样本的真实时间跨度一致。
    def getTime(self):
        # N个等间隔样本从t=0开始，共有N-1个时间区间。
        return max(0, sum(len(seg.poses) for seg in self.segments)-1) * self.dT

    # [v1.1 原代码保留 BEGIN] 第11步：完整统计跨采样段的路径长度。
    #     def getLength(self):
    #
    #         # if not cached
    #         if self.len_changed:
    #             self.len_changed = False
    #             self.length      = 0.0
    #
    #             for seg in self.segments:
    #                 for i in range(1, len(seg.poses)):
    #                     self.length += distEuclidean(seg.poses[i - 1], seg.poses[i])
    #
    #         return self.length
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第11步：完整统计跨采样段的路径长度。
    def getLength(self):
        if self.len_changed:
            # 后续段不重复首点，逐段求和会漏掉段间最后一个采样间隔。
            poses = self.getPoses()
            self.length = sum(distEuclidean(a, b) for a, b in zip(poses, poses[1:]))
            self.len_changed = False
        return self.length
    # # #}

# # #}

# # #{ class TrajectoryUtils
class TrajectoryUtils():

    # #{ __init__()
    def __init__(self, max_velocity, max_acceleration, dT):
        self.dT               = dT
        self.max_velocity     = max_velocity
        self.max_acceleration = max_acceleration
    # #}

    ## | ------------- Functions: trajectory (re)sampling ------------- |

    # # #{ interpolateHeading()

    # [v1.1 原代码保留 BEGIN] 连续轨迹的必要依赖：最短角差、路程插值及零长度处理。
    #     def interpolateHeading(self, waypoints:Viewpoint):
    #         '''
    #         Interpolates linearly the UAV heading between waypoints.
    #
    #         Parameters:
    #             waypoints (list[Pose]): list of poses where some of them contain None in place of heading
    #
    #         Returns:
    #             wps_interp (list[Pose]): list of interpolated waypoints with size equaling the input waypoints
    #         '''
    #
    #         assert waypoints[0].heading  is not None, "[tsp_trajectory::interpolateHeading()] Heading of first waypoints sample is None. Can't interpolate."
    #         assert waypoints[-1].heading is not None, "[tsp_trajectory::interpolateHeading()] Heading of last waypoints sample is None. Last sample should equal the start."
    #
    #         wps_interp = []
    #
    #         idx = 0
    #         while idx != (len(waypoints) - 1):
    #             g_from  = waypoints[idx]
    #
    #             # Find subtraj: sequence of poses (included) with defined heading
    #             subtraj = [g_from]
    #             for i in range(idx + 1, len(waypoints)):
    #                 pose = waypoints[i]
    #                 subtraj.append(pose)
    #
    #                 # Break at first heading-less pose
    #                 if pose.heading is not None:
    #                     idx = i
    #                     break
    #
    #             # get desired heading change from the beginning to end of this subtrajectory
    #             g_to  = subtraj[-1]
    #             delta_heading = wrapAngle(g_to.heading - g_from.heading)
    #
    #             # get initial heading and subtraj length
    #             hdg_from    = wrapAngle(g_from.heading)
    #             current_heading = hdg_from
    #             subtraj_len = self.getLength(subtraj)
    #
    #             ## | ----------------------- Interpolate ---------------------- |
    #
    #             # include start node
    #             wps_interp.append(subtraj[0])
    #
    #             # interpolate headings
    #             for i in range(1, len(subtraj) - 1):
    #
    #                 subtraj_point_0 = subtraj[i - 1].point
    #                 subtraj_point_1 = subtraj[i].point
    #
    #                 # [STUDENTS TODO] Implement heading interpolation here
    #                 # Guidelines:
    #                 #  - A subtrajectory is a segment of the trajectory consisting of consecutive poses. A subtrajectory is defined by its start and end poses.
    #                 #  - Interpolate the heading linearly as a function of the distance along the subtrajectory.
    #                 #  - Always wrap angles to the range (-pi, pi) using wrapAngle() from utils.py.
    #                 #  - Use distEuclidean() from utils.py to compute distances between points.
    #
    #                 # [STUDENTS TODO] Change variable 'desired_heading', nothing else
    #                 desired_heading = waypoints[0].heading
    #
    #                 # replace heading
    #                 current_heading   = desired_heading
    #                 wp         = subtraj[i]
    #                 wp.heading = desired_heading
    #                 wps_interp.append(wp)
    #
    #         # include the very last node
    #         wps_interp.append(waypoints[-1])
    #
    #         return wps_interp
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 连续轨迹的必要依赖：最短角差、路程插值及零长度处理。
    def interpolateHeading(self, waypoints):
        # 第14—16步连续时间化所需的前置依赖：按实际累计路程补齐中间朝向。
        # 复制输入，防止修改TSP缓存；指定heading的观察点作为不可跨越的锚点。
        if not waypoints or waypoints[0].heading is None or waypoints[-1].heading is None:
            raise ValueError('路径首尾必须具有朝向')
        result = [Pose(*wp.asList()) for wp in waypoints]
        anchors = [i for i, wp in enumerate(result) if wp.heading is not None]
        for start, stop in zip(anchors, anchors[1:]):
            distances = [0.]
            for i in range(start + 1, stop + 1):
                distances.append(distances[-1] + distEuclidean(result[i-1], result[i]))
            delta = wrapAngle(result[stop].heading - result[start].heading)
            for offset in range(1, stop - start):
                # 全段位置相同时没有路程比例，用序号比例描述原地转向。
                ratio = distances[offset] / distances[-1] if distances[-1] > 1e-12 else offset / (stop-start)
                result[start+offset].heading = wrapAngle(result[start].heading + delta*ratio)
        return result

    # # #}

    # #{ sampleStraightSegmentWithStops()
    # [v1.1 原代码保留 BEGIN] 第11—13步：按方向利用各轴动力学，同步朝向，正确采样短段和末端。
    #     def sampleStraightSegmentWithStops(self, start, stop):
    #         '''
    #         Samples linear segment with zero velocity at start and stop.
    #
    #         Parameters:
    #             start (Pose): start of the segment
    #             stop (Pose): end of the segment
    #
    #         Returns:
    #             poses (list[Pose]): list of poses which respect dynamic constraints of the UAV
    #         '''
    #
    #         # print("sampleWithStops from", start, "to", stop)
    #
    #         init_velocity     = 0
    #         final_velocity    = 0
    #         sample_start_time = 0
    #
    #         poses                = []
    #         trajectory_part_time = 0
    #         # acc = 0 # no jeck jet
    #         dist_total = distEuclidean(start, stop)
    #         #print("dist_total", dist_total)
    #
    #         # [STUDENTS TODO] Rework the method to per-axis computation if you want to exploit the allowed dynamics in all axes
    #         # Set minimal velocity/acceleration to the axis limit with minimal constraint
    #         min_ax_vel = min(self.max_velocity)
    #         min_ax_acc = min(self.max_acceleration)
    #
    #         time_from_init_to_max_vel  = (min_ax_vel - init_velocity) / min_ax_acc
    #         time_from_max_to_final_vel = (min_ax_vel - final_velocity) / min_ax_acc
    #
    #         dist_from_init_to_max_vel  = 0.5 * (min_ax_vel + init_velocity) * time_from_init_to_max_vel  # average speed * time
    #         dist_from_max_vel_to_final = 0.5 * (min_ax_vel + final_velocity) * time_from_max_to_final_vel  # average speed * time
    #
    #         """
    #         print("time_from_init_to_max_vel", time_from_init_to_max_vel, "s")
    #         print("time_from_max_to_final_vel", time_from_max_to_final_vel, "s")
    #         print("dist_from_init_to_max_vel", dist_from_init_to_max_vel, "m")
    #         print("dist_from_max_vel_to_final", dist_from_max_vel_to_final, "m")
    #         """
    #
    #         if dist_total < dist_from_init_to_max_vel + dist_from_max_vel_to_final: # can not reach maximal speed in straight line
    #             #print("can not reach max vel in trajectory")
    #             t      = 0
    #             sample = 0
    #
    #             if init_velocity == 0 and final_velocity == 0:
    #
    #                 time_to_possible_max_vel = math.sqrt(dist_total / min_ax_acc)
    #                 velocity_in_middle       = time_to_possible_max_vel * min_ax_acc
    #                 trajectory_part_time     = 2 * time_to_possible_max_vel
    #
    #             elif init_velocity > final_velocity:  # initial velocity is larger than final, in the end is additinal decelerating
    #
    #                 time_final_decel         = (init_velocity - final_velocity) / min_ax_acc
    #                 dist_final_decel         = time_final_decel * (init_velocity + final_velocity) * 0.5
    #                 dist_acc_decc            = dist_total - dist_final_decel
    #                 time_to_possible_max_vel = (-init_velocity + math.sqrt(init_velocity ** 2 + min_ax_acc * dist_acc_decc)) / min_ax_acc
    #                 velocity_in_middle       = init_velocity + time_to_possible_max_vel * min_ax_acc
    #                 trajectory_part_time     = time_to_possible_max_vel + time_final_decel
    #
    #             else:
    #
    #                 time_init_accel          = (final_velocity - init_velocity) / min_ax_acc
    #                 dist_init_accel          = time_init_accel * (init_velocity + final_velocity) * 0.5
    #                 dist_acc_decc            = dist_total - dist_init_accel
    #                 time_to_possible_max_vel = time_init_accel + (-final_velocity + math.sqrt(final_velocity ** 2 + min_ax_acc * dist_acc_decc)) / min_ax_acc
    #                 velocity_in_middle       = init_velocity + time_to_possible_max_vel * min_ax_acc
    #
    #                 """
    #                 print("time_init_accel", time_init_accel)
    #                 print("dist_init_accel", dist_init_accel)
    #                 print("dist_total", dist_total)
    #                 print("dist_acc_decc", dist_acc_decc)
    #                 print("such dist is", 0.5 * (velocity_in_middle + init_velocity) * time_to_possible_max_vel * 2)
    #                 """
    #                 trajectory_part_time = 2 * time_to_possible_max_vel - time_init_accel
    #
    #             """
    #             print("time_to_possible_max_vel", time_to_possible_max_vel)
    #             print("velocity_in_middle", velocity_in_middle)
    #             print("sample_start_time", sample_start_time)
    #             """
    #
    #             while (sample + 1) * self.dT <= time_to_possible_max_vel - sample_start_time:
    #
    #                 t = (sample + 1) * self.dT + sample_start_time
    #                 v = init_velocity + min_ax_acc * t
    #                 s = init_velocity * t + 0.5 * min_ax_acc * (t ** 2)
    #
    #                 pose_in_dist = poseInDistance(start, stop, s)
    #                 poses.append(pose_in_dist)
    #                 sample += 1
    #                 #print("t", t, "v", v, "s", s, "sample", sample)
    #
    #
    #             #print("end acc")
    #
    #             while (sample + 1) * self.dT <= trajectory_part_time - sample_start_time:
    #
    #                 t      = (sample + 1) * self.dT + sample_start_time
    #                 t_part = t - time_to_possible_max_vel
    #                 v      = velocity_in_middle - min_ax_acc * t_part
    #                 s      = time_to_possible_max_vel * 0.5 * (velocity_in_middle + init_velocity) + velocity_in_middle * t_part - 0.5 * min_ax_acc * (t_part ** 2)
    #
    #                 pose_in_dist = poseInDistance(start, stop, s)
    #                 poses.append(pose_in_dist)
    #                 sample += 1
    #                 #print("t", t, "v", v, "s", s, "sample", sample)
    #
    #             #print("end decc")
    #
    #         else:  # can reach maximal speed in straight line
    #
    #             #print("can reach max vel")
    #             dist_constant_speed  = dist_total - dist_from_init_to_max_vel - dist_from_max_vel_to_final
    #             time_constant_speed  = dist_constant_speed / min_ax_vel
    #             trajectory_part_time = time_from_init_to_max_vel + time_constant_speed + time_from_max_to_final_vel
    #
    #             """
    #             print("time_constant_speed", time_constant_speed, "s")
    #             print("dist_constant_speed", dist_constant_speed, "m")
    #             print("trajectory_part_time", trajectory_part_time)
    #             """
    #             t = 0
    #             sample = 0
    #             while (sample + 1) * self.dT <= time_from_init_to_max_vel - sample_start_time:
    #
    #                 t = (sample + 1) * self.dT + sample_start_time
    #                 v = init_velocity + min_ax_acc * t
    #                 s = init_velocity * t + 0.5 * min_ax_acc * (t ** 2)
    #
    #                 pose_in_dist = poseInDistance(start, stop, s)
    #                 poses.append(pose_in_dist)
    #                 sample += 1
    #                 #print("t", t, "v", v, "s", s, "sample", sample)
    #
    #             #print("end acc")
    #
    #             while (sample + 1) * self.dT <= time_from_init_to_max_vel + time_constant_speed - sample_start_time:
    #
    #                 t      = (sample + 1) * self.dT + sample_start_time
    #                 t_part = t - time_from_init_to_max_vel
    #                 v      = min_ax_vel
    #                 s      = dist_from_init_to_max_vel + v * t_part
    #
    #                 pose_in_dist = poseInDistance(start, stop, s)
    #                 poses.append(pose_in_dist)
    #                 sample += 1
    #                 #print("t", t, "v", v, "s", s, "sample", sample)
    #
    #             #print("end const")
    #
    #             while (sample + 1) * self.dT <= time_from_init_to_max_vel + time_constant_speed + time_from_max_to_final_vel - sample_start_time:
    #
    #                 t      = (sample + 1) * self.dT + sample_start_time
    #                 t_part = t - (time_from_init_to_max_vel + time_constant_speed)
    #                 v      = min_ax_vel - min_ax_acc * t_part
    #                 s      = (dist_total - dist_from_max_vel_to_final) + min_ax_vel * t_part - 0.5 * min_ax_acc * (t_part ** 2)
    #
    #                 pose_in_dist = poseInDistance(start, stop, s)
    #                 poses.append(pose_in_dist)
    #                 sample += 1
    #                 #print("t", t, "v", v, "s", s, "sample", sample)
    #
    #             if final_velocity == 0 and poses[-1] != stop:
    #                 poses.append(stop)
    #                 #print("t last", "v", 0, "s", dist_total)
    #
    #         return poses, trajectory_part_time
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第11—13步：按方向利用各轴动力学，同步朝向，正确采样短段和末端。
    def sampleStraightSegmentWithStops(self, start, stop):
        # 用共同进度u∈[0,1]同步x/y/z/heading；p_i=p_i0+delta_i*u。
        # 因而u速度上限=min(v_i/|delta_i|)，加速度同理；零位移轴不施加限制。
        # 相比取各轴最小值，水平飞行不再被未参与运动的z轴上限拖慢。
        if self.dT <= 0:
            raise ValueError('采样间隔必须为正')
        initial = np.array(start.asList(), dtype=float)
        delta = np.array(stop.asList(), dtype=float) - initial
        delta[3] = wrapAngle(delta[3])
        velocities = np.array(list(self.max_velocity) + [getattr(self, 'max_heading_rate', 1.)])
        accelerations = np.array(list(self.max_acceleration) + [getattr(self, 'max_heading_acceleration', 1.)])
        if not np.all(np.isfinite([initial, delta, velocities, accelerations])) or np.any(velocities <= 0) or np.any(accelerations <= 0):
            raise ValueError('位姿必须有限，各维速度和加速度上限必须为正')
        moving = np.abs(delta) > 1e-12
        if not moving.any():
            return [Pose(*stop.asList())], self.dT
        speed = float(np.min(velocities[moving] / np.abs(delta[moving])))
        accel = float(np.min(accelerations[moving] / np.abs(delta[moving])))
        # 总进度长度为1：短段采用三角形，长段采用梯形速度曲线。
        peak = min(speed, math.sqrt(accel))
        ramp = peak / accel
        cruise = max(0., (1. - peak*peak/accel) / peak)
        duration = 2*ramp + cruise
        count = max(1, int(math.ceil(duration / self.dT)))
        # 把完整曲线均匀放慢到整数个dt，终点仍处于零速，无短尾段时间误差。
        result = []
        for k in range(1, count + 1):
            t = k * duration / count
            if t < ramp:
                progress = 0.5*accel*t*t
            elif t < ramp + cruise:
                progress = 0.5*peak*ramp + peak*(t-ramp)
            else:
                progress = 1. - 0.5*accel*(duration-t)**2
            values = initial + np.clip(progress, 0., 1.)*delta
            values[3] = wrapAngle(values[3])
            result.append(Pose(*values))
        result[-1] = Pose(*stop.asList())
        # 返回值不含起点，与原调用协议一致。
        return result, count * self.dT
    # #}

    # #{ sampleTrajectoryThroughWaypoints()
    # [v1.1 原代码保留 BEGIN] 第11—16步：可执行连续采样分支，失败时有记录地回退停车模式。
    #     def sampleTrajectoryThroughWaypoints(self, trajectory, with_stops, smooth_path, smoothing_la_dist, smoothing_sampling_step, velocity_limits, acceleration_limits):
    #         '''
    #         Samples trajectory such that it respects dynamic constraints
    #
    #         Parameters:
    #             trajectory (Trajectory): the trajectory which waypoints should be sampled
    #             with_stops (bool): zero velocity will is required in every waypoint
    #             smooth_path (bool): if True, path will be smoothed
    #             smoothing_la_dist (float): smoothing look-ahead distance in meters
    #             smoothing_sampling_dist (float): smoothing sampling distance in meters
    #             velocity limits (4x1 list of floats): velocity limits for (x, y, z, heading)
    #             acceleration limits (4x1 list of floats): acceleration limits for (x, y, z, heading)
    #
    #         Returns:
    #             trajectory (Trajectory): trajectory which samples respect dynamic constraints of the UAV
    #         '''
    #
    #         if with_stops:
    #
    #             # Interpolate heading between waypoints
    #             trajectory.waypoints = self.interpolateHeading(trajectory.waypoints)
    #
    #             # Iterate through sequential waypoint pairs
    #             for w_idx in range(1, len(trajectory.waypoints)):
    #                 pose_from = trajectory.waypoints[w_idx - 1]
    #                 pose_to   = trajectory.waypoints[w_idx]
    #
    #                 # Sample waypoint-to-waypoint line segments with stops at each start and end
    #                 poses, _ = self.sampleStraightSegmentWithStops(pose_from, pose_to)
    #
    #                 # Add starting pose
    #                 if w_idx == 1:
    #                     poses = [pose_from] + poses
    #
    #                 trajectory.setSegment(w_idx - 1, poses)
    #
    #         else:
    #
    #             # If path smoothing is required, smooth the path
    #             if smooth_path:
    #                 print('[SMOOTHING PATH]')
    #                 waypoints = self.getSmoothPath(trajectory.waypoints, smoothing_la_dist, smoothing_sampling_step)
    #
    #             # Otherwise, use the waypoints
    #             else:
    #                 waypoints = trajectory.waypoints
    #
    #             if not waypoints:
    #                 return None
    #
    #             # Interpolate heading between waypoints
    #             traj_hdg_interp = self.interpolateHeading(waypoints)
    #             # Parametrize trajectory
    #             toppra_trajectory = self.getParametrizedTrajectory(traj_hdg_interp, velocity_limits, acceleration_limits)
    #
    #             sampling_step = trajectory.dT
    #
    #             # STUDENTS TODO: Sample the path parametrization 'toppra_trajectory' (instance of TOPPRA library).
    #             raise NotImplementedError('[STUDENTS TODO] Trajectory sampling not finished. You have to implement it on your own.')
    #             # Tips:
    #             #  - check code examples for TOPPRA: https://hungpham2511.github.io/toppra/auto_examples/index.html
    #             #  - use 'toppra_trajectory' and the predefined sampling step 'sampling_step'
    #
    #             samples = [] # [STUDENTS TODO] Fill this variable with trajectory samples
    #
    #             # Convert to Trajectory class
    #             poses      = [Pose(q[0], q[1], q[2], q[3]) for q in samples]
    #             trajectory = self.posesToTrajectory(poses)
    #
    #         return trajectory
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第11—16步：可执行连续采样分支，失败时有记录地回退停车模式。
    def sampleTrajectoryThroughWaypoints(self, trajectory, with_stops, smooth_path, smoothing_la_dist, smoothing_sampling_step, velocity_limits, acceleration_limits):
        if self.dT <= 0 or not np.all(np.isfinite(velocity_limits)) or not np.all(np.isfinite(acceleration_limits)) or min(velocity_limits) <= 0 or min(acceleration_limits) <= 0:
            raise ValueError('dt、速度及加速度上限必须有限且为正')
        self.max_velocity, self.max_acceleration = tuple(velocity_limits[:3]), tuple(acceleration_limits[:3])
        self.max_heading_rate, self.max_heading_acceleration = velocity_limits[3], acceleration_limits[3]
        waypoints = self.interpolateHeading(trajectory.waypoints)
        # [v1.1 新增] 原始指定朝向点是巡检锚点，dt采样也必须足够接近它们。
        self._required_viewpoints = [wp for wp in trajectory.waypoints if wp.heading is not None]
        if len(waypoints) == 1:
            return self.posesToTrajectory([waypoints[0]])
        if not with_stops and len(waypoints) > 1:
            try:
                # 非平滑折线的尖角必须停车；只有启用样条平滑才尝试连续飞行。
                if not smooth_path:
                    raise ValueError('连续飞行需要平滑路径')
                parametrization = self.getParametrizedTrajectory(waypoints, velocity_limits, acceleration_limits)
                if parametrization is None or not self._splineIsSafe(smoothing_sampling_step):
                    raise ValueError('时间化不可行、缺少空间校验器或样条安全检查失败')
                poses = self._sampleParametrization(parametrization, velocity_limits, acceleration_limits)
                self.last_sampling_mode = 'continuous'
                result = self.posesToTrajectory(poses)
                result.sampling_mode = self.last_sampling_mode
                return result
            except (ValueError, RuntimeError) as error:
                # 优化失败必须可见，并回退到原有分段停车的几何路线。
                print('[v1.1] 连续轨迹回退到停车轨迹:', error)
        # 停车模式先验证整段几何；回退不意味着允许不安全的原始路径。
        validator = getattr(self, 'spatial_validator', None)
        if validator is not None and any(not validator(a.point.asList(), b.point.asList(), 0.) for a, b in zip(waypoints, waypoints[1:])):
            raise ValueError('原始停车路径也不安全，拒绝生成轨迹')
        trajectory = Trajectory(self.dT, waypoints)
        for index, (start, stop) in enumerate(zip(waypoints, waypoints[1:])):
            poses, _ = self.sampleStraightSegmentWithStops(start, stop)
            if index == 0:
                poses.insert(0, start)
            trajectory.setSegment(index, poses)
        self.last_sampling_mode = 'stops' if with_stops else 'fallback_stops'
        trajectory.sampling_mode = self.last_sampling_mode
        return trajectory
    # #}

    # #{ checkCollisionsOnHorizonAndBreak()
    def checkCollisionsOnHorizonAndBreak(self, trajectory, reference_trajectory, horizon, safety_distance):
        '''
        This function is not used in the code. This serves as an inspiration for students to implement their own collision avoidance algorithm.
        '''

        # TODO: there is still some edge-case deadlock

        total_delay     = 0

        ref_poses     = reference_trajectory.getPoses()
        ref_poses_len = len(ref_poses)

        traj_poses     = trajectory.getPoses()
        traj_poses_len = len(traj_poses)

        print('[checkCollisionsOnHorizonAndBreak] horizon count: {:d}'.format(horizon))

        s_idx = 0
        while s_idx < traj_poses_len:

            traj_poses_len = len(traj_poses)
            collision_idx  = None

            for i in range(s_idx, min(s_idx + horizon, traj_poses_len)):

                # Reference trajectory has ended
                if i == ref_poses_len:
                    s_idx = traj_poses_len
                    break

                # Check if trajectories at this index collide
                if distEuclidean(traj_poses[i].point, ref_poses[i].point) < safety_distance:
                    collision_idx = i
                    break

            # Go to next sample if no collision was found
            if collision_idx is None:
                s_idx += 1
                continue

            # Derive velocities
            traj_velocities = self.getDerivation(traj_poses)

            min_velocity, min_velocity_idx = float("inf"), 0

            # Find sample with minimal velocity
            for i in range(s_idx, collision_idx - 1):
                vel_norm = traj_velocities[i].point.norm()
                if vel_norm < min_velocity:
                    min_velocity, min_velocity_idx = vel_norm, i

            # Add extra samples to delay the trajectory
            i = 0
            while i == 0 or distEuclidean(traj_poses[collision_idx].point, ref_poses[collision_idx].point) < safety_distance:

                # TODO: slow down continually instead of as fast as possible

                # Slow down with maximal deceleration
                stopping_dist  = min(0.0, min_velocity * self.dT - 0.5 * min_ax_acc * (self.dT ** 2))

                # Compute the sample pose
                stopping_start = traj_poses[min_velocity_idx]
                if stopping_dist > 0.0:
                    stopping_end = traj_poses[min_velocity_idx + 1]
                    pose         = poseInDistance(stopping_start, stopping_end, stopping_dist)
                else:
                    pose = stopping_start

                # Add the sample pose
                traj_poses = traj_poses[:min_velocity_idx + i + 1] + [pose] + traj_poses[min_velocity_idx + i + 1:]
                total_delay += 1

                min_velocity = stopping_dist / self.dT
                i += 1

            # Go from the start again as new collisions may occur from the delays
            s_idx = 0

        # Convert to Trajectory class
        trajectory = self.posesToTrajectory(traj_poses)

        return trajectory, total_delay * self.dT
    # #}

    # #{ computeTimeParametrization()
    # [v1.1 原代码保留 BEGIN] 第14—16步：保留观察点的样条、TOPPRA时间化与空间和动力学复查。
    #     def computeTimeParametrization(self, waypoints, v_lims, a_lims):
    #         '''
    #         Computes time parametrization among the given waypoints.
    #
    #         Parameters:
    #             waypoints (list[Pose]): the waypoints to be parametrized
    #             v_lims (4x1 list of floats): velocity limits for (x, y, z, heading)
    #             a_lims (4x1 list of floats): acceleration limits for (x, y, z, heading)
    #
    #         Returns:
    #             trajectory (TOPPRA instance): time parametrization of the given waypoints
    #         '''
    #         def get_normalized_distances(waypoints):
    #             distances = [0]
    #             for i in range(1, len(waypoints)):
    #                 dist = distEuclidean(waypoints[i].point, waypoints[i-1].point)
    #                 distances.append(distances[-1] + dist)
    #             total_distance = distances[-1]
    #             return [d / total_distance for d in distances]
    #
    #         wp_lists = [wp.asList() for wp in waypoints]
    #         normalized_distances = get_normalized_distances(waypoints)
    #         path = ta.SplineInterpolator(normalized_distances, wp_lists)
    #         pc_vel     = constraint.JointVelocityConstraint(v_lims)
    #         pc_acc     = constraint.JointAccelerationConstraint(a_lims)
    #         instance   = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel", gridpt_max_err_threshold=1e-7)
    #         trajectory = instance.compute_trajectory()
    #
    #         return trajectory
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第14—16步：保留观察点的样条、TOPPRA时间化与空间和动力学复查。
    def computeTimeParametrization(self, waypoints, v_lims, a_lims):
        # 原地转向同样需要参数长度；使用四维弦长仅作为样条参数，
        # 最终速度与加速度仍分别以米制、角度制限制，并非把弧度当成米限速。
        values = np.array([wp.asList() for wp in waypoints], dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(values, axis=0), axis=1) > 1e-10]
        values = values[keep]
        if len(values) < 2:
            return None
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))]
        knots = distances / distances[-1]
        self._parametrization_path = ta.SplineInterpolator(knots, values)
        self._parametrization_knots = knots
        # 使用穿过所有原始位姿的样条，避免旧lookahead平滑跳过巡检观察点。
        instance = algo.TOPPRA([constraint.JointVelocityConstraint(v_lims),
                               constraint.JointAccelerationConstraint(a_lims)],
                              self._parametrization_path, parametrizer='ParametrizeConstAccel',
                              gridpt_max_err_threshold=1e-7)
        return instance.compute_trajectory(0., 0.)

    def _splineIsSafe(self, sampling_step):
        # 三次样条在每个结点区间内的二阶导是线性的，所以端点范数最大值
        # 给出该区间内曲率项的上界M。曲线到弦段偏差不超过M*ds²/8。
        # 将这一偏差加入安全距离验证，避免仅验证样本点漏掉弯道碰撞。
        validator = getattr(self, 'spatial_validator', None)
        if validator is None:
            return False
        path = self._parametrization_path
        for left, right in zip(self._parametrization_knots, self._parametrization_knots[1:]):
            chord = np.linalg.norm(path(right)[:3] - path(left)[:3])
            count = max(4, int(np.ceil(chord / max(sampling_step, 1e-3))))
            nodes = np.linspace(left, right, count+1)
            for a, b in zip(nodes, nodes[1:]):
                bound = max(np.linalg.norm(path(a, 2)[:3]), np.linalg.norm(path(b, 2)[:3]))
                margin = bound * (b-a)**2 / 8 + 1e-9
                if not validator(path(a)[:3], path(b)[:3], margin):
                    return False
        return True

    def _sampleParametrization(self, parametrization, velocity_limits, acceleration_limits):
        # TOPPRA本身处理前后向可达性约束，包括提前减速；这里补全dt采样。
        # 对导数和最终离散差分复查，必要时全局放慢，保持样条几何不变。
        duration = float(parametrization.duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('TOPPRA返回非法持续时间')
        v_lims, a_lims = np.array(velocity_limits), np.array(acceleration_limits)
        probe_times = np.linspace(0., duration, max(101, int(np.ceil(duration / min(self.dT/4, .02)))+1))
        v = np.abs(parametrization(probe_times, 1))
        a = np.abs(parametrization(probe_times, 2))
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
            raise ValueError('TOPPRA导数包含非有限值')
        stretch = max(1., float(np.max(v/v_lims)), float(np.sqrt(np.max(a/a_lims)))) * 1.001
        for _ in range(4):
            count = max(1, int(np.ceil(duration * stretch / self.dT)))
            samples = np.asarray(parametrization(np.linspace(0., duration, count+1)))
            if not np.all(np.isfinite(samples)):
                raise ValueError('TOPPRA样本包含非有限值')
            # 首尾补静止样本，只用于检查边界差分，不改变最终轨迹采样时间。
            padded = np.vstack([samples[0], samples, samples[-1]])
            velocities = np.diff(padded, axis=0) / self.dT
            accelerations = np.diff(velocities, axis=0) / self.dT
            ratio = max(float(np.max(np.abs(velocities)/v_lims)),
                        float(np.sqrt(np.max(np.abs(accelerations)/a_lims))))
            if ratio <= 1. + 1e-8:
                # 样条穿过观察点，并不保证固定dt样本足够接近观察点。
                # 同时检查位置和朝向；不满足时减速增加样本，仍失败则回退停车。
                position_tolerance = getattr(self, 'inspection_position_tolerance', .3)
                heading_tolerance = getattr(self, 'inspection_heading_tolerance', .2)
                covered = True
                for wp in getattr(self, '_required_viewpoints', []):
                    distances = np.linalg.norm(samples[:, :3] - wp.point.asArray(), axis=1)
                    angles = np.abs((samples[:, 3] - wp.heading + np.pi) % (2*np.pi) - np.pi)
                    if not np.any((distances <= position_tolerance) & (angles <= heading_tolerance)):
                        covered = False
                        break
                if not covered:
                    stretch = (count*self.dT/duration) * 1.5
                    continue
                # 回放与控制器连接相邻样本时，弦段也必须安全。
                if any(not self.spatial_validator(a[:3], b[:3], 0.) for a, b in zip(samples, samples[1:])):
                    raise ValueError('采样后的连接线不安全')
                return [Pose(q[0], q[1], q[2], wrapAngle(q[3])) for q in samples]
            stretch = (count*self.dT/duration) * ratio * 1.01
        raise ValueError('采样轨迹未通过动力学差分复查')
    # #} end of computeTimeParametrization()

    ## | -------- Functions: collision prevention/avoidance ------- |

    # # #{ resolveCollisions()
    def resolveCollisions(self, method, problem, trajectories, safety_distance):
        '''
        Postprocesses given trajectories such that there are no collisions.

        Parameters:
            method (string)
            problem (mrim_resources/InspectionProblem)
            trajectories (list[N]): list of possibly colliding trajectories

        Returns:
            trajectories (list[N]): list of non-colliding trajectories (hopefully)
            delayed_robots (list[N]): indices of delayed robots
            delays (list[N]): delay in seconds for each robot
        '''

        # # #{ Resolve collisions and resample trajectories

        print("[COLLISION AVOIDANCE] method: {:s}".format(method))
        delay_robot_idx, delay_t = None, 0.0
        delayed_robots, delays   = [], []

        ## |  [COLLISION AVOIDANCE METHOD #1]: Delay 2nd UAV by the length of the 1st UAV trajectory  |
        if method == 'delay_2nd_till_1st_UAV_finishes':

            delay_robot_idx = problem.number_of_robots - 1
            delay_t         = trajectories[0].getTime()

            # Delay trajectory of the second UAV at start by the length of the first-UAV trajectory
            trajectories[1].delayStart(delay_t)

        ## |  [COLLISION AVOIDANCE METHOD #2]: Delay UAV with shorter trajectory at start until there is no collision occurring  |
        elif method == 'delay_till_no_collisions_occur':

            raise NotImplementedError('[STUDENTS TODO] Collision prevention method \'delay_till_no_collisions_occur\' not finished. You have to finish it on your own.')
            # Tips:
            #  - you might select which trajectory it is better to delay
            #  - the smallest delay step is the sampling step stored in variable 'self.dT'

            delay_step = self.dT
            traj_times = [t.getTime() for t in trajectories]
            traj_lens  = [t.getLength() for t in trajectories]

            # Decide which UAV should be delayed
            # [STUDENTS TODO] CHANGE BELOW
            delay_robot_idx, nondelay_robot_idx = 0, 1

            # TIP: use function `self.trajectoriesCollide()` to check if two trajectories are in collision
            collision_flag, collision_idx = ...

            while collision_flag:

                # delay the shorter-trajectory UAV at the start point by sampling period
                delay_t += delay_step

                # [STUDENTS TODO] use function `trajectory.delayStart(X)` to delay a UAV at the start location by X seconds

                # keep checking if the robot trajectories collide
                collision_flag, _ = ...

        # # #}

        if delay_robot_idx is not None:
            delayed_robots, delays = [delay_robot_idx], [delay_t]

        return trajectories, delayed_robots, delays
    # # #}

    ## | ---------------- Functions: path smoothing --------------- |

    # #{ getSmoothPath()
    def getSmoothPath(self, waypoints, lookahead_distance, sampling_step):
        '''
        Smooths given waypoint path.

        Parameters:
            waypoints (list[Pose]): the waypoints to be parametrized
            lookahead_distance (float): smoothing look-ahead distance in meters
            sampling_step (float): smoothing sampling distance in meters

        Returns:
            path (list[Pose]): smoothed waypoint sequence
        '''

        path       = []
        curr_pose  = waypoints[0]
        path_index = 0

        path.append(curr_pose)

        # Main control loop
        while True:

            prev_path_index  = path_index
            goal, path_index = self.getLookaheadPoint(curr_pose, waypoints, lookahead_distance, path_index)

            if goal is None:
                return None

            if path_index == len(waypoints) - 1 and distEuclidean(waypoints[-1], curr_pose) < sampling_step:

                path[-1].heading = waypoints[path_index].heading # preserve last heading
                print("Goal reached. End point = ", curr_pose, "Waypoints goal = ", waypoints[-1])
                break

            # preserve assigned required heading
            if not prev_path_index == path_index:
                path[-1].heading = waypoints[prev_path_index].heading

            velocity = 1.0
            dt       = sampling_step / velocity

            curr_pose         = simulateStep(curr_pose, goal, velocity, dt)
            curr_pose.heading = None
            path.append(curr_pose)

        return path

    # #}

    ## | -------------------- Helper functions -------------------- |

    # # #{ getDerivation()
    def getDerivation(self, samples):
        '''
        Computes 1st order derivation of given samples of dimension N.

        Parameters:
            samples (list[Pose]): the samples to be derived

        Returns:
            derivatives (list[Pose]): smoothed waypoint sequence of size Nx1
        '''
        derivatives = []
        derivatives.append(Pose(0.0, 0.0, 0.0, 0.0))

        for k in range(1, len(samples)):
            dx       = (samples[k].point.x - samples[k-1].point.x) / self.dT
            dy       = (samples[k].point.y - samples[k-1].point.y) / self.dT
            dz       = (samples[k].point.z - samples[k-1].point.z) / self.dT
            dheading = angleDiff(samples[k-1].heading, samples[k].heading) / self.dT
            derivatives.append(Pose(dx, dy, dz, dheading))

        return derivatives
    # # #}

    # # #{ trajectoriesCollide()
    def trajectoriesCollide(self, traj_a, traj_b, safety_distance):
        '''
        Computes whether and possibly where two trajectories collide.

        Parameters:
            traj_a (Trajectory): first trajectory
            traj_b (Trajectory): second trajectory
            safety_distance (float): minimal distance between any two trajectory samples

        Returns:
            bool: collision flag
            int: collision index
        '''
        samples_a, samples_b = traj_a.getPoses(), traj_b.getPoses()
        min_len              = min([len(samples_a), len(samples_b)])
        for i in range(min_len):
            if distEuclidean(samples_a[i].point, samples_b[i].point) < safety_distance:
                return True, i
        return False, 0
    # # #}

    # # #{ computeCollisionSegmentsOfTwoTrajectories()
    def computeCollisionSegmentsOfTwoTrajectories(self, traj_A, traj_B, safety_distance):
        '''
        Computes collision segments between two trajectories.

        Parameters:
            traj_a (Trajectory): first trajectory
            traj_b (Trajectory): second trajectory
            safety_distance (float): minimal distance between any two trajectory samples

        Returns:
            segments (list): list of tuples (start_idx, end_idx) of colliding segments start and end indices
        '''

        segments = []

        # Compute which trajectory is shorter/longer
        trajs                   = [traj_A, traj_B]
        poses                   = [t.getPoses() for t in trajs]
        traj_times              = [t.getTime() for t in trajs]
        shorter_idx, longer_idx = np.argmin(traj_times), np.argmax(traj_times)

        # Compute hysteresis window (max stopping length)
        max_stop_time = max([self.max_velocity[i] / self.max_acceleration[i] for i in range(3)])
        hysteresis    = int(np.ceil(max_stop_time / self.dT))

        # Iterate over the collisions list
        col_prev                                 = pointCollidesWithPath(poses[shorter_idx][0].point, poses[longer_idx], safety_distance)
        longest_col_seg_sample_count             = 0
        safety_flag, first_col_idx, last_col_idx = True, 0, 0

        for i in range(1, len(poses[shorter_idx])):
            col_now = pointCollidesWithPath(poses[shorter_idx][i].point, poses[longer_idx], safety_distance)

            # collision->collision: do nothing
            if col_prev and col_now:
                col_prev = col_now
                continue

            # safe->collision: mark unsafe and store the index of first collision
            elif not col_prev and col_now:
                safety_flag   = False
                first_col_idx = i

            # collision->safe: store the index of last collision
            elif col_prev and not col_now:
                last_col_idx = i - 1

            # safe->safe: use hysteresis to mark the collision segment over
            elif not safety_flag and i - last_col_idx > hysteresis:
                safety_flag = True
                segments.append((first_col_idx, last_col_idx))

            col_prev = col_now

        return segments
    # # #}

    # #{ getLookaheadPoint()
    def getLookaheadPoint(self, pose, path, la_dist, path_index):
        '''
        Finds a point on the path at certain distance.

        Parameters:
            pose (Pose): the initial pose from which the looking starts from
            path (list[Pose]): the list of poses
            la_dist (float): distance from pose
            path_index (int): index of the pose in the path

        Returns:
            Pose: the lookahead point
            int: path_index of the found lookahead point
        '''
        goal_candidates = []

        while True:
            if path_index > 0:
                v_begin = path[path_index - 1]
                v_end   = path[path_index]
            else:
                v_begin = pose
                v_end   = path[path_index]
            # print("v begin and end = ", v_begin, v_end)

            # go to next segment if dist traveled is lower than lookahead dist and end of segment is closer than la dist
            dist_to_seg = segmentPointDist(v_begin.point, v_end.point, pose.point)
            dist_to_end = distEuclidean(pose.point, v_end.point)
            # print("dist to seg, dist_to_end = ", dist_to_seg, dist_to_end)

            if dist_to_end < la_dist:

                if path_index == len(path) - 1: # end of the path too close
                    goal_candidates.append(v_end)
                    break
                else:
                    path_index = min(path_index + 1, len(path) - 1)

            else:
                intersections   = lineSphereIntersections(pose.point, la_dist, v_begin.point, v_end.point)
                goal_candidates = []
                for i in range(len(intersections)):
                    intersection = intersections[i]
                    goal_candidates.append(Pose(intersection, pose.heading))
                # print("Goal candidates: ", goal_candidates)
                break

        if len(goal_candidates) == 0:
            goal_candidates.append(v_end)
            print("[ERROR] No intersection found, something bad happened. Consider reparametrization.")
            # return None, None
        elif len(goal_candidates) == 1 and distEuclidean(goal_candidates[0].point, v_end.point) > distEuclidean(pose.point, v_end.point):
            goal_candidates[0] = v_end

        dists = [distEuclidean(gc.point, v_end.point) for gc in goal_candidates]
        # print("Goal candidates dists: ", dists)
        return goal_candidates[np.argmin(dists)], path_index

    # #} end of getLookaheadPoint()

    # #{ unwrapHeadingInPath()
    def unwrapHeadingInPath(self, path):
        '''
        Unwraps heading in the entire path.

        Parameters:
            path (list[Pose])

        Returns:
            path_unwrapped (list[Pose]): list of 3D poses with each pose having positive
        '''
        path_unwrapped = [path[0]]

        for k in range(1, len(path)):
            prev_heading = path_unwrapped[-1].heading
            if abs(prev_heading - path[k].heading) < np.pi:
                path_unwrapped.append(path[k])
            else:
                heading_diff = angleDiff(prev_heading, path[k].heading)
                pose         = Pose(path[k].point, prev_heading + heading_diff)
                path_unwrapped.append(pose)

        return path_unwrapped
    # #} end of unwrapHeadingInPath()

    # #{ getParametrizedTrajectory()
    def getParametrizedTrajectory(self, waypoints, v_lims, a_lims):
        '''
        Finds time parametrization throughout the given waypoints.

        Parameters:
            waypoints (list[Pose]): the waypoints to be parametrized
            v_lims (4x1 list of floats): velocity limits for (x, y, z, heading)
            a_lims (4x1 list of floats): acceleration limits for (x, y, z, heading)

        Returns:
            trajectory (TOPPRA instance): time parametrization of the given waypoints
        '''

        print('[PARAMETRIZING TRAJECTORY]')
        unwrapped_heading_path = self.unwrapHeadingInPath(waypoints)
        trajectory             = self.computeTimeParametrization(unwrapped_heading_path, v_lims, a_lims)

        return trajectory
    # #} end of

    ## | ----------------- Getters and conversions ---------------- |

    # # #{ getLength()
    def getLength(self, poses):
        '''
        Finds length of trajectory poses.

        Parameters:
            poses (list[Pose]): the poses list

        Returns:
            length (float): length of the trajectory
        '''
        length = 0.0
        for i in range(1, len(poses)):
            length += distEuclidean(poses[i - 1].point, poses[i].point)
        return length
    # # #}

    # # #{ posesToTrajectory()
    def posesToTrajectory(self, poses):
        '''
        Converts poses to Trajectory.

        Parameters:
            poses (list): the poses list

        Returns:
            trajectory (Trajectory): 1-segment trajectory
        '''

        # convert to Trajectory class (hack: with 1 segment only)
        trajectory = Trajectory(self.dT, [poses[0], poses[-1]])
        trajectory.setSegment(0, poses)

        return trajectory
    # # #}

# # #}
