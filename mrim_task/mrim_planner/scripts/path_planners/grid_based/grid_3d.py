"""
Highly simplistic 3D grid for 3D path planners
@author: F. Nekovar
"""

import numpy as np
from scipy import ndimage

# # #{ class Grid3D
class Grid3D():

    occupied_cells = set()

    def __init__(self, idx_zero, dimensions, resolution_xyz, resolution_hdg=None):

        self.resolution_xyz = resolution_xyz
        self.dim            = dimensions
        self.idx_zero       = idx_zero
        self.array          = np.zeros(dimensions)

    # # #{ idxIsOccupied()
    # [v1.1 原代码保留 BEGIN] 第5、8步：越界视为不可通行。
    #     def idxIsOccupied(self, idx):
    #         try:
    #             return bool(self.array[idx])
    #         except IndexError:
    #             return False
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5、8步：越界视为不可通行。
    def idxIsOccupied(self, idx):
        # NumPy负索引会回绕到数组尾部，故必须显式拦截所有越界情况。
        if any(i < 0 or i >= self.dim[axis] for axis, i in enumerate(idx)):
            return True
        return bool(self.array[tuple(idx)])
    # # #}

    # # #{ xyzIsOccupied()
    # [v1.1 原代码保留 BEGIN] 第8步：修正米制位置到网格索引的重复偏移。
    #     def xyzIsOccupied(self, xyz):
    #         idx = self.metricToIndex(xyz)
    #         return self.idxIsOccupied(tuple(map(sum, zip(idx, self.idx_zero))))
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第8步：修正米制位置到网格索引的重复偏移。
    def xyzIsOccupied(self, xyz):
        # metricToIndex已经减去了世界原点，不能再把米制原点加到索引上。
        return self.idxIsOccupied(self.metricToIndex(xyz))
    # # #}

    # # #{ obstacleBetween()
    # [v1.1 原代码保留 BEGIN] 第5、7—8步：保守网格遍历与连续线段检查。
    #     def obstacleBetween(self, pt1, pt2):
    #
    #         (x1, y1, z1) = pt1
    #         (x2, y2, z2) = pt2
    #         ListOfPoints = []
    #         ListOfPoints.append(pt1)
    #         dx = abs(x2 - x1)
    #         dy = abs(y2 - y1)
    #         dz = abs(z2 - z1)
    #         if (x2 > x1):
    #             xs = 1
    #         else:
    #             xs = -1
    #         if (y2 > y1):
    #             ys = 1
    #         else:
    #             ys = -1
    #         if (z2 > z1):
    #             zs = 1
    #         else:
    #             zs = -1
    #
    #         # Driving axis is X-axis"
    #         if (dx >= dy and dx >= dz):
    #             p1 = 2 * dy - dx
    #             p2 = 2 * dz - dx
    #             while (x1 != x2):
    #                 x1 += xs
    #                 if (p1 >= 0):
    #                     y1 += ys
    #                     p1 -= 2 * dx
    #                 if (p2 >= 0):
    #                     z1 += zs
    #                     p2 -= 2 * dx
    #                 p1 += 2 * dy
    #                 p2 += 2 * dz
    #                 ListOfPoints.append((x1, y1, z1))
    #
    #         # Driving axis is Y-axis"
    #         elif (dy >= dx and dy >= dz):
    #             p1 = 2 * dx - dy
    #             p2 = 2 * dz - dy
    #             while (y1 != y2):
    #                 y1 += ys
    #                 if (p1 >= 0):
    #                     x1 += xs
    #                     p1 -= 2 * dy
    #                 if (p2 >= 0):
    #                     z1 += zs
    #                     p2 -= 2 * dy
    #                 p1 += 2 * dx
    #                 p2 += 2 * dz
    #                 ListOfPoints.append((x1, y1, z1))
    #
    #         # Driving axis is Z-axis"
    #         else:
    #             p1 = 2 * dy - dz
    #             p2 = 2 * dx - dz
    #             while (z1 != z2):
    #                 z1 += zs
    #                 if (p1 >= 0):
    #                     y1 += ys
    #                     p1 -= 2 * dz
    #                 if (p2 >= 0):
    #                     x1 += xs
    #                     p2 -= 2 * dz
    #                 p1 += 2 * dy
    #                 p2 += 2 * dx
    #                 ListOfPoints.append((x1, y1, z1))
    #
    #         for point in ListOfPoints:
    #             if self.idxIsOccupied(point):
    #                 return True
    #         return False
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第5、7—8步：保守网格遍历与连续线段检查。
    def obstacleBetween(self, pt1, pt2):
        # 采用三维supercover遍历：穿过格面、格棱或格角时，所有接触的格子都检查。
        # 原Bresenham只选一条中心格链，可能漏掉斜向切角接触的占用格。
        from itertools import combinations
        cell, target = np.array(pt1, dtype=int), np.array(pt2, dtype=int)
        delta = target - cell
        step = np.sign(delta).astype(int)
        dt = np.full(3, np.inf)
        np.divide(1.0, np.abs(delta), out=dt, where=delta != 0)
        crossing = dt / 2
        if self.idxIsOccupied(tuple(cell)):
            return True
        while not np.array_equal(cell, target):
            axes = np.flatnonzero(np.isclose(crossing, min(crossing), rtol=0, atol=1e-12))
            for count in range(1, len(axes) + 1):
                for subset in combinations(axes, count):
                    touched = cell.copy()
                    touched[list(subset)] += step[list(subset)]
                    if self.idxIsOccupied(tuple(touched)):
                        return True
            cell[axes] += step[axes]
            crossing[axes] += dt[axes]
        return not self.metricSegmentValid(self.indexToMetric(pt1), self.indexToMetric(pt2))

    def metricSegmentValid(self, start, end):
        # 连续空间复查使用原始点云，避免网格量化掩盖真实碰撞。
        from path_planners.geometry import segment_clear
        if getattr(self, 'spatial_validator', None) is not None:
            return self.spatial_validator(start, end, 0.)
        if hasattr(self, 'bounds'):
            from data_types import Point
            if not self.bounds.valid(Point(*start[:3])) or not self.bounds.valid(Point(*end[:3])):
                return False
        return segment_clear(start, end, self.obstacles_tree, self.safety_distance)
    # # #}

    # # #{ setObstacles()
    # [v1.1 原代码保留 BEGIN] 第8步：保守障碍膨胀，兼容现代NumPy。
    #     def setObstacles(self, obstacle_points, safety_distance):
    #         for opt in obstacle_points:
    #             idx = self.metricToIndex([opt.x, opt.y, opt.z])
    #             try:
    #                 self.array[idx] = 1
    #             except:
    #                 print("Obstacle Index out of bounds of Gridspace!")
    #
    #
    #         safety_steps = np.ceil(safety_distance / self.resolution_xyz).astype(np.int)
    #         x,y,z        = np.ogrid[-safety_steps:safety_steps+1, -safety_steps:safety_steps+1 , -safety_steps:safety_steps+1]
    #         mask         = x**2 + y**2 + z**2 <= safety_steps**2
    #         self.array   = ndimage.binary_dilation(self.array, mask)
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第8步：保守障碍膨胀，兼容现代NumPy。
    def setObstacles(self, obstacle_points, safety_distance):
        from scipy.spatial import KDTree
        self.safety_distance = safety_distance
        coordinates = np.asarray([[p.x, p.y, p.z] for p in obstacle_points], dtype=float).reshape(-1, 3)
        self.obstacles_tree = KDTree(coordinates)
        self.array.fill(0)
        # 增加半个体素对角线，覆盖障碍点量化到格心带来的位置误差。
        radius = safety_distance + np.sqrt(3) * self.resolution_xyz / 2
        steps = int(np.ceil(radius / self.resolution_xyz))
        x, y, z = np.ogrid[-steps:steps+1, -steps:steps+1, -steps:steps+1]
        mask = (x*x + y*y + z*z) * self.resolution_xyz**2 <= radius**2 + 1e-12
        # 允许位于网格外、但安全球影响网格的障碍参与膨胀，避免边界漏障碍。
        padded = np.zeros(tuple(d + 2 * steps for d in self.dim), dtype=bool)
        for p in coordinates:
            idx = tuple(i + steps for i in self.metricToIndex(p))
            if all(0 <= idx[a] < padded.shape[a] for a in range(3)):
                padded[idx] = True
        padded = ndimage.binary_dilation(padded, structure=mask)
        self.array = padded[tuple(slice(steps, steps + d) for d in self.dim)]
    # # #}

    # # #{ metricToIndex()
    # [v1.1 原代码保留 BEGIN] 第8步：移除np.int与隐式高度夹取。
    #     def metricToIndex(self, xyz):
    #         x, y, z = xyz[0], xyz[1], xyz[2]
    #         ix = np.round((x - self.idx_zero[0]) / self.resolution_xyz).astype(np.int)
    #         iy = np.round((y - self.idx_zero[1]) / self.resolution_xyz).astype(np.int)
    #         iz = np.round((z - self.idx_zero[2]) / self.resolution_xyz).astype(np.int)
    #         iz = 0 if iz < 0 else iz
    #         return (ix, iy, iz)
    # [v1.1 原代码保留 END]

    # [v1.1 新实现] 第8步：移除np.int与隐式高度夹取。
    def metricToIndex(self, xyz):
        # 不再把负高度索引夹到0；由边界检查明确拒绝非法状态。
        return tuple(int(np.rint((xyz[i] - self.idx_zero[i]) / self.resolution_xyz)) for i in range(3))
    # # #}

    # # #{ indexToMetric()
    def indexToMetric(self, idx):
        ix, iy, iz = idx[0], idx[1], idx[2]
        return (self.idx_zero[0] + ix * self.resolution_xyz,
                self.idx_zero[1] + iy * self.resolution_xyz,
                self.idx_zero[2] + iz * self.resolution_xyz,
                None)
    # # #}

# # #}
