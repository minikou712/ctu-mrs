"""[v1.1 新增] RRT、A* 和轨迹复查共用的三维线段几何检查。"""

import numpy as np


def segment_clear(start, end, tree, clearance):
    """检查整条闭线段与障碍点的距离，等于安全距离也视为不安全。

    先查询以线段中点为球心、半长加安全距离为半径的候选障碍；
    任何可能碰撞的障碍都在此球内。再投影到闭线段计算精确距离，
    避免固定步长采样漏掉采样点之间的障碍。该保证针对输入点云。
    """
    start, end = np.asarray(start[:3], dtype=float), np.asarray(end[:3], dtype=float)
    if clearance < 0 or not np.isfinite(clearance):
        raise ValueError('安全距离必须有限且非负')
    if not np.all(np.isfinite([start, end])):
        return False
    delta = end - start
    length = float(np.linalg.norm(delta))
    indices = tree.query_ball_point((start + end) / 2, length / 2 + clearance + 1e-10)
    if not indices:
        return True
    offsets = tree.data[indices] - start
    fraction = np.clip(offsets.dot(delta) / length**2, 0, 1) if length > 1e-12 else np.zeros(len(indices))
    distances = np.linalg.norm(offsets - fraction[:, None] * delta, axis=1)
    return bool(np.all(distances > clearance + 1e-10))


class EnvironmentValidator:
    """[v1.1 新增] 原始点云、水平多边形与高度的共同检查器。

    水平线段不能接触或穿越多边形边界；曲线复查时margin还会收缩
    水平和高度允许区。支持凹多边形，不增加Shapely等第三方依赖。
    """

    def __init__(self, problem, clearance):
        from scipy.spatial import KDTree
        self.tree = KDTree(np.array([[p.x, p.y, p.z] for p in problem.obstacle_points], dtype=float).reshape(-1, 3))
        polygon = np.array([[p.x, p.y] for p in problem.safety_area], dtype=float)
        if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
            raise ValueError('水平安全区至少需要三个有限顶点')
        area = sum(self._cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))
        if abs(area) < 1e-10:
            raise ValueError('水平安全区面积必须大于零')
        self.edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
        # 拒绝自交或相邻重复顶点，避免对非法安全区给出模糊结论。
        for i, (a, b) in enumerate(self.edges):
            if np.linalg.norm(b-a) < 1e-10:
                raise ValueError('水平安全区不能含重复相邻顶点')
            for j, (c, d) in enumerate(self.edges):
                if j > i+1 and not (i == 0 and j == len(self.edges)-1):
                    if self._segment_distance(a, b, c, d) < 1e-10:
                        raise ValueError('水平安全区不能自交')
        self.clearance = clearance
        self.min_height, self.max_height = problem.min_height, problem.max_height

    @staticmethod
    def _cross(a, b):
        return float(a[0]*b[1] - a[1]*b[0])

    @staticmethod
    def _point_distance(p, a, b):
        delta = b-a
        fraction = np.clip(np.dot(p-a, delta) / np.dot(delta, delta), 0, 1) if np.dot(delta, delta) > 1e-20 else 0
        return float(np.linalg.norm(p-a-fraction*delta))

    @classmethod
    def _segment_distance(cls, a, b, c, d):
        first = cls._cross(b-a, c-a) * cls._cross(b-a, d-a)
        second = cls._cross(d-c, a-c) * cls._cross(d-c, b-c)
        # 严格相交；共线或端点接触由下面的距离处理。
        if first < 0 and second < 0:
            return 0.
        return min(cls._point_distance(a, c, d), cls._point_distance(b, c, d),
                   cls._point_distance(c, a, b), cls._point_distance(d, a, b))

    def _inside(self, p):
        inside = False
        for a, b in self.edges:
            if (a[1] > p[1]) != (b[1] > p[1]):
                crossing_x = a[0] + (p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if p[0] < crossing_x:
                    inside = not inside
        return inside

    def __call__(self, start, end, margin=0.):
        a, b = np.array(start[:3], dtype=float), np.array(end[:3], dtype=float)
        if not np.all(np.isfinite([a, b])) or margin < 0 or not np.isfinite(margin):
            return False
        if min(a[2], b[2])-margin < self.min_height or max(a[2], b[2])+margin > self.max_height:
            return False
        if not self._inside(a[:2]) or not self._inside(b[:2]):
            return False
        if any(self._segment_distance(a[:2], b[:2], c, d) <= margin + 1e-10 for c, d in self.edges):
            return False
        return segment_clear(a, b, self.tree, self.clearance + margin)
