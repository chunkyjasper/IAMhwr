from itertools import cycle

import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams


# Data representation and preprocessing


# Coordinate Classes
class PointSet:
    def __init__(self, points=None, w=None, h=None, file_name=None, gt=None):
        self.points = points if points else []
        self.w = w
        self.h = h
        self.file_name = file_name
        self.gt = gt

    def add_point(self, point):
        self.points.append(point)
            
    def sample_size(self):
        return len(self.points)

    def get_stroke_group(self):
        strokes_n = set([p.stroke for p in self.points])
        return [[p for p in self.points if p.stroke == n] for n in strokes_n]

    def get_lines(self):
        lines = []
        strokes = self.get_stroke_group()
        for s in strokes:
            for i in range(len(s) - 1):
                if i == len(s) - 2:
                    lines.append(Line(s[-2], s[-1], eos=True))
                else:
                    lines.append(Line(s[i], s[i + 1]))
        return lines

    def total_length(self):
        return sum([l.length() for l in self.get_lines()])

    def mean(self):
        lines = self.get_lines()
        sum_px, sum_py, sum_l = (0, 0, 0)
        for l in lines:
            sum_px += l.proj_x()
            sum_py += l.proj_y()
            sum_l += l.length()

        return np.array([sum_px / sum_l, sum_py / sum_l])

    def sd_x(self, mean_x=None):
        if mean_x is None:
            mean_x = self.mean()[0]
        lines = self.get_lines()
        sum_vx = sum([l.var_x(mean_x) for l in lines])
        sum_l = sum([l.length() for l in lines])
        return (sum_vx / sum_l) ** .5

    def sd_y(self, mean_y=None):
        if mean_y is None:
            mean_y = self.mean()[1]
        lines = self.get_lines()
        sum_vy = sum([l.var_y(mean_y) for l in lines])
        sum_l = sum([l.length() for l in lines])

        return (sum_vy / sum_l) ** .5

    def normalize_points(self):
        mean_x, mean_y = self.mean()
        sd_y = self.sd_y(mean_y=mean_y)
        for p in self.points:
            p.normalize(mean_x, mean_y, sd_y)
        return self

    def range_x(self):
        xs = [p.x for p in self.points]
        return min(xs), max(xs)

    def range_y(self):
        ys = [p.y for p in self.points]
        return min(ys), max(ys)

    def down_sample_distance(self, d_th):
        strokes = self.get_stroke_group()
        ret = []
        removed = 0
        for s in strokes:
            for i in range(len(s)):
                if i == 0 or i == len(s) - 1 or Line(s[i], s[i - 1 - removed]).length() > d_th:
                    removed = 0
                    ret.append(s[i])
                else:
                    removed += 1
        self.points = ret
        return ret

    def down_sample_angle(self, cos_th):
        strokes = self.get_stroke_group()
        ret = []
        removed = 0
        for s in strokes:
            for i in range(len(s)):
                if i == 0 or i == len(s) - 1:
                    removed = 0
                    ret.append(s[i])
                else:
                    cs = Line(s[i], s[i - 1 - removed]).cosine_similarity(Line(s[i + 1], s[i]))
                    if cs < cos_th:
                        removed = 0
                        ret.append(s[i])
                    else:
                        removed += 1
        self.points = ret
        return ret

    def resample_distance(self, d):
        strokes = self.get_stroke_group()
        ret = []
        for s in strokes:
            if len(s) > 0:
                ret.append(s[0])

            for i in range(1, len(s)):
                line = Line(ret[-1], s[i])
                l = line.length()
                if l > d:
                    # interpolate
                    f = d / l
                    iteration = int(l / d)
                    for j in range(iteration):
                        ret.append(line.interpolate(f * (j + 1)))
                elif l == d:
                    ret.append(s[i])
                elif i < d:
                    continue
            ret.append(s[-1])
        self.points = ret
        return ret

    def slope_correction(self):
        def homogeneous_co(x, y):
            mat = np.ones((len(x), 3))
            mat[:, 0] = x
            mat[:, 1] = y
            return mat

        def translation_matrix(x, y):
            mat = np.identity(3)
            mat[2][0] = x
            mat[2][1] = y
            return mat

        def rotation_matrix(rad):
            mat = np.identity(3)
            mat[:2, :2] = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            return mat

        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        best_fit = np.poly1d(np.polyfit(x, y, 1))
        rad = np.arctan2(best_fit(x[-1]) - best_fit(x[0]), x[-1] - x[0])
        x_mid = (x[-1] + x[0]) / 2
        pivot_x, pivot_y = x_mid, best_fit(x_mid)
        co = homogeneous_co(x, y)
        co = np.matmul(co, translation_matrix(-pivot_x, -pivot_y))
        co = np.matmul(co, rotation_matrix(rad))
        co = np.matmul(co, translation_matrix(pivot_x, pivot_y))
        for i in range(len(co)):
            self.points[i].x, self.points[i].y = co[i][0], co[i][1]
        return self.points

    def up_sample_short_stroke(self, n):
        strokes = self.get_stroke_group()
        new_strokes = []
        for i in range(len(strokes)):
            s = strokes[i]
            l = len(s)
            if l < n:
                if l == 1:
                    new_strokes.append([s[0] for _ in range(n)])
                else:
                    pts = PointSet(s)
                    resample_d = pts.total_length()/n
                    new_strokes.append(pts.resample_distance(resample_d))
            else:
                new_strokes.append(s)
        self.points = [inner for outer in new_strokes for inner in outer]
        return self.points

    def generate_features(self, preprocess=None, pad=0, add_pad=0, fset=1):
        if preprocess:
            self.preprocess(**preprocess)
        lines = self.get_lines()
        features = []
        for l in lines:
            if fset == 1:
                features.append(l.get_features())
            elif fset == 2:
                features.append(l.get_features_2())
        features = np.array(features)
        dim = pad if pad else features.shape[0]
        dim += add_pad
        result = np.zeros((dim, features.shape[1]))
        result[:features.shape[0], :] = features
        return result

    def preprocess(self, down_d=0, down_cos=1, slope_correction=False,
                   normalize=False, resample_distance=0, up_sample=0):
        if slope_correction:
            self.slope_correction()
        if normalize:
            self.normalize_points()
        if down_cos < 1:
            self.down_sample_angle(down_cos)
        if resample_distance > 0:
            self.resample_distance(d=resample_distance)
        if up_sample > 0:
            self.up_sample_short_stroke(up_sample)
        if down_d > 0:
            self.down_sample_distance(down_d)

        return self.sample_size()

    def plot_samples(self):
        rcParams['figure.figsize'] = 10, 10
        groups = self.get_stroke_group()
        for g in groups:
            x = [p.x for p in g]
            y = [-p.y for p in g]
            plt.plot(x, y, '.', linewidth=2, color=(0, 0, 0))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_strokes(self):
        def colorline(x, y, color, linewidth=20):
            n_pts = len(x)
            colors = [(color + (1 - 0.9 * j / n_pts,)) for j in range(n_pts)]
            segments = make_segments(x, y)
            lc = mcoll.LineCollection(segments, colors=colors, linewidth=linewidth)
            ax = plt.gca()
            ax.add_collection(lc)
            return lc

        def make_segments(x, y):
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        rcParams['figure.figsize'] = 10, 10
        c = cycle([(1, 0, 0), (0, 0.6, 0), (0, 0, 0.7)])
        fig, ax = plt.subplots()
        xmin, xmax = self.range_x()
        xpad = (xmax - xmin) * 0.05
        xmin, xmax = xmin - xpad, xmax + xpad
        ymin, ymax = self.range_y()
        ypad = (ymax - ymin) * 0.05
        ymin, ymax = -ymax - ypad, -ymin + ypad
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        groups = self.get_stroke_group()
        for g in groups:
            color = next(c)
            x = [p.x for p in g]
            y = [-p.y for p in g]
            path = mpath.Path(np.column_stack([x, y]))
            verts = path.interpolated(steps=3).vertices
            x, y = verts[:, 0], verts[:, 1]
            colorline(x, y, color, linewidth=2)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_both(self):
        self.plot_samples()
        self.plot_strokes()

    def __repr__(self):
        return "<PointSet w={} h={} points_len={}>".format(self.w, self.h, len(self.points))


class Point:

    def __init__(self, stroke, time, x, y):
        self.stroke = stroke
        self.time = time
        self.x = x
        self.y = y

    def coordinates(self):
        return np.array([self.x, self.y])

    def normalize(self, mean_x, mean_y, sd):
        self.x = (self.x - mean_x) / sd
        self.y = (self.y - mean_y) / sd
        return self

    def displace(self, co_diff, t_diff):
        new_x, new_y = self.coordinates() + co_diff
        return Point(self.stroke, self.time + t_diff, new_x, new_y)

    def __repr__(self):
        return "<Point stroke={} time={} x={} y={}>".format(self.stroke,
                                                            self.time, self.x, self.y)


class Line:
    def __init__(self, p1, p2, eos=False):
        self.p1 = p1
        self.p2 = p2
        self.eos = eos

    def vec(self):
        return self.p2.coordinates() - self.p1.coordinates()

    def length(self):
        return np.linalg.norm(self.vec())

    def time_diff(self):
        return self.p2.time - self.p1.time

    def cosine_similarity(self, l):
        if self.length() * l.length() < 1e-5:
            return np.inf
        return np.dot(self.vec(), l.vec()) / (self.length() * l.length())

    def proj_x(self):
        return self.length() * (self.p1.x + self.p2.x) / 2

    def proj_y(self):
        return self.length() * (self.p1.y + self.p2.y) / 2

    def var_x(self, mean_x):
        return self.length() / 3 * ((self.p2.x - mean_x) ** 2 + (self.p1.x - mean_x) ** 2 +
                                    (self.p1.x - mean_x) * (self.p2.x - mean_x))

    def var_y(self, mean_y):
        return self.length() / 3 * ((self.p2.y - mean_y) ** 2 + (self.p1.y - mean_y) ** 2 +
                                    (self.p1.y - mean_y) * (self.p2.y - mean_y))

    def interpolate(self, pc):
        return self.p1.displace(self.vec() * pc, self.time_diff() * pc)

    # x, y, delta_x, delta_y, down, up
    def get_features(self):
        x_start = self.p1.x
        y_start = self.p1.y
        delta_x, delta_y = self.vec()
        down = not self.eos
        up = self.eos
        return np.array([x_start, y_start, delta_x, delta_y, down, up])

    # x, y, normalized direction, length, down, up
    def get_features_2(self):
        x_start = self.p1.x
        y_start = self.p1.y
        length = self.length()
        direction_x, direction_y = self.vec() / length
        down = not self.eos
        up = self.eos
        return np.array([x_start, y_start, direction_x, direction_y, length, down, up])

    def __repr__(self):
        return "<Line\n" + self.p1.__repr__() + '\n' + self.p2.__repr__() + '\n>'
