import copy

import numpy as np


class Coverage:
    def __init__(self, areas, insertion_covered, dwell, gap):
        self.areas = np.array(areas, dtype=float, copy=True)
        self.insertion_covered = np.array(insertion_covered, dtype=bool, copy=True)
        if self.areas.ndim != 1 or self.areas.shape != self.insertion_covered.shape or not np.isfinite(self.areas).all() or np.any(self.areas < 0):
            raise ValueError('Coverage requires matching finite nonnegative face areas')
        self.dwell = float(dwell)
        self.gap = float(gap)
        self.last_seen = np.full(len(self.areas), -np.inf)
        self.dwell_time = np.zeros(len(self.areas))
        self.first_seen = np.full(len(self.areas), -1.)
        self.previous_visible = np.empty(0, dtype=int)

    def observe(self, visible, time, duration):
        if duration < 0 or not np.isfinite(time):
            raise ValueError('Observation time/duration must be valid')
        visible = np.unique(np.asarray(visible, dtype=int))
        if np.any(visible < 0) or np.any(visible >= len(self.areas)):
            raise ValueError('Face ID is out of range')
        restarted = visible[time - self.last_seen[visible] > self.gap + 1e-9]
        self.dwell_time[restarted] = 0.
        eligible = np.intersect1d(visible, self.previous_visible, assume_unique=True)
        self.dwell_time[eligible] += duration
        qualified = eligible[(self.dwell_time[eligible] >= self.dwell - 1e-9) & (self.first_seen[eligible] < 0)]
        self.first_seen[qualified] = time
        self.last_seen[visible] = time
        self.previous_visible = visible

    def clone(self):
        return copy.deepcopy(self)

    @property
    def target_mask(self):
        return ~self.insertion_covered & (self.first_seen < 0)

    @property
    def added_area(self):
        return float(self.areas[~self.insertion_covered & (self.first_seen >= 0)].sum())

    @property
    def total_area(self):
        return float(self.areas.sum())


def choose_near_best(gains, costs, tolerance):
    gains, costs = np.asarray(gains, dtype=float), np.asarray(costs, dtype=float)
    if gains.shape != costs.shape or gains.ndim != 1 or gains.size == 0 or not np.isfinite(gains).all() or not np.isfinite(costs).all():
        raise ValueError('Gains and costs must be matching finite nonempty vectors')
    if not 0 <= tolerance < 1:
        raise ValueError('Tolerance must be in [0,1)')
    admitted = np.flatnonzero(gains >= float(gains.max()) * (1 - tolerance) - 1e-9)
    return int(admitted[np.argmin(costs[admitted])])
