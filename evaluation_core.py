import numpy as np


def evaluation_methods():
    return [mode + '_' + camera for mode in ['fixed', 'slowdown', 'wait', 'coordinated']
            for camera in ['endoscopy', 'new']]


def evaluation_times(duration_s, budget_s, step_s):
    landmarks = [duration_s]
    if 0.0 < budget_s < duration_s:
        landmarks.append(budget_s)
    return np.unique(np.r_[np.arange(0.0, duration_s, step_s), landmarks])


class CoverageAccumulator:
    def __init__(self, areas, dwell=.4, gap=.5):
        self.areas = np.asarray(areas, dtype=float)
        self.dwell = dwell
        self.gap = gap
        self.last_seen = np.full(len(areas), -np.inf)
        self.segment_dwell = np.zeros(len(areas))
        self.segment_counted = np.zeros(len(areas), dtype=bool)
        self.count = np.zeros(len(areas), dtype=int)
        self.first_time = np.full(len(areas), -1.)
        self.repeat_time = np.full(len(areas), -1.)

    def update(self, visible, time, duration):
        visible = np.unique(np.asarray(visible, dtype=int))
        restarted = visible[time - self.last_seen[visible] > self.gap + 1e-9]
        self.segment_dwell[restarted] = 0
        self.segment_counted[restarted] = False
        self.last_seen[visible] = time
        self.segment_dwell[visible] += duration
        qualified = visible[(self.segment_dwell[visible] >= self.dwell - 1e-9) & ~self.segment_counted[visible]]
        new = qualified[self.count[qualified] == 0]
        self.first_time[new] = time
        repeated_new = qualified[(self.count[qualified] == 1) & (self.repeat_time[qualified] < 0)]
        self.repeat_time[repeated_new] = time
        self.count[qualified] += 1
        self.segment_counted[qualified] = True
        total = self.areas.sum()
        covered = self.areas[self.count > 0].sum()
        observed = self.areas @ self.count
        repeated = self.areas @ np.maximum(self.count - 1, 0)
        return {'coverage_pct': float(100 * covered / total), 'unseen_pct': float(100 * (1 - covered / total)),
                'repeat_pct': float(100 * repeated / observed) if observed > 0 else None}
