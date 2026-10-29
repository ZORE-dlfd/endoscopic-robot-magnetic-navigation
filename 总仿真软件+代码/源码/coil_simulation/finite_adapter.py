from contextlib import contextmanager

import numpy as np


class SourceAdapter:
    def __init__(self, array):
        self.array = array
        self.evaluations = 0
        self.peak_current_a = 0.0
        self.peak_saturation_fraction = 0.0
        self.peak_field_error_t = 0.0
        self.peak_force_error_n = 0.0
        self.max_anchor_error_m = 0.0

    def responses(self, offsets, axes, unused_point_moment):
        offsets = np.asarray(offsets)
        if offsets.shape != (1, 8, 3) or not np.allclose(axes[0], self.array.axes):
            raise ValueError("Unexpected magnetic source geometry")
        points = self.array.offsets - offsets[0]
        if not np.allclose(points, points[0], atol=1e-10):
            raise ValueError("Source layout is not a rigid translated array")
        self.max_anchor_error_m = max(self.max_anchor_error_m, float(np.linalg.norm(points[0])))
        fields, gradients = self.array.responses(points[0])
        return fields[None], gradients[None]

    @contextmanager
    def installed(self, simulation):
        previous_responses = simulation.coil_responses
        previous_layout = simulation.default_array_layout
        previous_evaluate = simulation.evaluate

        def checked_evaluate(*args, **kwargs):
            result = previous_evaluate(*args, **kwargs)
            fraction = self.array.validate_currents(result["currents"])
            if not np.isfinite(result["derivative"]).all():
                raise ValueError("Nonfinite dynamics")
            self.evaluations += 1
            self.peak_current_a = max(self.peak_current_a, float(np.max(np.abs(result["currents"]))))
            self.peak_saturation_fraction = max(self.peak_saturation_fraction, fraction)
            self.peak_field_error_t = max(self.peak_field_error_t, float(result["field_error"]))
            self.peak_force_error_n = max(self.peak_force_error_n, float(result["force_error"]))
            return result

        simulation.coil_responses = self.responses
        simulation.default_array_layout = lambda: (self.array.offsets.copy(), self.array.axes.copy())
        simulation.evaluate = checked_evaluate
        try:
            yield self
        finally:
            simulation.coil_responses = previous_responses
            simulation.default_array_layout = previous_layout
            simulation.evaluate = previous_evaluate

    def report(self):
        return {name: value for name, value in vars(self).items() if name != "array"}
