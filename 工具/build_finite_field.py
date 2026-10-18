from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'源码'))

import numpy as np
from numba import njit, prange
from coil_simulation.finite_sources import (CoilDesign, _core_nodes, _winding_nodes,
                                           _integrate_core, _integrate_winding)


@njit(parallel=True)
def sample_table(radial, axial, core_nodes, core_weights, winding_nodes, winding_elements):
    result = np.zeros((len(radial), len(axial), 2, 5))
    for radial_index in prange(len(radial)):
        for axial_index in range(len(axial)):
            if radial[radial_index] <= .028+1e-12 and axial[axial_index] <= .026+1e-12:
                continue
            point = np.array([radial[radial_index], 0., axial[axial_index]])
            core_field, core_gradient = _integrate_core(point, core_nodes, core_weights)
            winding_field, winding_gradient = _integrate_winding(point, winding_nodes, winding_elements)
            for source_kind in range(2):
                field = winding_field if source_kind == 0 else core_field
                gradient = winding_gradient if source_kind == 0 else core_gradient
                result[radial_index, axial_index, source_kind] = np.array([
                    field[0], field[2], gradient[0, 0], gradient[0, 2], gradient[2, 2]])
    return result


def build(destination):
    started = time.perf_counter()
    design = CoilDesign(turns=1200)
    grid = np.unique(np.r_[np.arange(0., .1501, .0005), np.arange(.152, .502, .002),
                           np.arange(.51, 3.001, .01)])
    core_nodes, core_weights = _core_nodes(design)
    winding_nodes, winding_elements = _winding_nodes(design)
    responses = sample_table(grid, grid, core_nodes, core_weights, winding_nodes, winding_elements)
    np.savez_compressed(destination, radial_m=grid, axial_m=grid, responses=responses)
    print(f'Finite-source display table: {responses.shape}, {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    build(ROOT/'数据'/'clon001'/'finite_field.npz')
