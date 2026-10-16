import csv
import json
import time

import numpy as np
import vtk
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from camera_model import ROOT, SETTINGS
from evaluation_core import CoverageAccumulator, evaluation_methods, evaluation_times
from ray_check import mesh_from_geometry


def evaluate():
    geometry = json.loads((ROOT / 'geometry.json').read_text(encoding='utf-8'))
    trajectories = json.loads((ROOT / 'camera_trajectories.json').read_text(encoding='utf-8'))
    points = np.asarray(geometry['points'])
    triangles = points[np.asarray(geometry['faces'])]
    centers = triangles.mean(axis=1)
    normals = np.asarray(geometry['normals'])
    areas = np.asarray(geometry['areas_mm2'])
    mesh = mesh_from_geometry(geometry)
    ids = np.arange(1, len(areas) + 1, dtype=np.uint32)
    colors = np.column_stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255]).astype(np.uint8)
    scalars = numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName('face_id')
    mesh.GetCellData().SetScalars(scalars)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().BackfaceCullingOff()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(SETTINGS['evaluation_width'], SETTINGS['evaluation_height'])
    window.AddRenderer(renderer)
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    camera = renderer.GetActiveCamera()
    camera.SetViewAngle(SETTINGS['fov_vertical_deg'])
    camera.SetClippingRange(SETTINGS['near_mm'], SETTINGS['far_mm'])
    results = {}
    started = time.time()
    for method in evaluation_methods():
        track = trajectories[method]
        matrices = Rotation.from_quat(track['quaternion']).as_matrix()
        times = np.asarray(track['time_s'])
        eval_times = evaluation_times(times[-1], trajectories['original_duration_s'], SETTINGS['evaluation_step_s'])
        indices = np.minimum(np.searchsorted(times, eval_times), len(times) - 1)
        counter = CoverageAccumulator(areas, SETTINGS['minimum_dwell_s'], SETTINGS['repeat_gap_s'])
        rows = []
        for number, (at_time, index) in enumerate(zip(eval_times, indices)):
            origin = np.asarray(track['position'][index])
            frame = matrices[index]
            camera.SetPosition(*origin)
            camera.SetFocalPoint(*(origin + frame[:, 0]))
            camera.SetViewUp(*frame[:, 2])
            window.Render()
            capture.Modified()
            capture.Update()
            image = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).astype(np.int32)
            face_ids = image[:, 0] + (image[:, 1] << 8) + (image[:, 2] << 16)
            counts = np.bincount(face_ids, minlength=len(areas) + 1)
            if len(counts) > len(areas) + 1:
                raise ValueError('Unexpected face colors: disable antialiasing/lighting')
            delta = centers - origin
            distance = np.linalg.norm(delta, axis=1)
            incidence = np.abs(np.einsum('ij,ij->i', normals, delta)) / np.maximum(distance, 1e-8)
            valid = ((distance >= SETTINGS['min_observation_distance_mm']) &
                     (distance <= SETTINGS['max_observation_distance_mm']) &
                     (incidence >= np.cos(np.deg2rad(SETTINGS['max_incidence_deg']))) &
                     (counts[1:] >= SETTINGS['min_projected_pixels']))
            duration = at_time - eval_times[number - 1] if number else 0.
            row = counter.update(np.flatnonzero(valid), float(at_time), float(duration))
            effective = float(100 * counts[1:][valid].sum() / len(face_ids))
            row.update({'time_s': float(at_time), 'arc_s_mm': float(track['arc_s_mm'][index]),
                        'effective_pct': effective, 'completion_pct': float(100 * track['arc_s_mm'][index] / track['arc_s_mm'][-1])})
            rows.append(row)
            if number % 150 == 0:
                print(method, number, '/', len(indices), 'coverage', round(row['coverage_pct'], 3), flush=True)
        effective_integral = np.trapezoid([row['effective_pct'] for row in rows], eval_times)
        integral_curve = np.r_[0, np.cumsum(np.diff(eval_times) * (np.asarray([row['effective_pct'] for row in rows[1:]]) + np.asarray([row['effective_pct'] for row in rows[:-1]])) / 2)]
        for index, row in enumerate(rows):
            row['effective_time_mean_pct'] = float(integral_curve[index] / eval_times[index]) if eval_times[index] > 0 else row['effective_pct']
        budget_index = min(np.searchsorted(eval_times, trajectories['original_duration_s'], side='right') - 1, len(rows) - 1)
        results[method] = {'rows': rows, 'first_seen_s': counter.first_time.tolist(), 'repeat_seen_s': counter.repeat_time.tolist(),
                           'final': {**rows[-1], 'effective_time_mean_pct': float(effective_integral / times[-1])},
                           'common_budget': rows[budget_index]}
        with (ROOT / f'evaluation_{method}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    window.Finalize()
    output = {'methods': results, 'settings': SETTINGS, 'target_area_mm2': float(areas.sum()),
              'target_triangle_count': len(areas), 'budget_s': trajectories['original_duration_s'],
              'elapsed_compute_s': time.time() - started,
              'notice': 'Shared decimated entire lumen surface; raster cell visibility; centroid distance/incidence approximation; simulated geometry quality only; no clinical texture validation'}
    (ROOT / 'evaluation.json').write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    summary = {method: {'common_budget': result['common_budget'], 'full_path': result['final']} for method, result in results.items()}
    (ROOT / 'evaluation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'seconds': output['elapsed_compute_s'], 'results': summary}))


if __name__ == '__main__':
    evaluate()
