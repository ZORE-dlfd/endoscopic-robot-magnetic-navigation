import json
from pathlib import Path

import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

ROOT = Path(__file__).resolve().parent


def mesh_from_geometry(geometry):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(geometry['points'], dtype=float), deep=True))
    faces = np.asarray(geometry['faces'], dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack([np.full(len(faces), 3), faces]).ravel(), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def check():
    geometry = json.loads((ROOT / 'geometry.json').read_text(encoding='utf-8'))
    queries = json.loads((ROOT / 'ray_queries.json').read_text(encoding='utf-8'))
    mesh = mesh_from_geometry(geometry)
    tree = vtk.vtkStaticCellLocator()
    tree.SetDataSet(mesh)
    tree.BuildLocator()
    results = []
    for origin, target in queries:
        origin, target = np.array(origin), np.array(target)
        direction = target - origin
        distance = np.linalg.norm(direction)
        origin += direction / max(distance, 1e-12) * .25
        hit_t, subid = vtk.mutable(0.), vtk.mutable(0)
        hit = np.zeros(3)
        pcoords = np.zeros(3)
        cellid = vtk.mutable(0)
        result = tree.IntersectWithLine(origin, target, .001, hit_t, hit, pcoords, subid, cellid)
        results.append(bool(result and float(hit_t) < .98))
    (ROOT / 'ray_results.json').write_text(json.dumps(results), encoding='utf-8')
    print('occlusion queries', len(results), 'blocked', sum(results))


if __name__ == '__main__':
    check()
