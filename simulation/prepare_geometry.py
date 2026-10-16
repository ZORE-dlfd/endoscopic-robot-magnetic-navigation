import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import slicer
import slicer.util
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'Magnetic 1 - Field scopes'
ANATOMY = ROOT.parent.parent / '安全地图_路径规划' / '结果_colon_001_含曲率' / 'colon_model.stl'
ENDOSCOPY = Path(r'D:\3dslicer\3D Slicer 5.10.0\lib\Slicer-5.10\qt-scripted-modules\Endoscopy.py')


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def prepare():
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(ANATOMY))
    reader.Update()
    original = reader.GetOutput()
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(original)
    decimator.SetTargetReduction(.82)
    decimator.VolumePreservationOn()
    decimator.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(decimator.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    mesh = normals.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    faces = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    cell_normals = vtk_to_numpy(mesh.GetCellData().GetNormals())
    triangles = points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    valid = areas > 1e-8
    faces, areas, cell_normals = faces[valid], areas[valid], cell_normals[valid]
    geometry = {'points': np.round(points, 5).tolist(), 'faces': faces.tolist(), 'areas_mm2': areas.tolist(),
                'normals': cell_normals.tolist(), 'bounds': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                'original_triangles': original.GetNumberOfCells(), 'evaluation_triangles': len(faces),
                'surface_notice': 'Colon001 inner-wall mesh, decimated shared evaluation surface; no clinical texture',
                'coordinate_transform': 'none; raw STL agrees with source trajectory coordinate bounds'}
    source = np.genfromtxt(BASE / 'source_trajectory_state.csv', delimiter=',', names=True)
    source_points = np.column_stack([source[axis + '_mm'] for axis in 'xyz'])
    if not np.all((source_points.min(axis=0) >= points.min(axis=0) - 2) & (source_points.max(axis=0) <= points.max(axis=0) + 2)):
        raise ValueError('Anatomy/path bounds mismatch')
    tree = ast.parse(ENDOSCOPY.read_text(encoding='utf-8'))
    logic = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EndoscopyLogic')
    compiled = ast.Module(body=[logic], type_ignores=[])
    namespace = {'np': np, 'vtk': vtk, 'slicer': slicer, 'json': json}
    exec(compile(compiled, str(ENDOSCOPY), 'exec'), namespace)
    EndoscopyLogic = namespace['EndoscopyLogic']
    curve = slicer.vtkMRMLMarkupsCurveNode()
    curve.SetCurveTypeToLinear()
    for point in source_points:
        curve.AddControlPointWorld(*point)
    baseline = EndoscopyLogic(dl=.5)
    baseline.setControlPointsByResamplingAndInterpolationFromInputCurve(curve)
    resampled = baseline.resampledCurve
    count = resampled.GetNumberOfControlPoints()
    quaternions = []
    arc = []
    for index in range(count):
        orientation = np.zeros(4)
        resampled.GetNthControlPointOrientation(index, orientation)
        matrix = EndoscopyLogic.orientationToMatrix3x3(orientation)
        internal = np.column_stack([matrix[:, 2], matrix[:, 0], matrix[:, 1]])
        quaternion = vtk.vtkQuaterniond()
        quaternion.FromMatrix3x3(internal)
        values = np.zeros(4)
        quaternion.Get(values)
        quaternions.append([float(values[1]), float(values[2]), float(values[3]), float(values[0])])
        arc.append(float(EndoscopyLogic.distanceAlongCurveOfNthControlPoint(resampled, index)))
    write_json('geometry.json', geometry)
    write_json('endoscopy_baseline.json', {'arc_s_mm': arc, 'quaternion': quaternions,
               'source': str(ENDOSCOPY), 'sha256': hashlib.sha256(ENDOSCOPY.read_bytes()).hexdigest(),
               'settings': 'installed EndoscopyLogic dl=0.5mm, linear input path, no user camera keyframes; equal FOV in evaluator'})
    manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in BASE.iterdir() if path.is_file()}
    manifest[str(ANATOMY)] = hashlib.sha256(ANATOMY.read_bytes()).hexdigest()
    write_json('baseline_readonly_manifest.json', manifest)
    print(json.dumps({'triangles': len(faces), 'endoscopy_samples': count, 'area_mm2': float(areas.sum())}))


if __name__ == '__main__':
    prepare()
