import numpy as np


STOPS = np.array([0, .12, .28, .46, .64, .82, 1])
COLORS = np.array([[16, 0, 122], [0, 40, 216], [0, 201, 237], [66, 232, 117], [232, 242, 28], [255, 134, 17], [197, 0, 0]]) / 255


def risk_color(values):
    values = np.asarray(values)
    return np.stack([np.interp(values, STOPS, COLORS[:, channel]) for channel in range(3)], axis=-1)


def field_lengths(magnitudes, mode, scope):
    magnitudes = np.asarray(magnitudes, dtype=float)
    if not magnitudes.size:
        return magnitudes
    original = (6.5/magnitudes[0] if magnitudes[0] > 1e-10 else 0.) if mode == 'turning' else {'total': .4, 'residual': .58}[mode]
    scale = original if scope == 'local' else min(original, 10/max(1e-10, magnitudes.max()))
    return magnitudes*scale*1.15


def vertices(points, colors=(.3, .4, .5), normals=None, seen=None):
    points = np.asarray(points).reshape(-1, 3)
    result = np.zeros((len(points), 11), dtype=np.float32)
    result[:, :3] = points
    if normals is not None:
        result[:, 3:6] = normals
    result[:, 6:9] = colors
    result[:, 9:11] = -1 if seen is None else seen
    return result


def segments(points, colors=(.3, .4, .5)):
    points = np.asarray(points)
    pairs = np.stack([points[:-1], points[1:]], axis=1).reshape(-1, 3)
    if np.asarray(colors).ndim == 2:
        colors = np.stack([colors[:-1], colors[1:]], axis=1).reshape(-1, 3)
    return vertices(pairs, colors)


def arrows(origins, vectors, colors=(.3, .4, .8), lengths=10):
    origins, vectors = np.atleast_2d(origins), np.atleast_2d(vectors)
    direction = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    lengths = np.broadcast_to(np.asarray(lengths), (len(origins),))[:, None]
    ends = origins + direction * lengths
    side = np.cross(direction, [0, 0, 1])
    vertical = np.linalg.norm(side, axis=1) < .01
    side[vertical] = np.cross(direction[vertical], [0, 1, 0])
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    data = np.stack([origins, ends, ends, ends - direction*lengths*.24 + side*lengths*.12, ends, ends - direction*lengths*.24 - side*lengths*.12], axis=1).reshape(-1, 3)
    colors = np.broadcast_to(colors, (len(origins), 3)).repeat(6, axis=0)
    return vertices(data, colors)


def capsule():
    rings = []
    for axial, radius in [(-3-3*np.cos(angle), 3*np.sin(angle)) for angle in np.linspace(0, np.pi/2, 9)]:
        rings.append((axial, radius))
    rings += [(3+3*np.sin(angle), 3*np.cos(angle)) for angle in np.linspace(0, np.pi/2, 9)]
    angles = np.linspace(0, np.pi*2, 33)
    mesh = np.array([[[axial, radius*np.cos(angle), radius*np.sin(angle)] for angle in angles] for axial, radius in rings])
    triangles = []
    for ring in range(len(rings)-1):
        for side in range(32):
            triangles.extend([mesh[ring, side], mesh[ring+1, side], mesh[ring, side+1], mesh[ring, side+1], mesh[ring+1, side], mesh[ring+1, side+1]])
    points = np.array(triangles)
    normals = points.copy()
    normals[:, 0] -= np.clip(normals[:, 0], -3, 3)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    colors = np.tile([.94, .58, .12], (len(points), 1))
    colors[points[:, 0] > 4.3] = [.16, .24, .30]
    return vertices(points, colors, normals)


class Domains:
    def __init__(self, replay):
        self.replay = replay
        indices, previous = [0], 0
        for index in range(1, len(replay.centerline)-1):
            if replay.source['arc_s_mm'][index] - replay.source['arc_s_mm'][previous] >= 24:
                indices.append(index)
                previous = index
        indices.append(len(replay.centerline)-1)
        self.centers = replay.centerline[indices]
        normal = np.column_stack([replay.source['n'+axis][indices] for axis in 'xyz'])
        binormal = np.column_stack([replay.source['b'+axis][indices] for axis in 'xyz'])
        self.center_points = np.stack([self.centers, self.centers-8*normal, self.centers+8*normal, self.centers-8*binormal, self.centers+8*binormal], axis=1).reshape(-1, 3)
        self.bounds = np.asarray(replay.registry['field_world_bounds_mm'])
        self.whole_points = np.stack(np.meshgrid(*[np.linspace(*bounds, count) for bounds, count in zip(self.bounds, [9, 7, 13])], indexing='ij'), axis=-1).reshape(-1, 3)

    def points(self, frame, scope):
        if scope == 'local':
            return np.stack(np.meshgrid(*[np.linspace(-30, 30, 5)]*3, indexing='ij'), axis=-1).reshape(-1, 3) + frame.position
        return self.center_points if scope == 'centerline' else self.whole_points

    def bounds_for(self, frame, scope):
        if scope == 'local':
            return np.stack([frame.position-30, frame.position+30], axis=1)
        if scope == 'centerline':
            return np.stack([self.centers.min(axis=0)-18, self.centers.max(axis=0)+18], axis=1)
        return self.bounds

    def valid(self, points, frame, scope):
        bounds = self.bounds_for(frame, scope)
        valid = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
        if scope == 'centerline':
            valid &= np.min(np.sum((points[:, None, :] - self.centers[None, :, :])**2, axis=2), axis=1) <= 18**2
        body = (points-frame.position) @ frame.body_rotation
        axial = np.maximum(np.abs(body[:, 0])-3, 0)
        valid &= axial**2 + body[:, 1]**2 + body[:, 2]**2 > 9
        return valid

    def streamline_vertices(self, frame, scope, mode, color_range):
        points = self.points(frame, scope)
        seeds = points[::max(1, int(np.ceil(len(points)/48)))]
        bounds = self.bounds_for(frame, scope)
        step = 1.6 if scope == 'local' else max(1.6, np.min(bounds[:, 1]-bounds[:, 0])/40)
        output = []
        for sign in [-1, 1]:
            current = seeds.copy()
            active = self.valid(current, frame, scope)
            for iteration in range(120):
                direction = self.replay.field(frame, current, mode).copy()
                direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
                midpoint = current + direction*step*.5*sign
                direction = self.replay.field(frame, midpoint, mode).copy()
                direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
                following = current + direction*step*sign
                active &= self.valid(following, frame, scope)
                if not active.any():
                    break
                magnitudes = np.linalg.norm(self.replay.field(frame, current[active], mode), axis=1)
                pairs = np.stack([current[active], following[active]], axis=1).reshape(-1, 3)
                output.append(vertices(pairs, risk_color(magnitudes/color_range).repeat(2, axis=0)))
                current = following
        return np.concatenate(output) if output else vertices([])
