(function (root) {
  const axes = ['x', 'y', 'z'];
  const colorscale = [[0, '#10007a'], [0.12, '#0028d8'], [0.28, '#00c9ed'], [0.46, '#42e875'], [0.64, '#e8f21c'], [0.82, '#ff8611'], [1, '#c50000']];
  const divergentColorscale = [[0, '#174ea6'], [0.5, '#f7f7f7'], [1, '#b2182b']];

  function riskColors(value) {
    const fraction = Math.max(0, Math.min(1, value));
    const upper = colorscale.findIndex(stop => stop[0] >= fraction);
    const firstStop = colorscale[Math.max(0, upper - 1)], secondStop = colorscale[Math.max(1, upper)];
    const channels = color => [1, 3, 5].map(index => parseInt(color.slice(index, index + 2), 16));
    const first = channels(firstStop[1]), second = channels(secondStop[1]);
    const amount = (fraction - firstStop[0]) / (secondStop[0] - firstStop[0]);
    return first.map((component, index) => Math.round(component + amount * (second[index] - component)));
  }

  function robotGeometry(shape, length, diameter, sides = 24) {
    if (!['ellipse', 'capsule'].includes(shape) || length < diameter || diameter <= 0) throw new Error('Invalid robot geometry');
    const half = length / 2, radius = diameter / 2, capCenter = half - radius;
    const profile = [];
    if (shape === 'ellipse') {
      for (let ring = 1; ring < 16; ring++) {
        const angle = Math.PI * ring / 16;
        profile.push([-half * Math.cos(angle), radius * Math.sin(angle)]);
      }
    } else {
      for (let ring = 1; ring <= 6; ring++) {
        const angle = Math.PI / 2 * ring / 6;
        profile.push([-capCenter - radius * Math.cos(angle), radius * Math.sin(angle)]);
      }
      if (capCenter > 0) profile.push([capCenter, radius]);
      for (let ring = 1; ring < 6; ring++) {
        const angle = Math.PI / 2 * ring / 6;
        profile.push([capCenter + radius * Math.sin(angle), radius * Math.cos(angle)]);
      }
    }
    const vertices = [[-half, 0, 0]];
    for (const [position, width] of profile) {
      for (let side = 0; side < sides; side++) {
        const angle = Math.PI * 2 * side / sides;
        vertices.push([position, width * Math.cos(angle), width * Math.sin(angle)]);
      }
    }
    vertices.push([half, 0, 0]);
    const mesh = {x: vertices.map(point => point[0]), y: vertices.map(point => point[1]), z: vertices.map(point => point[2]), i: [], j: [], k: []};
    const triangle = (first, second, third) => {mesh.i.push(first); mesh.j.push(second); mesh.k.push(third);};
    for (let side = 0; side < sides; side++) triangle(0, 1 + (side + 1) % sides, 1 + side);
    for (let ring = 0; ring < profile.length - 1; ring++) {
      for (let side = 0; side < sides; side++) {
        const first = 1 + ring * sides + side, next = 1 + ring * sides + (side + 1) % sides;
        triangle(first, next, first + sides);
        triangle(next, next + sides, first + sides);
      }
    }
    const pole = vertices.length - 1, start = 1 + (profile.length - 1) * sides;
    for (let side = 0; side < sides; side++) triangle(start + side, start + (side + 1) % sides, pole);
    return mesh;
  }

  function insideRobot(shape, point, length, diameter) {
    const half = length / 2, radius = diameter / 2;
    if (shape === 'ellipse') return (point[0] / half) ** 2 + (point[1] ** 2 + point[2] ** 2) / radius ** 2 <= 1;
    if (shape !== 'capsule') throw new Error('Unsupported shape');
    const axialDistance = Math.max(Math.abs(point[0]) - (half - radius), 0);
    return axialDistance ** 2 + point[1] ** 2 + point[2] ** 2 <= radius ** 2;
  }

  function localField(state, displacementWorldMm) {
    return axes.map(axis => state[`applied_field_${axis}_mT`] + axes.reduce((sum, component, index) => sum + state[`applied_G_${axis}${component}_T_m`] * displacementWorldMm[index], 0));
  }

  function steeringVector(state) {
    const direction = axes.map(axis => state[`actual_axis_${axis}_unit`]);
    const field = axes.map(axis => state[`applied_field_${axis}_mT`]);
    const parallel = field.reduce((sum, value, index) => sum + value * direction[index], 0);
    return field.map((value, index) => value - parallel * direction[index]);
  }

  function fieldVector(state, point, mode, biasMilliTesla) {
    if (mode === 'turning') return steeringVector(state).map(value => value * 1000);
    const vector = localField(state, point);
    if (mode === 'total') return vector;
    if (mode === 'residual') return vector.map((value, index) => value - biasMilliTesla * state[`actual_axis_${axes[index]}_unit`]);
    throw new Error('Unknown field mode');
  }

  function forceProjection(state, prefix) {
    const direction = axes.map(axis => state[`ref_axis_${axis}_unit`]);
    const vector = axes.map(axis => state[`${prefix}_force_${axis}_mN`]);
    const parallel = vector.reduce((sum, value, index) => sum + value * direction[index], 0);
    const projected = direction.map(value => parallel * value);
    return {vector, parallel, projected, transverse: vector.map((value, index) => value - projected[index])};
  }

  function frame(state) {
    return ['axis', 'normal', 'binormal'].map((basis, index) => axes.map((axis, coordinate) => state[`actual_${basis}_${axis}_unit`] ?? Number(index === coordinate)));
  }

  function worldToBody(state, point) {
    return frame(state).map(direction => direction.reduce((sum, value, index) => sum + value * point[index], 0));
  }

  function transformedRobot(state, robot, scale = 1, centered = false) {
    const mesh = robotGeometry(robot.shape, robot.length_mm * scale, robot.diameter_mm * scale);
    const basis = frame(state);
    const center = centered ? [0, 0, 0] : axes.map(axis => state[`actual_position_${axis}_mm`]);
    const points = mesh.x.map((value, index) => {
      const body = [value, mesh.y[index], mesh.z[index]];
      return center.map((position, coordinate) => position + body.reduce((sum, component, basisIndex) => sum + component * basis[basisIndex][coordinate], 0));
    });
    return {...mesh, x: points.map(point => point[0]), y: points.map(point => point[1]), z: points.map(point => point[2])};
  }

  function fieldGrid(state, robot, halfSpanMm = 30, count = 5, mode = 'total', biasMilliTesla = 15) {
    const grid = {x: [], y: [], z: [], u: [], v: [], w: [], magnitude: []};
    for (let first = 0; first < count; first++) {
      for (let second = 0; second < count; second++) {
        for (let third = 0; third < count; third++) {
          const point = [first, second, third].map(index => -halfSpanMm + index * 2 * halfSpanMm / (count - 1));
          if (insideRobot(robot.shape, worldToBody(state, point), robot.length_mm, robot.diameter_mm)) continue;
          const vector = fieldVector(state, point, mode, biasMilliTesla);
          ['x', 'y', 'z'].forEach((axis, index) => grid[axis].push(point[index]));
          ['u', 'v', 'w'].forEach((axis, index) => grid[axis].push(vector[index]));
          grid.magnitude.push(Math.hypot(...vector));
        }
      }
    }
    return grid;
  }

  function fieldSlice(state, robot, mode, biasMilliTesla, plane = 'XY', count = 21, halfSpanMm = 30) {
    const selected = {XY:[0,1],XZ:[0,2],YZ:[1,2]}[plane];
    if (!selected) throw new Error('Unknown slice plane');
    const result = {x:[], y:[], z:[], magnitude:[]};
    for (let row = 0; row < count; row++) {
      const rowPoints = [], magnitudes = [];
      for (let column = 0; column < count; column++) {
        const point = [0,0,0];
        point[selected[0]] = -halfSpanMm + 2 * halfSpanMm * column / (count - 1);
        point[selected[1]] = -halfSpanMm + 2 * halfSpanMm * row / (count - 1);
        rowPoints.push(point);
        magnitudes.push(insideRobot(robot.shape, worldToBody(state,point),robot.length_mm,robot.diameter_mm) ? null : Math.hypot(...fieldVector(state,point,mode,biasMilliTesla)));
      }
      axes.forEach((axis,index)=>result[axis].push(rowPoints.map((point,column)=>axis==='z'&&magnitudes[column]===null?null:point[index])));
      result.magnitude.push(magnitudes);
    }
    return result;
  }

  function streamlines(state, robot, mode, biasMilliTesla, halfSpanMm = 30) {
    const lines = {x:[], y:[], z:[], magnitude:[]};
    const inDomain = point => point.every(value=>Math.abs(value)<=halfSpanMm + 1e-6) && !insideRobot(robot.shape,worldToBody(state,point),robot.length_mm,robot.diameter_mm);
    const unit = point => {
      const vector = fieldVector(state,point,mode,biasMilliTesla), length = Math.hypot(...vector);
      return length > 1e-10 ? vector.map(value=>value/length) : null;
    };
    for (let face = 0; face < 3; face++) {
      for (const sign of [-1,1]) {
        for (const first of [-20,0,20]) {
          for (const second of [-20,0,20]) {
            const seed = [0,0,0], transverseAxes = [0,1,2].filter(index=>index!==face);
            seed[face] = sign * halfSpanMm;
            seed[transverseAxes[0]] = first;
            seed[transverseAxes[1]] = second;
            for (const directionSign of [-1,1]) {
              let point = seed.slice();
              const path = [point];
              for (let step = 0; step < 100; step++) {
                const direction = unit(point);
                if (!direction) break;
                const midpoint = point.map((value,index)=>value + .8 * directionSign * direction[index]);
                const midpointDirection = unit(midpoint);
                if (!midpointDirection) break;
                const next = point.map((value,index)=>value + 1.6 * directionSign * midpointDirection[index]);
                if (!inDomain(next)) break;
                path.push(next);
                point = next;
              }
              if (path.length<3) continue;
              for (const sample of path) {
                axes.forEach((axis,index)=>lines[axis].push(sample[index]));
                lines.magnitude.push(Math.hypot(...fieldVector(state,sample,mode,biasMilliTesla)));
              }
              axes.forEach(axis=>lines[axis].push(null));
              lines.magnitude.push(0);
            }
          }
        }
      }
    }
    return lines;
  }

  function nearestIndex(times, time) {
    let lower = 0, upper = times.length - 1;
    while (lower < upper) {
      const middle = Math.ceil((lower + upper) / 2);
      if (times[middle] <= time) lower = middle;
      else upper = middle - 1;
    }
    return lower;
  }

  function nearestSampleIndex(values, target) {
    if (!Array.isArray(values) || values.length === 0 || !Number.isFinite(target)) throw new Error('Cannot find nearest sample');
    let best = 0, distance = Math.abs(values[0] - target);
    for (let index = 1; index < values.length; index++) {
      const nextDistance = Math.abs(values[index] - target);
      if (nextDistance < distance) { best = index; distance = nextDistance; }
    }
    return best;
  }

  function translateTrace(trace, origin) {
    const result = {...trace};
    for (const [index, axis] of axes.entries()) {
      const values = trace[axis];
      if (!Array.isArray(values)) continue;
      const add = value => Array.isArray(value) ? value.map(add) : value === null ? null : value + origin[index];
      result[axis] = add(values);
    }
    return result;
  }

  const api = {axes, colorscale, divergentColorscale, riskColors, robotGeometry, insideRobot, localField, fieldVector, steeringVector, forceProjection, fieldGrid, fieldSlice, streamlines, transformedRobot, nearestIndex, nearestSampleIndex, translateTrace};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ViewerCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
