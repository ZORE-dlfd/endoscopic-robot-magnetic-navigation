import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import model
from .redundant_actuator import default_array_layout


def save_payload(payload, destination):
    arrays = {}

    def encode(value, key='payload'):
        if isinstance(value, np.ndarray):
            if value.dtype.kind in 'biuf':
                arrays[key] = value
                return {'__array__': key}
            return value.tolist()
        if isinstance(value, dict):
            return {name: encode(item, key+'/'+name) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item, key+'/'+str(index)) for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    destination = Path(destination)
    schema = encode(payload)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'arrays.npz', **arrays)
    (destination/'metadata.json').write_text(json.dumps(schema, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False), 'utf-8')


def make_payload(original, insertion, withdrawal, config, settings, reports):
    payload = original.payload
    join = insertion[-1]['time_s']
    combined = [(row, row['time_s']) for row in insertion]
    combined += [(row, join+row['time_s']) for row in withdrawal[1:]]
    times = np.array([absolute for row, absolute in combined])
    count = len(times)
    data = {key: np.zeros(count) for key in original.data}
    data['phase'] = []
    camera = {key: [] for key in original.camera}
    source_positions, source_currents = [], []
    source = original.source
    for index, (row, absolute) in enumerate(combined):
        state, reference, demand = row['state'], row['reference'], row['demand']
        arc_mm = row['source_arc'] * 1000.
        rotation = row['body'].as_matrix()
        reference_rotation = reference['rotation'].as_matrix()
        for key in ['risk_norm','total_cost_raw','radius_mm','clearance_mm','curvature_inv_mm',
                    'speed_risk_limit_mm_s','speed_clearance_limit_mm_s','speed_curvature_limit_mm_s']:
            if key in source:
                data[key][index] = np.interp(arc_mm, source['arc_s_mm'], source[key])
        data['time_s'][index] = absolute
        data['arc_s_mm'][index] = arc_mm
        data['speed_target_mm_s'][index] = row['speed_cap'] * 1000.
        data['time_dilation'][index] = row['dilation']
        data['equivalent_normal_load_mN'][index] = config['mass_kg'] * np.linalg.norm(config['gravity_m_s2']) * 1000.
        data['position_error_mm'][index] = np.linalg.norm(state[:3]-reference['position_m']) * 1000.
        data['axis_error_deg'][index] = np.rad2deg(np.arccos(np.clip(np.dot(rotation[:,0],reference_rotation[:,0]),-1,1)))
        data['full_attitude_error_deg'][index] = np.rad2deg((reference['rotation'].inv()*row['body']).magnitude())
        data['quaternion_norm_error'][index] = abs(np.linalg.norm(state[6:10])-1)
        data['phase'].append('withdrawal' if row['reverse'] else 'insertion')
        for axis_index, axis in enumerate('xyz'):
            data[f'actual_position_{axis}_mm'][index] = state[axis_index]*1000.
            data[f'actual_velocity_{axis}_mm_s'][index] = state[axis_index+3]*1000.
            data[f'actual_acceleration_{axis}_mm_s2'][index] = row['derivative'][axis_index+3]*1000.
            data[f'ref_position_{axis}_mm'][index] = reference['position_m'][axis_index]*1000.
            data[f'ref_velocity_{axis}_mm_s'][index] = reference['tangent'][axis_index]*row['speed_cap']*1000.
            data[f'ref_acceleration_{axis}_mm_s2'][index] = demand['acceleration_command'][axis_index]*1000.
            data[f'actual_axis_{axis}_unit'][index] = rotation[axis_index,0]
            data[f'actual_normal_{axis}_unit'][index] = rotation[axis_index,1]
            data[f'actual_binormal_{axis}_unit'][index] = rotation[axis_index,2]
            data[f'ref_axis_{axis}_unit'][index] = reference_rotation[axis_index,0]
            data[f'actual_omega_body_{axis}_rad_s'][index] = state[axis_index+10]
            data[f'actual_alpha_body_{axis}_rad_s2'][index] = row['derivative'][axis_index+10]
        for quaternion_index, axis in enumerate('xyzw'):
            data['actual_q'+axis][index] = state[quaternion_index+6]
            data['ref_q'+axis][index] = reference['rotation'].as_quat()[quaternion_index]
        for axis, angle in zip(['roll','pitch','yaw'], row['body'].as_euler('xyz', degrees=True)):
            data[f'actual_{axis}_deg'][index] = angle
        data['actual_speed_mm_s'][index] = np.linalg.norm(state[3:6])*1000.
        data['ref_speed_mm_s'][index] = row['speed_cap']*1000.
        data['actual_acceleration_norm_mm_s2'][index] = np.linalg.norm(row['derivative'][3:6])*1000.
        data['ref_acceleration_norm_mm_s2'][index] = np.linalg.norm(demand['acceleration_command'])*1000.
        data['actual_angular_speed_rad_s'][index] = np.linalg.norm(state[10:13])
        required_gradient = model.minimum_gradient(demand['moment'], demand['force_command'])
        for prefix, field, gradient, force, torque in [
            ('required',demand['field'],required_gradient,demand['force_command'],demand['torque_command']),
            ('applied',row['field'],row['gradient'],row['force'],row['torque'])]:
            unavailable = np.dot(torque, rotation[:,0])*rotation[:,0]
            vectors = [('field',field,'mT',1000.),('force',force,'mN',1000.),
                       ('torque',torque,'uN_m',1e6),('unavailable_torque',unavailable,'uN_m',1e6),
                       ('resistance',demand['resistance'],'mN',1000.)]
            for name, vector, unit, scale in vectors:
                for axis_index, axis in enumerate('xyz'):
                    data[f'{prefix}_{name}_{axis}_{unit}'][index] = vector[axis_index]*scale
                norm_key = f'{prefix}_{name}_norm_{unit}'
                if norm_key in data:
                    data[norm_key][index] = np.linalg.norm(vector)*scale
            for first, first_axis in enumerate('xyz'):
                for second, second_axis in enumerate('xyz'):
                    data[f'{prefix}_G_{first_axis}{second_axis}_T_m'][index] = gradient[first,second]
            data[f'{prefix}_field_perpendicular_mT'][index] = np.linalg.norm(field-np.dot(field,rotation[:,0])*rotation[:,0])*1000.
            data[f'{prefix}_gradient_norm_T_m'][index] = np.linalg.norm(gradient)
            data[f'{prefix}_maxwell_trace_T_m'][index] = np.trace(gradient)
            data[f'{prefix}_maxwell_symmetry_T_m'][index] = np.linalg.norm(gradient-gradient.T)
            data[f'{prefix}_force_residual_N'][index] = np.linalg.norm(gradient.T@demand['moment']-force)
            data[f'{prefix}_torque_residual_N_m'][index] = np.linalg.norm(np.cross(demand['moment'],field)-(torque-unavailable))
        relative = Rotation.from_euler('ZYX',row['camera_angles'],degrees=True)
        world = row['body']*relative
        goal_world = row['body']*Rotation.from_euler('ZYX',row['goal'],degrees=True)
        position = state[:3]*1000.+rotation[:,0]*original.settings['mount_offset_mm']
        angle_error = np.rad2deg((world.inv()*goal_world).magnitude())
        values = dict(time_s=absolute,position=position,quaternion=world.as_quat(),
            target=position+goal_world.as_matrix()[:,0]*20.,goal_quaternion=goal_world.as_quat(),
            cone_residual_deg=np.rad2deg(np.arccos(np.clip(relative.as_matrix()[0,0],-1,1))),
            tracking_error_deg=angle_error,preview_mm=20.,joint_angle_deg=row['camera_angles'],
            joint_velocity_deg_s=row['camera_velocity'],joint_acceleration_deg_s2=row['camera_acceleration'],
            aim_error_deg=angle_error,goal_joint_angle_deg=row['goal'],arc_s_mm=arc_mm)
        for key in camera:
            camera[key].append(values[key])
        source_positions.append(row['positions']*1000.)
        source_currents.append(row['currents'])
    data['phase'] = np.asarray(data['phase'])
    reference_rotations = Rotation.from_quat(np.column_stack([data['ref_q'+axis] for axis in 'xyzw']))
    reference_rates = (reference_rotations[:-1].inv()*reference_rotations[1:]).as_rotvec()/np.diff(times)[:,None]
    reference_rates = np.vstack([reference_rates,reference_rates[-1]])
    for axis_index, axis in enumerate('xyz'):
        data[f'ref_omega_body_{axis}_rad_s'] = reference_rates[:,axis_index]
        data[f'ref_alpha_body_{axis}_rad_s2'] = np.gradient(data[f'ref_omega_body_{axis}_rad_s'],times)
    data['ref_angular_speed_rad_s'] = np.linalg.norm(np.column_stack([data[f'ref_omega_body_{axis}_rad_s'] for axis in 'xyz']),axis=1)
    payload['datasets'] = dict(capsule_roundtrip=data)
    payload['cameras'] = dict(roundtrip_new={key:np.asarray(value) for key,value in camera.items()})
    _, axes = default_array_layout()
    payload['coilSources'] = dict(positions_mm=np.asarray(source_positions),currents_a=np.asarray(source_currents),
        axes=axes,moment_per_amp_a_m2=settings.moment_per_amp,current_limit_a=settings.current_limit_a)
    payload['roundTrip'] = dict(join_time_s=join,join_index=len(insertion)-1,total_duration_s=float(times[-1]),
        insertion_duration_s=join,withdrawal_duration_s=withdrawal[-1]['time_s'],
        endpoint_position_gap_mm=float(np.linalg.norm(withdrawal[-1]['state'][:3]-insertion[0]['state'][:3])*1000.),
        note='Continuous magnetic dynamics; capsule, gimbal and array anchor carried across reversal. No pose reset.')
    payload['magneticSimulation'] = dict(method='eight-coil-field-priority-dynamics-v1', reports=reports,
        step_s=settings.step_s,output_s=settings.output_s,coil_count=8,
        field_model='Point-dipole sources, not FEM or measured hardware',
        camera_method='Original offline search goals indexed by route arc; actual gimbal PD rerun on new body poses',
        constraints='10 A current bounds; array translation bound; no inductance, thermal, joint or wall-contact model',
        no_position_forcing=True,no_camera_wait=True,patient_posture=payload.get('patientPosture'))
    payload['registry']['robots'][0]['config'] = dict(config,field_model='Eight point-dipole coils with bounded currents')
    faces = len(payload['geometry']['areas_mm2'])
    payload['evaluation']['methods']['slowdown_new']['first_seen_s'] = np.full(faces,-1.)
    payload['withdrawalCoverage']['masks'] = dict(insertion_covered=np.zeros(faces,bool),withdrawal_first_seen_s=np.full(faces,-1.))
    payload['withdrawalCoverage']['summary'] = dict(original.summary)
    payload['evaluation']['methods']['withdrawal_new']['rows'] = [dict(original.metric_rows[0],time_s=0.)]
    camera_rotations = Rotation.from_quat(payload['cameras']['roundtrip_new']['quaternion'])
    relative_rates = (camera_rotations[:-1].inv()*camera_rotations[1:]).as_rotvec()/np.diff(times)[:,None]
    payload['cameraMotion']['roundtrip_new'] = dict(roll_rate_deg_s=np.r_[0.,np.rad2deg(relative_rates[:,0])])
    return payload
