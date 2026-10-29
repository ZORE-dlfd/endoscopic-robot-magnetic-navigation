import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'源码'),str(ROOT/'构建'/'依赖')]

import numpy as np
from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.geometry import Domains
from coil_simulation.export import save_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,default=ROOT/'数据'/'clon001')
    assets = parser.parse_args().assets.resolve()
    replay = Replay(assets,load_predictive=False)
    replay.summary['constraint_violating_steps'] = replay.prediction['metrics']['constraint_violating_steps']
    replay.summary['method_notice'] = 'Eight-coil dynamics coupled with original dense window search + 1.2 s prediction + 0.01 s PD; actual wall coverage recomputed. Not measured or clinical coverage.'
    replay.prediction['metrics']['withdrawal_added_area_mm2'] = replay.summary['withdrawal_added_area_mm2']
    domains = Domains(replay)
    for scope in ['local','centerline','whole']:
        values = {mode:[] for mode in ['total','residual','turning']}
        for time_s in np.linspace(0.,replay.duration,31):
            frame = replay.frame(time_s)
            points = domains.points(frame,scope)
            points = points[domains.valid(points,frame,scope)]
            for mode in values:
                values[mode].extend(np.linalg.norm(replay.field(frame,points,mode),axis=-1).tolist())
        for mode, samples in values.items():
            upper = max(float(np.percentile(samples,99)),1e-3) if samples else 1.
            replay.registry['field_scope_ranges'][scope][mode] = [0.,upper]
    replay.payload['magneticSimulation']['color_scale_notice'] = '99th percentile display scales sampled across the new source field; source equations are not clipped.'
    save_payload(replay.payload,assets)
    sections = json.loads((ROOT/'数据'/'参考原版'/'algorithms.json').read_text('utf-8'))
    sections['机器人运动'] = '''八线圈磁源驱动的空间路径跟随

示例仍为 clon001，胶囊尺寸 12 × 6 mm，质量、磁矩及理想摩擦沿用原配置。肠壁、路线、机身参考姿态、相机与法向统一刚体变换为仰卧摆放假设，旧 Z 轴转为世界负 Y，旧 Y 转为世界 Z。世界 Z 竖直向上，重力始终为 (0, 0, −9.81) m/s²，不跟着肠道一起旋转。八线圈保持在世界正 Z 一侧，重新求电流并积分运动，不是把旧轨迹转横。原网格没有经核验的解剖方向，不能将本摆放称为患者真实体位还原。水平参考面 Z=0、肠壁最低点 20 mm 均为展示摆放，不参与接触计算。

进镜方向固定为肛门入口 → 肠道深部终点，随后连续退镜回到同一个肛门入口。路线、朝向、风险与弧长全部来自原版确认的进镜 source，不使用方向相反的旧 trajectory_state.csv。弧长 0 对应肛门，进镜增加、退镜减少。生成前核对原始肛门与深部坐标，生成后核验实际起点、换向点、退镜终点，方向或终点不符就拒绝发布，不倒放旧数据、不交换标签。

机器人不是沿中心线逐点移动，也不按原版时间表强制设定位置。当前位置和姿态由磁力、磁转矩进入动力学方程后积分得到。

平移：位置变化率 = 实际速度；质量 × 加速度 = 磁力 + 理想阻力 + 质量 × 重力加速度。转动：惯量 × 角加速度 + 角速度叉乘角动量 = 机体系磁转矩。姿态使用单位四元数。单磁偶极不直接控制绕自身磁矩轴线的滚转。

路径控制：沿路线方向产生推进或制动需求；横向按位置误差和横向速度生成纠偏力。期望磁力 = 质量 ×（期望加速度 − 重力加速度）− 理想阻力。理想阻力仍使用平滑滑动摩擦公式，法向负载近似为质量乘重力大小，未求肠壁接触。

风险、净空、曲率与前方制动包络共同限制速度。相机误差和角速度增加时，按 1.0～1.25 时间膨胀降低速度上限，不设置中途等待段。速度仍有上限反馈，但不追踪原版时间—速度轨迹。进退方向切换保留完整动力学状态，速度由力连续改变，不重置位置和姿态。

数据为离线固定步长 RK4 积分，播放时使用 Hermite 位置插值和 SLERP 姿态插值。拖动进度条不重新解磁场。全程判定同时核验路线弧长进度与空间终点距离，不改写弧长数组。该模型未包含肠壁形变、静摩擦、组织损伤、机械臂关节碰撞或临床安全验证。'''
    sections['磁场解算'] = '''八线圈磁场优先控制与磁源空间叠加

八个线圈为两层交错环状布局，方位角间隔 45°，极角交替为 35° 和 60°，与阵列中心距离交替为 160 mm 和 130 mm，轴线朝阵列中心。相对布局固定，阵列整体平移随行，不是八台机械臂独立运动。

阵列的“上方”指世界坐标正 Z，与固定向下的重力相对。修改为仰卧后，只改变肠道与参考姿态，不能把旧电流、重力和线圈一起旋转当成新的物理结果。当前数据是在横卧姿态下重新联合求解的磁驱动与摄像头轨迹。

每个线圈的磁矩 = 每安培等效磁矩系数 × 电流 × 线圈轴线单位向量。当前系数假设为 20 A·m²/A，每路线圈电流范围为 −10～10 A。该系数尚未由真实匝数、铁芯和温升标定。画面中的圆环半径 12 mm 仅为磁源位置示意，不代表已验证的物理线圈尺寸。

单线圈磁场采用磁偶极解析式：B = μ₀/(4π) × [3r（磁矩点乘r）/距离⁵ − 磁矩/距离³]。八个磁源的磁场及空间梯度分别相加。磁力 = 磁场梯度转置乘胶囊磁矩；磁转矩 = 胶囊磁矩叉乘磁场。整个空间的箭头、流线和切片按八个磁源直接计算，不再使用胶囊附近的一阶仿射场外推到整条肠道。

电流分配：先求满足目标磁场的基础电流，再在磁场矩阵零空间内调整电流，使实际磁力尽量接近期望磁力，同时限制电流。当目标不可达时优先减小磁场误差，不能保证任意力和朝向都可实现。理想反解只提出目标，实际动力学采用磁源输出。

阵列平移速度上限为 0.20 m/s，位姿更新周期 0.2 s，更新之间连续插值。电流采用理想瞬时调节，未模拟电感、驱动器带宽、热、磁滞或饱和。色标根据新磁源数据抽样的第 99 百分位设置，颜色饱和不表示物理磁场被截断。此版本是点偶极磁源模型下的方法仿真，不是有限元或实测磁场。'''
    notice = '【八线圈版】进镜继续使用原版前瞻目标与同一套 PD。退镜重新执行原版滑窗密集搜索＋1.2 秒短时预测＋原 PD：每 0.4 秒重新搜索，收益达到预测最优值 95% 的可行候选中优先选择平滑方案。未来机身位置按当前真实磁驱动状态与速度沿路线短时预测，预测不强制改变机器人。相机误差和角速度实时影响磁驱动减速；不退回旧的固定 37 方向，也不只搬用旧角度。蓝色进镜、橙色退镜新增面积均按新轨迹重新离线评价。\n\n'
    sections['进镜摄像头'] = notice + sections['进镜摄像头']
    sections['退镜摄像头'] = notice + sections['退镜摄像头']
    (assets/'algorithms.json').write_text(json.dumps(sections,ensure_ascii=False,indent=2),'utf-8')
    manifest = dict(schema=1,method='eight-coil-field-priority-dynamics-v1',sha256={
        name:hashlib.sha256((assets/name).read_bytes()).hexdigest()
        for name in ['metadata.json','arrays.npz','algorithms.json']})
    (assets/'magnetic_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
    (assets/'provenance.json').write_text(json.dumps(dict(example='clon001',
        data='Supine-assumption placement; recomputed eight-coil dynamics and dense predictive camera planning; original UI and mesh topology retained',
        patient_posture=replay.payload.get('patientPosture'),
        route_direction=replay.payload.get('routeDirection'),
        manifest=manifest),ensure_ascii=False,indent=2),'utf-8')
    report = dict(simulation=replay.payload['magneticSimulation'],coverage=replay.summary,
        camera_planning=dict(method=replay.prediction['method'],config=replay.prediction['config'],
            metrics=replay.prediction['metrics'],minimum_searched_directions=min(row['searched_direction_count'] for row in replay.prediction['decisions']),
            maximum_searched_directions=max(row['searched_direction_count'] for row in replay.prediction['decisions'])))
    (ROOT/'文档'/'八线圈仿真结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print('Assets finalized with source-field color scales and checksums.')


if __name__ == '__main__':
    main()
