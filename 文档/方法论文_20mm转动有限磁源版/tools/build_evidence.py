from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[1]
sys.path[:0] = [str(PROJECT / "源码"), str(PROJECT / "构建" / "依赖")]
from magnetic_app.domain.replay import Replay

BLUE = "#0072B2"
ORANGE = "#B85600"
GREEN = "#007A5E"
PURPLE = "#8253A5"
GRAY = "#6A737D"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, "#A33A50", "#526B20", "#005F73", "#866133"]


def save_figure(figure, name, description):
    folder = PAPER / "figures"
    figure.savefig(folder / f"{name}.pdf")
    figure.savefig(folder / f"{name}.png", dpi=240)
    plt.close(figure)
    return {"file": name, "description_zh": description, "source": "当前 clon001 回放；无平滑、无统计推断", "width_mm": 174}


def main():
    for name in ["figures", "data", "audit", "source", "build", "output/pdf"]:
        (PAPER / name).mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    matplotlib.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 8,
                               "axes.titlesize": 8.5, "axes.labelsize": 8,
                               "axes.spines.top": False, "axes.spines.right": False,
                               "axes.unicode_minus": False, "pdf.fonttype": 42,
                               "savefig.facecolor": "white", "lines.linewidth": 1.0})
    replay = Replay(load_predictive=False)
    report = json.loads((PROJECT / "文档/八线圈仿真结果.json").read_text("utf-8"))
    audit = json.loads((PROJECT / "文档/验证/磁源与相机逻辑核验.json").read_text("utf-8"))
    times = np.asarray(replay.times)
    data = pd.DataFrame(replay.data)
    for index in range(8):
        data[f"C{index + 1}_current_a"] = replay.coils.currents[:, index]
    data.to_csv(PAPER / "data/time_series.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(replay.source).to_csv(PAPER / "data/route.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"face_id": np.arange(len(replay.areas)), "area_mm2": replay.areas,
                  "insertion_first_seen_s": replay.insertion_seen,
                  "withdrawal_first_seen_s": replay.withdrawal_seen}).to_csv(
                      PAPER / "data/coverage_faces.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({key: replay.camera[key] for key in ["time_s", "arc_s_mm"]} | {
        f"{kind}_{axis}": np.asarray(replay.camera[key])[:, index]
        for kind, key in [("angle_deg", "joint_angle_deg"), ("velocity_deg_s", "joint_velocity_deg_s"),
                          ("acceleration_deg_s2", "joint_acceleration_deg_s2")]
        for index, axis in enumerate(["yaw", "pitch", "roll"])
    }).to_csv(PAPER / "data/camera.csv", index=False, encoding="utf-8-sig")
    stats = {"case": "clon001", "date": "2026-10-18", "replay_rows": len(times),
             "replay_columns": len(replay.data), "face_count": len(replay.areas),
             "route_points": len(replay.centerline), "join_time_s": replay.join_time,
             "duration_s": replay.duration, "effective_robot": replay.robot,
             "camera_settings": replay.settings, "report": report,
             "current_min_a": replay.coils.currents.min(axis=0).tolist(),
             "current_max_a": replay.coils.currents.max(axis=0).tolist(),
             "current_initial_a": replay.coils.currents[0].tolist(),
             "sampled_statistics": {},
             "inherited_metadata_not_active": {
                 "raw_input_bias_field_T": 0.015, "effective_bias_field_T": 0.01,
                 "raw_config_ode_method": "DOP853", "active_integrator": "RK4, step 0.02 s",
                 "raw_camera_evaluation_step_s": 0.4, "active_final_coverage_step_s": 0.1,
                 "registry_sample_count": replay.robot.get("sample_count"),
                 "actual_replay_sample_count": len(times),
                 "insertion_goal": "Inherited look-ahead goal indexed by arc, transformed to actual body; not recomputed from live images",
                 "velocity_policy": "Acceleration actively drives tangent speed toward an upper envelope; not unconstrained speed emergence"}}
    for key in ["actual_speed_mm_s", "ref_speed_mm_s", "position_error_mm", "axis_error_deg",
                "applied_field_norm_mT", "applied_field_perpendicular_mT", "applied_force_norm_mN",
                "applied_gradient_norm_T_m", "time_dilation", "clearance_mm"]:
        values = data[key].to_numpy(float)
        stats["sampled_statistics"][key] = {"min": float(values.min()), "max": float(values.max()),
                                             "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    field = data[[f"applied_field_{axis}_mT" for axis in "xyz"]].to_numpy()
    desired = data[[f"required_field_{axis}_mT" for axis in "xyz"]].to_numpy()
    force = data[[f"applied_force_{axis}_mN" for axis in "xyz"]].to_numpy()
    desired_force = data[[f"required_force_{axis}_mN" for axis in "xyz"]].to_numpy()
    stats["sampled_field_tracking_error_max_T"] = float(np.linalg.norm(field-desired, axis=1).max()/1000)
    stats["sampled_force_tracking_error_max_N"] = float(np.linalg.norm(force-desired_force, axis=1).max()/1000)
    stats["audit_reconstruction"] = {key: value for key, value in audit.items() if "error" in key}
    stats["environment"] = {"python": sys.version, "platform": platform.platform(),
                             "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (PAPER / "audit/evidence.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    manifest = []
    for folder in ["源码/coil_simulation", "源码/retreat_planner", "工具"]:
        for path in sorted((PROJECT/folder).glob("*.py")):
            manifest.append({"path": str(path.relative_to(PROJECT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for relative in ["数据/clon001/arrays.npz", "数据/clon001/metadata.json", "文档/八线圈仿真结果.json",
                     "数据/仿真输入/route_direction.json"]:
        path = PROJECT / relative
        manifest.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (PAPER / "audit/source_hashes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    figures = []
    width = 174/25.4
    figure, axis = plt.subplots(figsize=(width, 108/25.4))
    axis.set(xlim=(0, 10), ylim=(0, 7.3)); axis.axis("off")
    boxes = [(0.1, 5.75, 2.7, 1.05, "三维路线与几何风险\n弧长定位、净空、曲率"),
             (3.65, 5.75, 2.7, 1.05, "导航控制器\n纵向推进／制动＋横向纠偏"),
             (7.2, 5.75, 2.7, 1.05, "磁场优先电流分配\n先场后力；每路 ±10 A"),
             (7.2, 3.25, 2.7, 1.05, "八个等效线圈磁源\n合成磁场、梯度、力与转矩"),
             (3.65, 3.25, 2.7, 1.05, "刚体动力学积分\n位置、速度、姿态、角速度"),
             (0.1, 3.25, 2.7, 1.05, "相机目标＋受限 PD\n进镜前瞻／退镜密集预测"),
             (0.1, .75, 2.7, 1.05, "几何可见性与观察时长\n进镜覆盖＋退镜去重新增"),
             (3.65, .75, 2.7, 1.05, "相机误差与角速度\n降低速度上限，不设等待"),
             (7.2, .75, 2.7, 1.05, "阵列整体平移随行\n0.2 s 更新；限速插值")]
    for left, bottom, box_width, box_height, text in boxes:
        axis.add_patch(FancyBboxPatch((left,bottom),box_width,box_height,boxstyle="round,pad=0.08",
                                      facecolor="#F2F6FA",edgecolor=BLUE,linewidth=.9))
        axis.text(left+box_width/2,bottom+box_height/2,text,ha="center",va="center",fontsize=7.4)
    for begin,end in [((2.85,6.28),(3.53,6.28)),((6.42,6.28),(7.08,6.28)),
                      ((8.55,5.65),(8.55,4.42)),((7.08,3.78),(6.42,3.78)),
                      ((3.53,3.78),(2.85,3.78)),((1.45,3.15),(1.45,1.92)),
                      ((2.85,1.27),(3.53,1.27)),
                      ((8.55,1.92),(8.55,3.12))]:
        axis.add_patch(FancyArrowPatch(begin,end,arrowstyle="-|>",mutation_scale=9,color=GRAY))
    axis.annotate("状态真值反馈", xy=(5,5.65),xytext=(5,4.43),ha="center",va="center",fontsize=7,
                  arrowprops={"arrowstyle":"->","color":ORANGE},color=ORANGE)
    axis.annotate("",xy=(6.32,5.72),xytext=(6.28,1.8),
                  arrowprops={"arrowstyle":"->","color":ORANGE,"connectionstyle":"arc3,rad=.27"})
    axis.text(6.7,2.3,"速度上限反馈",fontsize=6.5,color=ORANGE,rotation=90)
    axis.text(5,.05,"仿真闭环：状态由模型直接获得；桌面软件仅回放，不含实物定位／电流反馈。",ha="center",fontsize=7)
    figure.subplots_adjust(left=.02,right=.98,bottom=.04,top=.98)
    figures.append(save_figure(figure,"fig01_architecture","导航、线圈、动力学、相机、覆盖和减速之间的反馈结构；实际硬件传感闭环尚未实现。"))
    figure = plt.figure(figsize=(width, 103/25.4))
    axis = figure.add_subplot(projection="3d")
    path = replay.centerline
    axis.plot(*path.T,color=GRAY,linewidth=1.2,label="既定路线")
    sources = replay.coils.positions[0]
    axis.scatter(*sources.T,c=COLORS,s=29,depthshade=False)
    for index, point in enumerate(sources): axis.text(*point,f" C{index+1}",fontsize=6)
    axis.scatter(*path[0],color=GREEN,s=32,marker="o",label="肛门入口")
    axis.scatter(*path[-1],color=PURPLE,s=36,marker="^",label="深部终点")
    axis.quiver(*(path[0]+[60,0,50]),0,0,-55,color="#A33A50",arrow_length_ratio=.2)
    axis.text(*(path[0]+[65,0,30]),"重力",fontsize=7)
    axis.set(xlabel="世界 X / mm",ylabel="世界 Y / mm",zlabel="世界 Z / mm")
    ranges=np.ptp(np.vstack([path,sources]),axis=0); axis.set_box_aspect(ranges)
    axis.view_init(elev=23,azim=-64); axis.legend(loc="upper left",fontsize=7,frameon=False)
    figure.subplots_adjust(left=.02,right=.95,bottom=.07,top=.98)
    figures.append(save_figure(figure,"fig02_geometry","同一真实坐标比例下的仰卧摆放假设、路线和初始八磁源位置；点标记不表示线圈尺寸。"))

    def timeline(axes):
        for axis in np.atleast_1d(axes):
            axis.axvline(replay.join_time,color=GRAY,linestyle=":",linewidth=.8)
            axis.set_xlim(0,replay.duration)
            axis.grid(axis="y",alpha=.17)
        np.atleast_1d(axes)[-1].set_xlabel("累计仿真时间 / s；虚线为进退镜切换")

    figure,axes=plt.subplots(4,1,figsize=(width,150/25.4),sharex=True,layout="constrained")
    axes[0].plot(times,data.actual_speed_mm_s,color=BLUE,label="实际速度")
    axes[0].plot(times,data.ref_speed_mm_s,color=ORANGE,linestyle="--",label="反馈速度上限")
    axes[0].set(ylabel="速度 / (mm/s)",ylim=(0,None));axes[0].legend(fontsize=7,ncol=2,frameon=False)
    axes[1].plot(times,data.risk_norm,color=PURPLE);axes[1].set(ylabel="几何风险指数",ylim=(0,1.05))
    axes[2].plot(times,data.position_error_mm,color=GREEN);axes[2].set(ylabel="路径偏差 / mm",ylim=(0,None))
    axes[3].plot(times,data.axis_error_deg,color=ORANGE);axes[3].set(ylabel="长轴误差 / °",ylim=(0,None))
    timeline(axes)
    figures.append(save_figure(figure,"fig03_motion","实际动力学速度、速度上限、几何风险、路径偏差和长轴误差。曲线为保存样本，不代表连续时间极值。"))
    figure,axes=plt.subplots(4,2,figsize=(width,154/25.4),sharex=True,sharey=True,layout="constrained")
    for index,axis in enumerate(axes.flat):
        axis.plot(times,replay.coils.currents[:,index],color=COLORS[index]);axis.set_title(f"C{index+1}",loc="left",pad=2)
        axis.axhline(10,color=GRAY,linestyle="--",linewidth=.6);axis.axhline(-10,color=GRAY,linestyle="--",linewidth=.6)
        axis.set(ylim=(-11,11));axis.set_yticks([-10,0,10])
        if index%2==0: axis.set_ylabel("电流 / A")
    timeline(axes.ravel())
    for axis in axes[:-1].ravel(): axis.set_xlabel("")
    figures.append(save_figure(figure,"fig04_currents","八路电流在进退镜全程独立变化；灰色横虚线为 ±10 A 约束。"))
    figure,axes=plt.subplots(4,1,figsize=(width,148/25.4),sharex=True,layout="constrained")
    for index,component in enumerate("xyz"):
        axes[0].plot(times,field[:,index],color=COLORS[index],label=f"B{component}",linestyle=["-","--","-."][index])
    axes[0].set(ylabel="磁场分量 / mT",ylim=(-11,11));axes[0].legend(ncol=3,fontsize=7,frameon=False)
    axes[1].plot(times,data.applied_field_norm_mT,color=BLUE);axes[1].set(ylabel="总场幅值 / mT",ylim=(0,11))
    axes[2].plot(times,data.applied_gradient_norm_T_m,color=PURPLE);axes[2].set(ylabel="梯度范数 / (T/m)",ylim=(0,None))
    axes[3].plot(times,data.applied_force_norm_mN,color=ORANGE);axes[3].set(ylabel="磁力 / mN",ylim=(0,None))
    timeline(axes)
    figures.append(save_figure(figure,"fig05_field_force","总场幅值接近10 mT而方向、梯度与磁力变化。总场轴从零起，避免放大数值舍入误差。"))
    camera_times=np.asarray(replay.camera['time_s'])
    figure,axes=plt.subplots(3,1,figsize=(width,139/25.4),sharex=True,layout="constrained")
    for index,name in enumerate(["偏航","俯仰","滚转"]):
        axes[0].plot(camera_times,np.asarray(replay.camera['joint_angle_deg'])[:,index],color=COLORS[index],label=name,linestyle=["-","--","-."][index])
        axes[1].plot(camera_times,np.asarray(replay.camera['joint_velocity_deg_s'])[:,index],color=COLORS[index],linestyle=["-","--","-."][index])
    axes[0].set_ylabel("相对关节角 / °");axes[0].legend(ncol=3,fontsize=7,frameon=False)
    axes[1].set(ylabel="关节角速度 / (°/s)",ylim=(-65,65))
    for bound in [-60,60]: axes[1].axhline(bound,color=GRAY,linestyle=":",linewidth=.7)
    covered=np.array([replay.areas[(replay.insertion_seen>=0)&(replay.insertion_seen<=time)].sum() for time in times])
    inserted=(replay.insertion_seen>=0)
    added=np.array([replay.areas[(~inserted)&(replay.withdrawal_seen>=0)&(replay.withdrawal_seen<=time-replay.join_time)].sum()
                    if time>=replay.join_time else 0. for time in times])
    total=replay.areas.sum()
    axes[2].fill_between(times,0,100*covered/total,color=BLUE,alpha=.24,label="进镜有效覆盖")
    axes[2].plot(times,100*covered/total,color=BLUE)
    axes[2].fill_between(times,100*covered/total,100*(covered+added)/total,color=ORANGE,alpha=.3,label="退镜去重新增")
    axes[2].plot(times,100*(covered+added)/total,color=ORANGE,linestyle="--")
    axes[2].set(ylabel="累计几何覆盖 / %",ylim=(0,100));axes[2].legend(ncol=2,fontsize=7,frameon=False)
    timeline(axes)
    pd.DataFrame({"time_s":times,"insertion_area_mm2":covered,"withdrawal_added_area_mm2":added}).to_csv(PAPER/'data/coverage_timeline.csv',index=False)
    figures.append(save_figure(figure,"fig06_camera_coverage","云台关节运动及蓝色进镜、橙色退镜新增去重覆盖；roll包角会出现表示跳变，不等价于真实瞬转。"))
    (PAPER/'audit/figure_manifest.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({"paper":str(PAPER),"figures":len(figures),"samples":len(times),"statistics":stats['sampled_statistics'],
                      "field_error_T":stats['sampled_field_tracking_error_max_T'],"force_error_N":stats['sampled_force_tracking_error_max_N']},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
