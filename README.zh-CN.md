# 内窥镜机器人磁导航仿真

> Endoscopic Robot Magnetic Navigation Simulation —— 一个离线、自包含的磁控胶囊相机在结肠模型中行进、带前瞻车载相机和表面覆盖评价的仿真器。
>
> [English version](README.md)

**状态：** 研究原型 · 单病例离线仿真 · **未经**临床验证。

---

## 这是什么（用大白话说）

想象一颗胶囊机器人——长 12 mm、宽 6 mm——被外部磁场牵引和转向，在一条大肠（结肠）模型里「游动」，前端装了一个小相机。这个项目完全离线地模拟了：

1. 胶囊在**磁力和磁力矩**作用下怎么运动；
2. 它前端的相机在行进中**看向哪里**（保持前视、尽量稳定）；
3. 结肠内壁有多少面积真正被相机**看到过**（覆盖率）；
4. 一个可**交互回放**的界面，能播放、暂停、拖进度条、切换相机策略。

它**不是**真实机器人，也**不是**真实影像，一切都是一条结肠模型的确定性数值回放。

## 怎么用

双击 `Magnetic_Camera.exe`。左侧第三人称和右侧第一人称共用一条时间条；播放、暂停、拖动均为确定性离线回放。上方运动方案可切换原速、只减速、只等待、减速＋等待，胶囊尺寸、质量和磁矩相同。相机策略可切换前瞻相机与 Endoscopy 基线。左侧可分别勾选肠道、覆盖染色、视锥、前瞻目标、磁场和各种物理向量，打开时默认显示整体磁场。本轮只做进镜协调，没有退镜与观察补拍。

评价卡是当前运动/相机组合的累计结果；下面比较表独立选择「仅相机/完整系统」和「共同预算/走完全程」，所以切换运动方案不会偷偷改变对照表的实验定义。覆盖颜色：红为未观察、青为观察过、黄为重复观察。覆盖模式时肠壁提高透明度以便阅读。

## 快速开始

- **Windows：** 双击 `Magnetic_Camera.exe`。它会解压内嵌的 HTML 到临时目录，再用 Edge 应用窗口打开。
- **任意浏览器：** 直接打开 `Magnetic_Camera.html`。

两者是同一个自包含页面。运行时**不需要**安装、服务器或任何外部文件。

## 建模流水线（跨文件）是怎么串起来的

这个项目是一条小「装配线」：每个脚本读上游文件（有些在**别的文件夹**里），写一个中间 JSON/CSV，交给下一步：

```
colon_model.stl              (源解剖网格, 187,206 面) ──┐
source_trajectory_state.csv  (中心线 xyz + 曲率/净空/风险) ├─┐
capsule_replay.csv + capsule_config.json (基线运动 + 机器人参数) ├─┤
magnetic_model.py            (力/力矩物理)                  ├─┤
build_simulation.py          (动力学求解器)                  ├─┤
3D Slicer Endoscopy.py       (基线相机)                    ├─┤
                                                            ▼ ▼
 1) prepare_geometry.py   → geometry.json (减面后网格, 33,696 面)
                            endoscopy_baseline.json
 2) build_trajectories.py → camera_trajectories.json
                            capsule_{slowdown,wait,cooperative}_replay.csv
 3) build_withdrawal.py   → capsule_withdrawal_replay.csv, withdrawal_camera.json
 4) evaluate.py           → evaluation.json, withdrawal_visibility_masks.json
 5) build_camera_viewer.py→ Magnetic_Camera.html (自包含, 约 35 MB)
 6) build_exe.py          → Magnetic_Camera.exe (内嵌 gzip 后的 HTML)
```

一个关键设计：**显示的网格和评价用的网格是同一张表面**，所以你在屏幕上看到的颜色，和后台打分的数字严格对应。

## 数据和基线

以 `Magnetic 1 - Field scopes/Magnetic_Viewer_v4.exe` 对应 HTML 和 CSV 为基线。解剖为 `结果_colon_001_含曲率/colon_model.stl`，路径同源；坐标范围核对一致，未再次 LPS/RAS 翻转。187,206 面原网格经 VTK QuadricDecimation 处理为 33,696 面，显示和评价使用同一表面。它是分割内表面代理，不含真实内镜纹理，也不包含未被分割的微小褶皱。

Endoscopy 基线直接调用本机 Slicer 5.10.0 `EndoscopyLogic`：输入相同线性路径、0.5 mm 等弧长重采样、不设人工姿态关键帧，保留默认相机方向和四元数插值。导出姿态按路径弧长插值到同一相机位置。公平评价时使用相同 60° 垂直 FOV、4:3、几何、观察门槛。这不是用户当前 Slicer 场景的自定义关键帧、FOV 或默认 30° 显示设置。

## 相机前瞻与防运动晃动

相机位置固定于机器人前端：

$$\mathbf{p}_{\mathrm{cam}} = \mathbf{p}_{\mathrm{robot}} + \mathbf{R}_{\mathrm{robot}}\,[5.8,\,0,\,0]^{\top}\ \mathrm{mm}$$

机器人实体尺寸 12 × 6 mm；前端镜头简化为壳体内可转的光轴。前瞻距离使用启发式，随速度 $v$（mm/s）、曲率 $\kappa$（1/mm）、归一化风险 $r$ 与净空 $c$ 变化：

$$L = \mathrm{clip}\!\left[\frac{12+1.8\,v}{1+25\,\kappa+0.8\,r},\;5,\;L_{\max}\right], \qquad L_{\max} = \mathrm{clip}[\,6+2c,\;5,\;45\,]$$

沿中心线逐步缩短前瞻，并用三角网格射线排除遮挡目标。方向筛选采用迟滞：新目标点积阈值为 0.50（约 60°），已跟踪目标保持阈值为 0.45（约 63.3°）；前瞻弧长变化限制为 10 mm/s。若暂时没有新目标，沿中心线维持上一前瞻距离；首次无候选时取最近的前方中心线位置，不把目标重置到机器人正前方的虚拟点。目标点始终由中心线插值得到；被遮挡状态仍单独标记，不代表相机一定能看见该点。目标朝向使用正交坐标架，上一帧 up 向量投影到新视平面，减少非必要滚转。

相对姿态为 $\mathbf{R}_{\mathrm{rel}} = \mathbf{R}_{\mathrm{robot}}^{\top}\mathbf{R}_{\mathrm{goal}}$，用 ZYX 欧拉关节参数 yaw/pitch/roll 描述可转镜头。关节控制律：

$$\ddot{\theta} = \mathrm{clip}\!\left[12\,e - 6\,\dot{\theta},\,-150,\,150\right]\ \text{deg/s}^2, \qquad |\dot{\theta}| \le 60\ \text{deg/s}$$

yaw、pitch 机械范围各为 ±89°，roll 可连续调节。数值积分子步 ≤0.01 s，在角限前采用制动速度 $\sqrt{2\,a_{\max}d}$。这些是关节运动假设，不代表已完成实体机构设计。

垂直 FOV 60°，水平约 75.2°，矩形视锥对角半角约 43.9°。为保证视锥整体不越过机器人后半空间，期望光轴相对本体前向轴最多偏转 45.0°（比 90°−43.9° 的几何上限多留约 1° 余量）；相机目标方向先投影到这个可视安全锥内，再由关节限速控制跟踪。摄像头朝向和机器人本体朝向分开；没有给单偶极子机器人添加主动轴向自转能力。

## 协同减速、等待和磁动力学

先区分两种误差：**锥外误差** $e_{\mathrm{cone}}=\arccos(\mathbf{u}_{\mathrm{target}}\cdot\mathbf{u}_{\mathrm{feasible}})$，表示中心线方向超出安全视锥后剩下的几何误差；**跟踪误差** $e_{\mathrm{track}}=2\arccos(|\mathbf{q}_{\mathrm{goal}}\cdot\mathbf{q}_{\mathrm{actual}}|)$，表示实际相机相对可达期望姿态的四元数夹角。两个式子的反三角结果都换成度。只因后一类可解决的滞后减速；目标超出前向锥但镜头已经跟上可达姿态时，不会因锥外残差反复等待。四元数用于误差度量，云台执行器仍为上述逐轴欧拉角 PD，不能把它说成完整的四元数云台控制器。

慢行因子（15° 死区、50°/s 速率软限）：

$$f = \mathrm{clip}\!\left[1+\frac{\max(e_{\mathrm{track}}-15^\circ,\,0)}{80^\circ}+\frac{\max(\omega_{\mathrm{rel}}-50\ \text{deg/s},\,0)}{120\ \text{deg/s}},\;1,\;1.25\right]$$

对 $f$ 用 $\sigma=6$ 个回放采样的高斯平滑后，重定时 $dt_{\mathrm{new}} = f\,dt_{\mathrm{old}}$。15° 死区避免把轻度常态误差当成全程慢行理由，50°/s 给 60°/s 的关节速率假设留余量；$f\le1.25$ 将局部参考速度降幅限制在约 20%。这是未经多病例标定的工程参数，不是论文给定的最优值。

等待按调度器模拟得到的当前关节角和角速度决定：逐轴误差最大值超过 24°、距离上次等待结束至少 8 s 时，冻结当前可达目标，用 0.01 s 子步积分同一 PD 动力学。只有预计在最多 1.2 s 内同时达到逐轴误差 ≤3° 和逐轴角速度 ≤5°/s，才插入所需时长；达不到则继续，不把不可解决的误差变成反复停等。全程等待总预算为原始时长的 3%，本例约 8.61 s。该预算是实验效率限制，不是安全保证；超出预算的滞后在结果中保留，不会被隐藏。

候选停等还要过一次实际重放状态核验：重算磁动力学和相机后，起点逐轴误差最大值需 ≥24°，终点逐轴误差 ≤3° 且逐轴角速度 ≤5°/s。没有滞后、不够稳定的候选全部删除，重新计算运动和前瞻目标，再核验剩余候选，直到全部通过。每轮只能删除候选，因此有限轮数内结束。

重定时维护单调时钟 $\tau(t)$（停等时 $\tau$ 不变），用链式法则缩放：

$$\mathbf{v}_{\mathrm{new}}=\mathbf{v}_{\mathrm{old}}\,\dot{\tau}, \qquad \mathbf{a}_{\mathrm{new}}=\mathbf{a}_{\mathrm{old}}\,\dot{\tau}^2+\mathbf{v}_{\mathrm{old}}\,\ddot{\tau}, \qquad \boldsymbol{\omega}_{\mathrm{new}}=\boldsymbol{\omega}_{\mathrm{old}}\,\dot{\tau}, \qquad \boldsymbol{\alpha}_{\mathrm{new}}=\boldsymbol{\alpha}_{\mathrm{old}}\,\dot{\tau}^2+\boldsymbol{\omega}_{\mathrm{old}}\,\ddot{\tau}$$

随后用原胶囊模型重新求解动力学，不对旧磁场序列简单拉长。

**磁动力学。** 平动：

$$M\,\ddot{\mathbf{p}} = \mathbf{F}_{\mathrm{mag}} + \mathbf{F}_{\mathrm{res}} + M\,\mathbf{g}$$

等效阻力：

$$\mathbf{F}_{\mathrm{res}} = -\mu\,M\,|\mathbf{g}|\,\frac{\mathbf{v}}{\sqrt{\mathbf{v}\cdot\mathbf{v}+\varepsilon^2}}$$

转动：

$$\mathbf{I}\,\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega}) = \mathbf{R}^{\top}\boldsymbol{\tau}_{\mathrm{mag}}$$

磁力与磁力矩：

$$\mathbf{F}_{\mathrm{mag}} = \mathbf{G}^{\top}\mathbf{m}, \qquad \boldsymbol{\tau}_{\mathrm{mag}} = \mathbf{m}\times\mathbf{B}$$

**磁场合成。** 给定需求转矩 $\boldsymbol{\tau}$，可控部分为其垂直分量（单偶极子无法直接实现绕磁轴的转矩）：

$$\boldsymbol{\tau}_\perp = \boldsymbol{\tau} - (\boldsymbol{\tau}\cdot\hat{\mathbf{m}})\,\hat{\mathbf{m}}$$

$$\mathbf{B} = \frac{\boldsymbol{\tau}_\perp \times \mathbf{m}}{|\mathbf{m}|^2} + B_{\mathrm{bias}}\,\hat{\mathbf{m}}$$

实现需求力的最小范数对称无迹梯度，其中 $\mathbf{u}=\mathbf{m}/|\mathbf{m}|$、$a=(\mathbf{F}\cdot\mathbf{u})/|\mathbf{m}|$、$\mathbf{b}=[\mathbf{F}-(\mathbf{F}\cdot\mathbf{u})\mathbf{u}]/|\mathbf{m}|$：

$$\mathbf{G} = a\left(\tfrac{3}{2}\mathbf{u}\mathbf{u}^{\top}-\tfrac{1}{2}\mathbf{I}\right) + \mathbf{b}\mathbf{u}^{\top} + \mathbf{u}\mathbf{b}^{\top}$$

现有参数不改：质量 0.5276 g、磁矩 0.02587 A·m²、摩擦系数 0.2、偏置场 15 mT、重力和初始条件。关节反作用、接触和真实磁源限制没有求解。大范围磁场仍为局部仿射模型外推，不是实际线圈全空间解。箭头长度仅视觉增加 15%，数值和单位不变。

## 四项评价

目标面积 $A_{\mathrm{total}}$ 在实验前固定为完整共享内表面，不随路径完成程度更改。评价 160×120，时间步 0.4 s；通过无抗锯齿、无光照的三角面 ID 离屏渲染获得每个像素首个可见面（包含遮挡）。两组使用相同渲染器，不以播放帧率计分。

有效面需：距离 2–80 mm、面法线与视线夹角不超过 75°、至少 2 个可见像素；连续/容许小间隔的合格观察达到 0.4 s。距离和角度使用面中心近似；角度取绝对余弦，因 STL 法线朝向可能不统一。这是几何质量门槛，不是病灶可辨识验证。

三角形面积（叉积）：

$$S=\tfrac{1}{2}\left|(\mathbf{p}_2-\mathbf{p}_1)\times(\mathbf{p}_3-\mathbf{p}_1)\right|$$

覆盖率：

$$C(T)=100\,\frac{\sum_i A_i\,O_i(T)}{A_{\mathrm{total}}}$$

有效视野占比及其时间平均（梯形积分）：

$$E(t)=100\,\frac{N_{\mathrm{valid\,wall\,pixels}}(t)}{N_{\mathrm{image\,pixels}}}, \qquad E_{\mathrm{mean}}(T)=\frac{1}{T}\int_0^T E(t)\,dt$$

重复观察率（同一面离开有效视野超过 0.8 s 后重新合格观察）：

$$R(T)=100\,\frac{\sum_i A_i\,\max[n_i(T)-1,\,0]}{\sum_i A_i\,n_i(T)}$$

仅相机对照使用当前运动方案相同的机器人位置和时间，只换相机策略；完整系统将原速 Endoscopy 与所选重定时方案比较。四种运动方案 × 两种相机，共八组评价。共同时间预算为原路径时间 286.99953 s，所有组均在这个精确时刻插入评价采样。所有指标预先计算，鼠标悬停、拖动、重复播放不会增加覆盖。

## 复现性与外部依赖

**预计算结果都已包含在内**，所以查看器可以直接运行。但要重跑完整流水线，需要本文件夹之外的输入：

- 解剖数据：`安全地图_路径规划/结果_colon_001_含曲率/colon_model.stl`（以及 `colon_model.mha`）
- 源轨迹 / 基线回放 / 机器人参数 / HTML 模板：`../Magnetic 1 - Field scopes/`
- 物理与求解器：`../Magnetic_shape_comparison/magnetic_model.py`、`../Magnetic 1/build_simulation.py`
- 基线相机与射线遮挡：本机 3D Slicer 5.10.0（`Endoscopy.py`、`PythonSlicer.exe`）

`prepare_geometry.py` 读入 STL，用 VTK 减面（187,206 → 33,696 面；二次误差度量、保体积）后写入 `geometry.json`。

## 资料

- 本机 `Endoscopy.py`：`D:/3dslicer/3D Slicer 5.10.0/lib/Slicer-5.10/qt-scripted-modules/Endoscopy.py`，来源哈希见 `endoscopy_baseline.json`。
- Slicer 5.10 Endoscopy 官方文档：<https://slicer.readthedocs.io/en/5.10/user_guide/modules/endoscopy.html>
- Three.js 官方 r160 固定版本：<https://github.com/mrdoob/three.js/tree/r160>，许可证见 `vendor/THREE_LICENSE.txt`。
- VTK 官方文档：<https://vtk.org/doc/nightly/html/classvtkWindowToImageFilter.html>
- MagRobot: An Open Simulator for Magnetically Navigated Robots —— 借鉴模块化磁场/机器人/观测思路，没有运行 MagRobot 或 SOFA。
- Image-Guided Control of an Endoscopic Robot for OCT Path Scanning，IEEE RA-L，2021，DOI:10.1109/LRA.2021.3087085。
- Constrained Visual Predictive Control of a Robotic Flexible Endoscope With Visibility and Joint Limits Constraints，IEEE RA-L，2025，DOI:10.1109/LRA.2024.3521679。

本实现借鉴「视野约束与执行器限制共同考虑」的思路，以已知三维中心线和肠壁网格代替图像特征；没有实现在线图像反馈、CBF 或完整 MPC。

## 本版限制

- 仅单个结肠病例（colon_001）；差异为描述性，不提供伪统计显著性。
- 前瞻并不保证全部黏膜可见；镜头无额外扫查任务。
- 15°、24°、8 s 和 3% 预算均为工程初值，不能写成文献证实有效的参数。
- 属于离线重放状态校验，**不是**播放时或实机运行时的在线闭环。
- 未建模在线闭环停等、图像清晰度、机器人遮挡镜头、真实镜头畸变。
- 原数据有净空不正的采样点，未通过真实机器人碰撞验证。
- 不保证真实黏膜可见性或临床安全性。

## 许可证

查看器内置了 Three.js r160（见 `vendor/THREE_LICENSE.txt`，MIT）。发布前请为其余代码选择合适的许可证。
