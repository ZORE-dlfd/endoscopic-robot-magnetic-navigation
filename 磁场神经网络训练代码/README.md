# 磁场与梯度代理训练代码

当前状态：基线模型和中线优先模型均已完成 GPU 训练；独立验证数据尚未接入。

## 本轮范围

本代码实现第一阶段：固定八线圈阵列的局部查询坐标输入，预测各线圈单位电流磁场和梯度。导航修正网络、搜索老师标签和闭环控制器接入属于后续工作。

本机 D:/mag/对应数据包 中的全部有效数据用于训练，不从本机病例、轨迹或邻域点切出验证集。验证数据由另一台设备提供，单独接入。病例是否对应不同患者仍需核对。

## 文件

| 文件 | 作用 |
|---|---|
| field_data.py | 读取已结束的数据包，核对来源、参数、轴序、有限值和磁源内部排除，去除完全重复坐标 |
| field_model.py | MLP、仅基于训练数据拟合的归一化、物理单位预测、模型保存读取 |
| train_field.py | CUDA 训练、日志、检查点、可选外部验证与早停 |
| evaluate_field.py | 外部数据独立评估、基线比较、导数一致性和记录电流重构 |
| train_config.json | 训练配置，尚未执行 |
| data_audit_current.json | 本次只读数据审计快照，后续新增数据不会自动更新此快照 |
| tests/test_interfaces.py | 无优化器、无权重更新的 CPU 接口测试 |

## 数据接口

每例保留完整目录：数据根目录/colon_编号/baseline_30A_long_4h/，含 dataset、config、validation、simulation、status.json。不能只复制 NPZ，参数、来源清单和状态用于检查兼容性。

- 输入 q_local_m[N,3]：阵列局部坐标，单位 m。
- 输出 96 维：前 24 维为逐线圈 8×3 磁场响应，单位 T/A；后 72 维为逐线圈 8×3×3 梯度响应，单位 T/(m·A)。
- 梯度下标为 G[场分量,空间求导方向]。
- 场展开使用 fields.transpose(0,2,1).reshape(N,24)，不能直接展开原 N×3×8。
- 坐标变换：q=R.T@(p-a)，B_world=R@B_local，G_world=R@G_local@R.T。
- 合成与受力：B=fields@I，G=sum(I[k]*gradients[k])，F=G.T@m，torque=cross(m,B)。

入口接收最终状态为 complete 或 incomplete、已生成数据集并通过保存的内部物理重构的病例。默认 include_incomplete=true：导航不完整的病例可提供实际覆盖位置的响应标签，但保留失败标记，不算导航成功。已有 package_validation.json 若失败则拒绝。未单独运行完整包验证的病例明确标记；本入口审计不能代替独立原求解器复核。

仍在写入、输入失败或没有数据集的病例自动跳过。开始训练时冻结来源清单和数据文件 SHA256；开始后新增病例不会自动加入这一轮，纳入新病例需另开运行目录。

磁源参数检查覆盖 design、layout_scale、offsets 和 axes。源码清单统一 Windows/Linux 路径分隔符后比较，不根据不同的原始总哈希直接放行。相同查询点标签冲突则拒绝，完全相同且标签一致的点只保留一次。

## 训练逻辑

初始 MLP：3-128-128-128-96，SiLU，AdamW，float32。输入逐维标准化，磁场/梯度各通道标准化；缩放器只使用训练数据拟合，常量通道设尺度下限。

损失为标准化后的场均方误差与梯度均方误差加权和，默认两组等权。超参数是待验证起点，不代表最佳设置。本版没有添加物理一致性损失，仅在评估中核对磁场导数与梯度输出，不是 PINN/FNO。

正式配置明确指定 CUDA。CUDA 不可用则报错，不自动改为 CPU。数据保留在 CPU 内存，每批送入 GPU。复用现有 F:/Anocanda/envs/pytorch/python.exe；本次已确认 PyTorch 2.7.1+cu118、RTX 2080 Ti 可用。

没有外部验证集时，按配置训练至固定轮数并保存 last.pt；不生成 best.pt，不按训练误差宣布验证成功。有外部验证时才按其损失保存 best.pt、执行早停。

检查点携带模型、归一化、轴序、训练病例、磁源/源码签名、训练范围和配置。保存优化器状态供后续扩展；本版未提供断点续训命令。

## PowerShell 命令

以下训练命令仅作未来使用说明，本次未执行。

只读审计，不建立模型或优化器：

    & 'F:\Anocanda\envs\pytorch\python.exe' -B 'D:\mag\训练代码\train_field.py' --config 'D:\mag\训练代码\train_config.json' --audit-only

接口测试，不更新权重，临时文件在本目录 test_artifacts 下自动清理：

    $env:TEMP='D:\mag\对应数据包\tmp'
    $env:TMP=$env:TEMP
    & 'F:\Anocanda\envs\pytorch\python.exe' -B -m unittest discover -s 'D:\mag\训练代码\tests' -p 'test_*.py' -v

将来确认开始训练后才运行：

    $env:TEMP='D:\mag\对应数据包\tmp'
    $env:TMP=$env:TEMP
    & 'F:\Anocanda\envs\pytorch\python.exe' -B 'D:\mag\训练代码\train_field.py' --config 'D:\mag\训练代码\train_config.json'

默认输出 D:/mag/训练结果/field_mlp_v1，已存在则拒绝覆盖。输出包括 status.json、history.jsonl、last.pt、可选 best.pt、normalizer.json、config.json、data_audit.json、source_sha256.json 和 training_reference.npz。训练参考数组用于评估重叠和简单基线，需随模型迁移。

## 另一台设备的验证数据

当前 validation_root=null，因为验证数据尚未接入本机。可以在训练开始前将其设为外部数据根目录，保留相同包结构；也可训练后把模型、training_reference.npz 和本代码复制到验证设备运行独立评估。

验证病例与训练病例重叠则拒绝；磁源/源码不一致则拒绝。距训练查询点不超过 1 微米的验证样本会排除并报告，阈值可提前配置；全部重叠则拒绝。患者身份无法只靠编号自动判定。

外部病例留出不等于空间外推，因为固定阵列场函数相同。报告给出距训练点的距离分组与训练包围盒外数量，不是严格空间分块留出。外部验证若用于选择模型，不能再称为独立最终测试。

外部评估示例，data-root 替换为验证设备真实路径：

    & 'F:\Anocanda\envs\pytorch\python.exe' -B 'D:\mag\训练代码\evaluate_field.py' --checkpoint 'D:\mag\训练结果\field_mlp_v1\last.pt' --data-root 'D:\mag\外部验证数据' --output 'D:\mag\训练结果\field_mlp_v1_external_eval' --device cuda

评估不更新缩放器或权重。指标含物理单位 RMSE/MAE/最大绝对误差/相对 L2、逐病例与逐线圈误差、最近邻和训练均值基线、空间距离分组及自动微分 dB/dq 与直接梯度输出的一致性。相对误差公式为 norm(error)/max(norm(reference), floor*sqrt(size))，不是逐点相对误差均值。

有保存轨迹时按实际电流、阵列旋转与磁矩重构场、梯度、力和力矩误差；与训练位置重叠的轨迹点同样排除。它复用原轨迹，不是接入网络重新跑闭环，不能证明网络改变电流决策后仍导航成功。

## 当前限制

本次仅检查接口和只读数据，未运行正式训练或 GPU 试训。测试使用随机初始模型/解析夹具，不产生训练模型。

误差容限、弱场分母、时延预算和轨迹容差需在最终测试前冻结。尚未实现与原求解器的单点/批量测速，未接入在线回退或闭环控制器，因此不会宣布模型通过或获得加速。

预测入口默认拒绝训练坐标包围盒外的点；盒内不代表可靠或无磁源。在线使用前还需原求解器复核、源区域排除、残差与约束检查及可靠回退。标签来自声明的有限磁源模型，不是独立 FEM 或实测真值。
