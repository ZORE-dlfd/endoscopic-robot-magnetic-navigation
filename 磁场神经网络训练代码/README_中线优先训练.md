# 中线优先磁场与梯度训练说明

这份说明用于让其他 AI 或开发者快速理解本次修改。代码目录是 `D:/mag/训练代码`，训练数据目录是 `D:/mag/对应数据包`，模型输出目录是 `D:/mag/训练结果`。

## 1. 任务目标

模型学习固定八线圈磁场阵列的局部响应函数：

- 输入：阵列局部坐标 `q_local_m[N, 3]`，单位为米。
- 输出：96 个数。
- 前 24 个数：8 个线圈各自的三维单位电流磁场，单位 `T/A`。
- 后 72 个数：8 个线圈各自的 `3 x 3` 磁场梯度，单位 `T/(m·A)`。
- 梯度下标顺序：`G[场分量, 空间求导方向]`。

模型只预测每个线圈单位电流的响应；实际机器人运行时，再使用实际电流、阵列姿态和磁矩合成世界坐标系磁场、梯度、磁力和力矩。

## 2. 这次改了什么

没有改变神经网络结构。网络仍然是：

```text
3 -> 128 -> 128 -> 128 -> 96
```

激活函数是 SiLU，优化器是 AdamW，训练使用 CUDA/float32。

改变的是样本损失权重。所有样本仍然参加训练，但让实际导航中更重要的区域对梯度更新影响更大：

| sampling_kind | 含义 | 权重 |
|---|---|---:|
| `trajectory` | 机器人实际进镜/退镜轨迹点 | 3.0 |
| `neighbor` | 轨迹附近的邻域点 | 2.0 |
| `workspace` | 随机工作空间点 | 0.25 |

每个小批次内，样本权重会先除以当前批次权重均值，再计算标准化磁场损失和标准化梯度损失。这样改变的是优化重点，不是删除随机工作空间数据。

## 3. 具体改动文件

- `field_data.py`：读取并保留每个样本的 `sampling_kind`，同时在审计中统计三类样本数量。
- `field_model.py`：`response_loss()` 增加可选的逐样本权重。
- `train_field.py`：从配置读取 `sampling_weights`，将权重用于训练和可用的外部验证损失。
- `train_config_centerline.json`：中线优先训练配置。
- `diagnose_centerline_fit.py`：按采样类别和病例统计训练集拟合误差。
- `tests/test_interfaces.py`：增加样本加权损失接口测试。
- `README_中线优先训练.md`：本说明文件。

## 4. 数据和训练配置

本轮使用 10 个已完成数据包中的全部有效样本：

- 总样本：162,272
- 轨迹点：40,670
- 邻域点：81,356
- 随机工作空间点：40,246
- `include_incomplete=true`：导航未完成病例的有效磁响应标签仍可纳入训练，但不代表导航成功。
- 没有从本地数据切出验证集。

配置文件：`D:/mag/训练代码/train_config_centerline.json`

主要配置：

- GPU：`cuda`
- GPU：NVIDIA GeForce RTX 2080 Ti
- 训练轮数：500
- batch size：2048
- 学习率：0.001
- 场损失权重：1.0
- 梯度损失权重：1.0
- 样本权重：trajectory=3.0、neighbor=2.0、workspace=0.25

## 5. 如何启动训练

PowerShell 命令如下。临时目录也指定到 D 盘：

```powershell
$env:TEMP='D:\mag\对应数据包\tmp'
$env:TMP=$env:TEMP
& 'F:\Anocanda\envs\pytorch\python.exe' -B `
  'D:\mag\训练代码\train_field.py' `
  --config 'D:\mag\训练代码\train_config_centerline.json'
```

只做数据审计、不训练：

```powershell
& 'F:\Anocanda\envs\pytorch\python.exe' -B `
  'D:\mag\训练代码\train_field.py' `
  --config 'D:\mag\训练代码\train_config_centerline.json' --audit-only
```

打开训练进度窗口：

```powershell
Start-Process -FilePath 'F:\Anocanda\envs\pytorch\python.exe' `
  -ArgumentList @('-B','D:\mag\训练代码\show_training_progress.py','--folder','D:\mag\训练结果\field_mlp_centerline_v1')
```

## 6. 输出文件

中线优先模型目录：`D:/mag/训练结果/field_mlp_centerline_v1`

- `last.pt`：训练完成的模型检查点。
- `config.json`：实际使用的配置副本。
- `data_audit.json`：训练数据和病例审计。
- `history.jsonl`：每一轮训练损失。
- `status.json`：训练状态、GPU 和耗时。
- `normalizer.json`：输入和输出归一化参数。
- `training_reference.npz`：训练坐标、标签、病例和采样类别，评估时需要保留。
- `fit_diagnostics.json`：按病例和采样类别的训练集拟合结果。
- `对比结论.md`：与基线模型的简要对比。

基线模型目录：`D:/mag/训练结果/field_mlp_v1`

两套模型必须保留，不能覆盖。当前中线模型不是全工作空间模型的替代品；如果查询点超出训练范围，应用层应回退原始磁场求解器。

## 7. 当前结果如何理解

中线优先模型在训练数据上的结果：

- 轨迹磁场相对 L2 误差：约 9.6%。
- 轨迹梯度相对 L2 误差：约 24.2%。
- 轨迹邻域结果与轨迹点接近。
- 随机工作空间磁场相对 L2 误差：约 37.4%。
- 随机工作空间梯度相对 L2 误差：约 64.9%。

这表示模型更集中地拟合了实际导航中线附近区域。上述结果是训练集拟合，不是对新病例的独立验证。

## 8. 外部验证方法

另一台设备提供验证数据后，验证数据必须保留同样的病例包目录结构，并放在新的数据根目录。然后用：

```powershell
& 'F:\Anocanda\envs\pytorch\python.exe' -B `
  'D:\mag\训练代码\evaluate_field.py' `
  --checkpoint 'D:\mag\训练结果\field_mlp_centerline_v1\last.pt' `
  --data-root 'D:\mag\外部验证数据' `
  --output 'D:\mag\训练结果\field_mlp_centerline_v1_external_eval' `
  --device cuda
```

对基线模型重复一次，只替换 `--checkpoint` 和 `--output`：

```text
checkpoint = D:/mag/训练结果/field_mlp_v1/last.pt
```

两套模型必须在同一批外部验证样本上比较：

- 单线圈磁场误差。
- 单线圈梯度误差。
- 使用实际电流合成后的磁场和梯度误差。
- 磁力误差。
- 按病例、按轨迹附近距离分组的误差。
- 自动微分得到的 `dB/dq` 与直接梯度输出的一致性。

只有外部验证集上中线附近误差持续优于基线，才能认为鲁棒性提高。训练集误差下降本身不能证明新肠道或新中线泛化成功。

## 9. 当前限制

- 当前没有独立验证数据。
- 当前没有重新运行网络驱动的闭环导航。
- 当前没有硬件实测磁场验证。
- 中线优先权重可能降低远离中线的工作空间精度。
- 训练坐标包围盒内不等于每一点都可靠；接近磁源区域的积分标签还需要数值稳定性检查。
- 若新中线明显偏离已有轨迹分布，不能仅凭本轮训练误差作部署结论。
