# 内镜机器人八线圈磁导航

肠道磁导航仿真、磁场代理网络和内镜深度估计的代码与配套资料。

## 下载即用

前往 [最新 Release](https://github.com/ZORE-dlfd/endoscopic-robot-magnetic-navigation/releases/latest)，下载 `Windows-Portable-Heatmap-v3.0.0.zip`（若为分卷，下载该名称的全部 `.partNNN` 文件及 `Join-ReleaseParts.ps1`）。完整解压后，进入“软件”目录，双击 `Magnetic_Camera_8Coil.exe`，点击“播放”或按空格。

软件内含八线圈与机器人运动回放、内镜视角，以及 **6037 帧模型预测深度缓存**。热力图随播放和时间轴更新，无需安装 Python、CUDA、模型权重或启动深度服务。请保留 EXE 同级的 `_internal` 与 `缓存数据`，不要直接在压缩包中运行。运行环境为 Windows 64 位，显卡驱动需支持 OpenGL 3.3。

只看演示可下载 `Navigation-Heatmap-Demo.mp4`。

## 五个板块

| 板块 | 用途 | 获取方式 |
|---|---|---|
| [总仿真软件+代码](总仿真软件+代码) | 八线圈磁场、机器人进退镜、相机与深度热力图；含源码、回放资产、构建工具和深度服务 | 代码在仓库；完整 Windows 软件在 Release |
| [磁场神经网络训练代码](磁场神经网络训练代码) | 训练与评估 MLP，以局部三维坐标预测八线圈单位电流磁场和梯度；含中线优先加权训练 | 仓库 |
| [肠道磁场数据包](肠道磁场数据包) | 肠道掩膜、各病例磁响应样本、仿真配置及验证记录，供磁场网络训练与复核 | Release 的 `Magnetic-Field-Dataset-v3.0.0` |
| [深度模型代码](深度模型代码) | EndoStreamDepth 深度模型与训练代码，供内镜深度估计和深度服务使用 | 仓库 |
| [深度模型预训练权重](深度模型预训练权重) | `iter_20001_C3VD.pth`，用于搭建深度推理或训练环境；观看缓存无需下载 | Release 的 `Depth-Pretrained-Weights-v3.0.0` |

## 开发与资料

- 仿真入口：`总仿真软件+代码/源码/main.py`；磁场训练入口：`磁场神经网络训练代码/train_field.py`。具体配置见各目录说明，本地绝对路径需按解压位置调整。
- `Source-Snapshot-v3.0.0.zip` 保存本次源码与配套资料快照；与软件、数据、权重包合起来对应本次提供的全部文件。逐文件分发位置见 [资料分发清单](资料分发清单.json)。
- 超过单附件限制的 ZIP 按字节分卷。把同名全部分卷与 `Join-ReleaseParts.ps1` 放在同一目录，在 PowerShell 运行 `powershell -ExecutionPolicy Bypass -File .\Join-ReleaseParts.ps1`，合并后再解压 ZIP。校验值见 Release 的 `SHA256SUMS.txt`。
- 深度缓存用于离线回放；重新推理需配置深度服务与权重。保留 EndoStreamDepth、Mamba 等第三方代码的原有许可证与来源说明，软件第三方组件见包内说明。
