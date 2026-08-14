# 04 训练

这里保存Wuji2 20关节网络的数据加载器、训练入口、验证入口和实时损失监视器。

## 当前状态

两套100场景×256视角数据都已完成Stage03/04并通过20关节加载器合同：一套以
`pre_force_joint_positions_rad`监督q_opt，另一套以`joint_positions_rad`监督
Wuji2 1.0力调整后姿态。两者输入场景和相机完全相同，用于严格A/B，不能混合。

力调整监督版本已经完成50000步训练。独立10场景×10视角测试集也已生成；目前
仍没有单独validation场景族，因此不能报告best-validation checkpoint。

`experiments/wuji2_dexgraspnet2_train60_100seminal_256view_v1_scratch/`只是一份
停在第523步的早期q_opt中断记录，没有checkpoint和`summary.json`，不属于已完成
模型，也不是默认推理来源。目录保留是为了审计当时的损失日志；真正完成的模型
目录名称包含`force_adjusted_legacy_v1`。

## 数据加载器

`wuji2_dataset.py`对每个训练样本执行：

1. 读取一个单视角40000点；
2. 在场景有效抓取库中随机抽128条标签；
3. 把完整表面参考点转换到当前相机系；
4. 查找6 mm以内的最近单视角点；
5. 找到64条可见匹配后停止；
6. 有放回采样64条，得到`rot/trans/qpos/centers`。

主要批张量：

```text
point_clouds  (B,40000,3)
objectness   (B,40000)
graspness    (B,40000)
rot          (B,64,3,3)
trans        (B,64,3)
qpos         (B,64,20)
centers      (B,64)
```

## 训练入口

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  04_training/scripts/train_wuji2_scratch.py \
  --config <最终训练+验证配置.json> \
  --device cuda:0
```

配置中的`paths.output_root`决定实际数据根目录。训练结果只能写到：

```text
04_training/experiments/<experiment_name>_scratch/
```

其中包括：

- `run_config.json`；
- `metrics.jsonl`；
- `validation_metrics.jsonl`；
- `checkpoints/`；
- `summary.json`；
- `.training.lock`。

训练从随机初始化开始。初始化直接调用开源网络构造函数及其原始初始化规则，
固定使用开源随机种子`0`；只有关节输出维度从LEAP Hand的16改为Wuji2的20。
官方checkpoint不会加载，因为其16关节输出层与Wuji2不兼容。脚本在训练前后
校验官方checkpoint哈希，防止误覆盖。

默认训练参数由开源`train_dex_ours.yaml`直接读取：

```text
batch_size       8
max_iter         50000
num_workers      16
Adam lr          0.001
Cosine lr_min    1e-7
grad_clip        10.0
log_every        10
val_every        1000
save_every       5000
seed             0
```

训练入口默认会另起一个CPU绘图进程显示实时损失，不会阻塞反向传播，也不占GPU。
如果在纯终端/无桌面的环境运行，可以添加`--no-live-monitor`。

## 损失监视

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  04_training/scripts/monitor_wuji2_training_loss.py \
  --log 04_training/experiments/<实验>/metrics.jsonl
```

上图显示总损失以及`10×diffusion`（因为开源总损失中的扩散权重就是10）；
下图显示objectness、graspness和joint原始分量。`acc_objectness`只表示点的
物体/背景分类准确率，不表示抓取成功率。窗口使用`TkAgg`，每2秒刷新一次，
关闭窗口不会停止训练。

## 验证和测试

当前没有独立validation场景。现有独立test场景的数据几何、相机输入、姿态标签
和graspness均完整，但它的20关节`qpos`仍是
`pre_force_joint_positions_rad`（q_opt）。最新50000步checkpoint学习的则是
`joint_positions_rad`（Wuji2 1.0力调整后目标）。因此：

- 可以用最新checkpoint对这100个独立视角执行推理、排序、碰撞过滤和Isaac Sim
  物理测试；这些步骤不读取真实`qpos`；
- 不能把该组合产生的`loss_joint`或总损失当作同口径定量测试结果；
- 完整网络损失评估前，需要从同一10场景相机输入派生一套力调整后Stage01–04
  标签，且不得改变场景和点云。

`evaluate_wuji2_checkpoint.py`会读取checkpoint旁的`run_config.json`，比较训练配置
与测试配置的`pose_policy.training_joint_field`。两者不一致时会在加载GPU模型前
主动报错，防止误报告。以后如果生成独立validation场景族，也必须满足同一监督
目标合同，才能用`--split validation`选择checkpoint。
