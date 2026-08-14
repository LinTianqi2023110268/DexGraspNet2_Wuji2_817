# Wuji2官方USD迁移与鼠标/电话亭抓取失败审计

更新时间：2026-08-04  
范围：方案审计、代码迁移和静态验证；本轮没有启动第二个Isaac Sim进程，也没有把未运行的方案称为成功。

## 1. 结论先行

师兄提出“先换官方USD，再查动力学和表面参数”是正确方向，而且现有证据表明这不是小改动。

此前GUI脚本实际加载的是旧URDF派生的、再次修补并展平的USD：

```text
.../usd_cache/4822e27c496e_6a9c2c4aa668_editable_collision_v1/
  flat/wuji2_hand_editable.usd
```

从本次迁移开始，Isaac Sim唯一允许的Wuji2手运行资产是：

```text
01_environment/vendor/wuji-description/
  hand2/hand2_beta1/body/usd/right/wujihand2.usd
```

上游提交固定为：

```text
8271644a78d69ed9a4adcf9165d882c64ad33dfa
```

但“换USD”不能等同于“把路径字符串换掉”。至少还有三项必须同时处理：

1. 旧数据给的是`T_world_r_base_link`，官方USD根是`r_wrist`，必须转换根位姿；
2. 旧脚本会在运行时覆盖USD的关节驱动、自碰撞和接触参数，不关闭这些覆盖就没有测试官方USD；
3. DexGraspNet 2.0官方评估器把手根的6维位姿作为有限位置驱动DOF，而当前Isaac Sim脚本每个物理回调都用`set_world_pose()`直接改根状态。后者会产生非物理的推挤和接触冲量，是鼠标靠近手、物体被弹开的一项高风险来源。

因此，本次已经完成“官方资产部署、根坐标兼容、官方原生动力学基线入口”；下一轮物理实验应先做受控A/B，而不是继续盲调SQUEEZE幅度。

### 1.1 迁移边界：不能把旧标签改名冒充官方数据

当前完整训练集的单物体优化、六方向验证、手部SDF和场景碰撞标签，都是用旧`r_base_link`模型生成的。它们是不可变的历史证据，不会因为Isaac Sim改为加载官方USD就自动变成“官方USD标签”。本轮采取两层合同：

- 新的Isaac Sim抓取回放：只允许官方USD；
- 现有`q_opt`和根位姿：作为legacy兼容输入，先按关节名映射并转换到官方`r_wrist`，再重新做官方碰撞/物理验证；
- 新生成的数据集：目标是使用同一提交的官方URDF做离线FK，并以官方碰撞几何重新筛选；
- 旧Stage02/03和旧训练数据不静默覆盖，否则旧checkpoint与标签的来源会被破坏。

所以“运行时官方化”已经完成；“训练标签全部官方化”需要重建数据标签，是后续独立任务，不能在本轮方案审计中假装已经完成。

## 2. 官方USD与原URDF/旧USD究竟有什么不同

### 2.1 文件角色不同

URDF主要描述：

- 连杆树和固定/转动关节；
- 关节轴、上下限、URDF `effort`与`velocity`；
- 质量、惯量；
- visual/collision网格引用。

USD还能直接描述Isaac Sim/PhysX运行语义：

- `PhysicsArticulationRootAPI`放在哪个prim；
- 位置驱动的stiffness、damping、maxForce；
- articulation solver迭代数；
- 是否启用self-collision；
- 哪些连杆对被过滤；
- convexHull等PhysX碰撞近似；
- 分层引用、实例化、物理材质绑定。

所以，官方USD不是“官方URDF由我们本地再转一次”的同义词。应直接加载官方发布的分层USD；配套URDF只供PyTorch FK、关节表和离线几何代码使用。

### 2.2 本机直接读取组合后USD得到的数值

| 项目 | 官方Hand2 Beta1 USD | 旧GUI运行USD |
|---|---:|---:|
| 根刚体 | `r_wrist` | `r_base_link` |
| 20关节stiffness | 全部50 | 全部约17.4533 |
| 20关节damping | 全部2 | 全部0 |
| 20关节maxForce | 全部2 N·m | 0.2/0.3/0.6/2.0 N·m分级 |
| self-collision | 开启 | 关闭 |
| articulation solver | 32/1 | 32/1 |
| 碰撞 | 21个convexHull | 22个convexDecomposition |
| 刚体数 | 26（含5个tip查询刚体） | 21 |
| 有质量连杆总质量 | 约0.6207 kg | 约0.6897 kg |
| 物理摩擦材质 | 未写入 | 未写入 |

这里以`wujihand2.usd`组合后的实际属性为准。官方目录里的`config.yaml`仍记录转换时的另一组0.8～2 stiffness、0.03～0.05 damping，但最终发布的`configuration/wujihand2_physics.usd`实际写入的是50和2。运行时真正生效的是组合后USD，而不是旁边的生成配置说明。

官方README还明确说明Beta限制：`*_tip.STL`软指腹网格尚未挂成碰撞几何。因此：

- `r_*_tip`适合做指尖位置/IK查询点；
- 真正产生接触的是前一级`*_distal`碰撞壳；
- “指尖点到达目标”不保证实体指腹已经形成正确接触；
- 传感器应明确写成“末端碰撞连杆力”，不能误称tip marker本身的力。

### 2.3 为什么仅换USD仍可能看不到变化

旧导入器在`world.reset`后执行：

```python
controller.set_gains(...)
controller.set_max_efforts(...)
self_collision_attr.Set(False)
```

电话亭复现脚本还指定800 stiffness、20 damping以及旧URDF分级力矩。也就是说，即使把`HAND_USD`换成官方路径，只要继续运行这些语句，官方驱动与自碰撞仍会立刻被覆盖。

本次新增`wuji_hand2_official_usd_native`配置，第一轮A/B中：

- 不覆盖官方50/2/2 N·m；
- 不关闭官方self-collision和官方filtered pairs；
- 不覆盖官方手部convexHull碰撞设置；
- 物体的质量/碰撞参数仍单独记录；
- 手、物体、桌面的物理摩擦材质仍显式设置，因为官方手USD没有提供它。

## 3. 坐标系迁移：旧优化位姿怎样输入官方USD

### 3.1 旧模型和官方模型的根不同

旧数据保存：

```text
T_world_r_base_link
```

旧URDF内部还有固定关节：

```text
r_base_link -> r_wrist
translation = [0.000250158, 0.003000042, 0.028499851] m
```

官方Hand2 Beta1直接以`r_wrist`为根，而且冻结了新的轴表达。两套wrist坐标的轴转换为：

```text
C = T_legacy_wrist_from_official_wrist
  = [[0, 1,  0, 0],
     [1, 0,  0, 0],
     [0, 0, -1, 0],
     [0, 0,  0, 1]]
```

因此，旧优化根位姿进入官方USD前必须计算：

```text
T_world_official_wrist
  = T_world_legacy_base
  @ T_legacy_base_legacy_wrist
  @ T_legacy_wrist_from_official_wrist
```

不能直接把旧4×4矩阵交给官方`r_wrist`。

### 3.2 20关节值是否还能用

可以按关节名迁移，但不能按Isaac导入后的裸索引硬拷贝。

两版模型：

- 都有相同语义的20个关节名；
- 上下限和URDF effort/velocity按关节名完全一致；
- 关节局部轴表达改变，但官方坐标重表达后代表同一个物理转动。

已用5组随机20关节姿态做FK对照。完成上述wrist轴变换后，所有公共连杆原点的最大位置差小于0.0059 mm。因此旧`q_opt`可作为官方模型的兼容输入，前提是：

1. 根位姿使用上面的转换；
2. q按关节名重排；
3. 仍需重新做官方碰撞壳下的自碰撞、桌面/场景碰撞和物理验证。

原因是“运动学一致”不代表“碰撞几何完全一致”：旧运行USD是convexDecomposition，官方是convexHull。

## 4. 两阶段Isaac Sim脚本到底怎样完成抓取

### 4.1 NPZ任务文件输入了什么

任务文件的核心不是一个“抓取姿态”，而是两条同步轨迹：

```text
waypoint_pose_world       : (candidate, stage, 4, 4)
waypoint_joint_positions : (candidate, stage, 20)
joint_order              : 20个语义关节名
waypoint_names           : PREGRASP/COVER/GRASP/SQUEEZE/LIFT
```

其中优化结果进入GRASP：

```text
GRASP root = 优化后的手根4×4世界位姿
GRASP q    = 优化后的20关节q_opt
```

PREGRASP/COVER是张开手和接近轨迹；SQUEEZE是由q_opt继续计算出的关节目标；LIFT保持SQUEEZE关节目标并改变手根位置。

新的任务生成器会显式写入：

```text
hand_root_pose_frame = legacy_r_base_link
```

导入器看到这个字段后转换成官方`r_wrist`。以后新生成的官方根任务也可以直接写`official_r_wrist`，不会发生二次转换。

### 4.2 第一步脚本：导入和校验

`01bg...py`是薄包装器，负责固定任务、物理配置和实验身份；真正导入由`01_import_scene_assets.py`完成：

1. 校验任务NPZ、JSON、场景manifest和对象身份；
2. 校验官方USD/URDF的SHA256，防止资产被无意改动；
3. 创建World、重力、桌面和物理材质；
4. 引用官方`wujihand2.usd`；
5. 把旧`T_world_r_base_link`轨迹转换为官方`r_wrist`轨迹；
6. 建立`SingleArticulation`；
7. 导入场景物体并恢复稳定场景位姿；
8. 根据`joint_order`把20维q重排到Isaac articulation DOF顺序；
9. 创建每根手指到“选定目标物体”的`RigidContactView`；
10. 用PREGRASP根位姿和q初始化手；
11. 把所有句柄保存到`builtins.WUJI2_MANUAL_GRASP_CONTEXT`，暂停等待第二步。

关键Isaac命令及含义：

```python
add_reference_to_stage(HAND_USD, "/World/Wuji2Hand")
# 把官方USD引用到当前场景

hand = SingleArticulation(prim_path=articulation_root_path, ...)
# 建立可控制的20关节articulation句柄

hand.set_world_pose(position=..., orientation=...)
# 直接写手根世界状态；这不是关节力控制

hand.set_joint_positions(q)
# 直接初始化关节状态，仅适合reset/初始化

hand.apply_action(ArticulationAction(joint_positions=q_target))
# 给有限位置驱动发送目标；PhysX依据K/D/maxForce追踪

RigidContactView(... filter_paths_expr=[[target_rigid_path]])
# 只测该手指与指定目标物体之间的接触力
```

### 4.3 第二步脚本：逐物理步执行

`02bg...py`选择连续模式和时间缩放；`02_execute_grasp_on_loaded_scene.py`注册physics callback。

每个物理回调中：

1. 根据当前段计算插值比例；
2. 根平移线性插值，根旋转用四元数SLERP；
3. 20关节目标在相邻waypoint之间插值；
4. 写根位姿；
5. 用`ArticulationAction(joint_positions=target_qpos)`发送关节目标；
6. PhysX前进一步，生成接触、摩擦和物体运动；
7. 到端点后保持，记录关节误差、物体速度和接触力；
8. SQUEEZE后恢复目标物体重力；
9. LIFT结束后检查目标上升高度和是否跟随手移动。

官方DexGraspNet 2.0原始评估器的对应代码是：

```python
target_qpos = start + alpha * (end - start)
simulator.set_actor_actions("robot", target_qpos)
simulator.step()
```

但官方`robot_name + '_free'`把3平移+3旋转也放进DOF目标，整只手由有限驱动追踪26维目标。我们的当前脚本则固定根articulation并每步调用`set_world_pose()`。两者不是同一种接触动力学。

## 5. 原抓取脚本可能存在的问题

### P0：旧根位姿直接给新根会错位——已修

症状：手看起来接近目标，但掌心、指尖和优化时的实体关系整体旋转/偏移；接触点与Trimesh不一致。

处理：加入显式`r_base_link -> official r_wrist`转换、资产哈希和运行时frame记录。

### P0：运行时参数覆盖掩盖官方USD——已建立原生基线

症状：换了USD但抓取行为几乎没变化，或仍看到800/20、旧力矩、自碰撞关闭。

处理：官方原生profile拒绝覆盖50/2/2 N·m、self-collision和convexHull。

### P0：手根每步直接改状态，而不是有限驱动——尚待第二阶段解决

症状：接近时目标被手“吸过来/推走”；接触瞬间冲量大；减小时间步只能缓和，不能修复根控制语义。

建议：

1. 第一轮仍保留`fixed_teleport`，只用于隔离比较旧/新手资产；
2. 第二轮建立官方同构的6自由度虚拟腕：3个prismatic + 3个revolute，连同20手指关节一起用有限position drive；
3. 或把官方`r_wrist`改为free articulation，用显式6D PD/velocity servo，但必须记录位置/旋转误差与最大根力；
4. 真机阶段则由机械臂控制腕位姿，手控制器只接收20关节目标。

不要把`set_world_pose()`在接触阶段误解成机械臂“保持位姿”。它是直接状态写入。

### P1：官方self-collision与旧SQUEEZE目标可能冲突

旧脚本为避免抖动直接关掉手内碰撞；官方USD开启self-collision并带有上游filtered pairs。旧30 mm IK结果可能在旧碰撞模型中可用、在官方convexHull中自碰撞。

建议在仿真前逐waypoint检查：

- PREGRASP、COVER、GRASP、SQUEEZE的手内碰撞对；
- q是否超过限制；
- 指尖目标误差；
- GRASP到SQUEEZE沿线，而不只检查两个端点。

如果官方self-collision导致初始爆炸，应判该q/路径无效，而不是先全局关闭自碰撞。只有明确的装配重叠对可以按官方filtered pairs过滤。

### P1：物理指腹不是`*_tip`查询点

官方Beta 1没有把tip软垫挂成碰撞。当前接触传感器正确地回退到`*_distal`，但WidthMapper仍在移动tip查询点。这可能造成：

- 数学指尖移动30 mm，实体碰撞壳实际移动方向/距离不同；
- 指尖达到目标但没有拇指-对侧指的稳定接触；
- 某些细长物体被distal侧面向下压桌面。

需要同时绘制/记录tip查询点与distal真实接触点，不能只看关节目标。

### P1：摩擦不是官方USD自带参数

官方手USD没有PhysicsMaterial。现在电话亭基线显式使用：

```text
hand friction = 0.2
object friction = 1.0
table friction = 1.0
restitution = 0
object mass = 0.1 kg
```

这些是DexGraspNet 2.0复现设置，不是Wuji2官方USD的“表面真值”。师兄要求查表面参数时，应该按实验矩阵测，而不是凭感觉把摩擦改到3：

- 手指材料0.2/0.5/1.0；
- 物体材料来自实际材质估计或统一1.0基线；
- 同时记录静摩擦与动摩擦，不能只设一个含糊的`friction`；
- 恢复系数先固定0，避免碰撞弹飞；
- 记录combine mode，避免PhysX用不同合并规则。

### P1：开环SQUEEZE只追位置，不保证力闭合

当前电话亭策略：30 mm tip-normal IK，然后开环位置控制。它没有保证：

- 拇指与至少一根对侧手指同时接触；
- 接触法向构成抗重力的力闭合；
- 切向摩擦裕度大于物体重力；
- 中指/无名指没有持续向桌面施力。

现有执行器已经能读取每根手指的目标过滤接触力向量。下一步应保留原始q路径，但把SQUEEZE/LIFT改成：

1. 缓慢到达名义SQUEEZE；
2. 每指记录接触力、方向、关节误差；
3. 至少确认“拇指 + 一个对侧手指”持续接触；
4. 对持续向下压桌面的手指停止加深或略回退；
5. 抬升中若某一侧失去接触，仅在原SQUEEZE方向小幅增加有限目标，不直接锁死关节状态；
6. 所有目标仍受官方2 N·m maxForce约束。

### P2：时间步不是唯一原因

从1/120减到更小并增加substep可以改善高速穿透和数值稳定，但不能修复：

- 坐标系错误；
- 根状态直接写入；
- 错误的碰撞壳；
- 没有对向接触；
- 摩擦不足；
- 不可达的30 mm IK目标。

所以顺序应是先修P0/P1，再做1/120与1/240的时间步A/B。

## 6. 鼠标案例的针对性判断

已审计的旧成功候选身份：

```text
scene=0000
target segmentation=2
source candidate=2000695
GRASP=q_opt
旧主方案SQUEEZE≈q_grasp + 0.90*(q_squeeze-q_grasp)
continuous gravity
minimum-jerk
抬升前速度清零
```

主要风险：

1. 所谓旧成功仍使用旧URDF派生USD，不是官方Hand2 USD；
2. 旧导入配置使用1000 stiffness、50 damping和分级小力矩，不能回答官方50/2/2 N·m是否更好；
3. 任务曾有6个waypoint，GRASP→SQUEEZE的20关节L2变化约1.477 rad，属于相当大的开环闭合；
4. 鼠标薄且低，根直接下压、distal接触和桌面接触容易共同把它横向推出；
5. “抬升前速度清零”能减小瞬态，但不能补回已经丢失的对向夹持。

推荐鼠标A/B顺序：

```text
M0  官方USD原生驱动 + 旧q_opt + 不SQUEEZE，检查几何/接触初态
M1  官方USD原生驱动 + 原0.90闭合轨迹 + fixed_teleport（只隔离资产）
M2  M1 + 6DOF有限腕驱动（替代接触阶段set_world_pose）
M3  M2 + 接触调节SQUEEZE，目标为拇指-食/中指对向接触
M4  仅在M3稳定后比较摩擦0.2/0.5/1.0与dt 1/120、1/240
```

鼠标不应首先增加SQUEEZE幅度；薄物体更可能因过闭合被挤走。

## 7. 电话亭案例的针对性判断

当前审计身份：

```text
scene=0033
target segmentation=6
source candidate=6000173
GRASP=Wuji2 1.0 q_opt
SQUEEZE=30 mm、keep_z、20步全关节IK
GRASP→SQUEEZE关节L2变化约0.698 rad，单关节最大约0.313 rad
物体重力在SQUEEZE前被抵消，SQUEEZE后恢复
world +Z抬升200 mm
```

主要风险：

1. 30 mm是五个查询tip各自目标，不是实体夹持宽度缩小30 mm；
2. `keep_z=True`清零的是手根坐标中的z分量，不是世界竖直方向，不保证“不压桌面”；
3. upright电话亭细长，单侧接触很容易形成转矩并滑出；
4. 重力从关闭到开启与LIFT连接处，会突然出现完整物重；若SQUEEZE只有位置目标而没有稳定对向预载，物体会立刻滑落；
5. 旧根直接写状态在高物体上产生的力臂更大，轻微位姿偏差也会把物体推倒；
6. 官方self-collision恢复后，旧30 mm目标可能不可达，需要先看关节跟踪误差而不是继续加时间。

推荐电话亭A/B顺序：

```text
P0  官方USD原生驱动，停在GRASP，观察真实distal接触与self-collision
P1  30 mm名义SQUEEZE，开启五指实时力曲线，但不抬升
P2  若拇指/对侧没有同时接触，降低或分指调整IK目标，不增大整体闭合
P3  使用6DOF有限腕驱动重新执行PREGRASP→COVER→GRASP
P4  SQUEEZE后先在重力下保持0.5～1 s，接触稳定后再慢速抬升
P5  抬升中使用有限接触维持，不使用`set_joint_positions`硬锁
```

## 8. 师兄提出的“测接触力、摩擦力”怎样落地

现有脚本已经为每根手指创建目标过滤contact view，并能读取：

```python
matrix = view.get_contact_force_matrix(dt=step_size)
force_on_finger_world = matrix.sum(axis=(0, 1))
force_on_object_world = -force_on_finger_world
```

当前实时窗口主要画每指力的模长和；结果JSON还保存当前/EMA世界力向量、最大力和向下压物体的分量。下一版诊断应补齐：

- 接触点世界坐标；
- 接触法向；
- 法向力`F_n`；
- 切向力`F_t`；
- 摩擦裕度`mu*F_n - |F_t|`；
- 接触点相对滑移速度；
- 每个关节的目标角、实际角、误差、估计驱动力矩；
- 手根目标/实际6D误差；
- 目标物体位置、姿态、线速度、角速度；
- 手内碰撞对和冲量。

判断抓取时不能只看“总力大”。一个手指向下70 N可能很大，却是在把物体压进桌面；真正需要的是拇指和对侧手指的方向对抗、摩擦锥裕度以及抬升阶段持续接触。

## 9. 推荐的最终控制结构

### 仿真复现层

```text
官方wujihand2.usd
  + 官方原生关节drive/self-collision/collider
  + 显式手/物/桌物理材质
  + 6DOF有限腕驱动
  + 20关节有限position drive
  + 目标过滤接触力诊断
```

### 抓取动作层

```text
PREGRASP：虎口方向后退，手指张开
    ↓
COVER：有限腕驱动接近，不直接瞬移根
    ↓
GRASP：到q_opt，保持并确认物体未被推走
    ↓
SQUEEZE：沿已定义的Wuji2指腹方向缓慢闭合
         每指根据目标接触力独立停止/回退
    ↓
SETTLE：恢复重力后保持，确认拇指+对侧接触
    ↓
LIFT：腕沿世界+Z慢速抬升，有限关节驱动维持接触
```

### 真机映射层

真机可以实现同一思想：机械臂负责6D腕轨迹，Wuji2控制器负责20关节位置/电流或力矩限制。仿真中的`set_world_pose()`和`set_joint_positions()`硬写状态不能直接视为真机命令；真机应使用控制器目标与反馈闭环。

## 10. 本轮已经落地的文件

- 官方仓库：`01_environment/vendor/wuji-description/`
- 资产与坐标合同：`config/wuji2_official_asset.json`
- 旧根到官方根转换：`src/wuji2_dgn2/official_asset.py`
- 只读USD审计：`01_environment/audit_wuji2_usd.py`
- 官方USD审计结果：`01_environment/wujihand2_official_usd_audit.json`
- 旧运行USD审计结果：`01_environment/wuji2_legacy_runtime_usd_audit.json`
- 当前LEAP迁移成功路线：`06_leap_to_wuji2_final_pipeline/`
- 当前原生Wuji2网络路线：`07_wuji2_network_3p3r_sim/`
- 两条路线均采用3P+3R手根；旧06直接位姿手根执行器在合并后移除。
- 旧电话亭及A～Z对照入口已在最终整理时删除，不再是可运行工程入口。

## 11. 下一次实际试验的停止条件

下一步不要一次改变五个参数。每一轮只改变一项，并保存完整结果JSON：

1. 先在GRASP停住，资产/坐标/self-collision必须正确；
2. 再只做SQUEEZE，不抬升，确认对向接触；
3. 再恢复重力保持；
4. 最后抬升；
5. 成功至少同时满足：目标上升超过30 mm、仍跟随手、没有其他物体代替目标被抬起、没有爆炸/大穿透、关节误差和接触力在记录范围内。

如果某一步失败，就在该层判责，不再用更大摩擦、更大刚度或更深SQUEEZE把问题掩盖过去。
