# Task1 开发规范：双 Franka Panda 协同搬运长杆

## 1. 项目目标

使用 **MuJoCo + Python** 实现一个最小可运行的双臂协同操作 Demo。

当前阶段仅实现课程设计中的 **Task1：双臂同时搬运长杆**。

系统中包含两台 Franka Panda 机械臂、一根长杆、工作平面和目标区域。长杆初始静置于桌面，两台机械臂位于长杆同侧或两侧的合理位置，初始均不接触长杆。任务开始后，两台机械臂协同伸向长杆两端，完成抓取、同步抬升、搬运、放置和释放。

本阶段不要求接入真实 VLM/VLA 模型，但程序结构需要为后续“语言解析、角色分配、多视角感知、失败重规划”保留接口。

---

## 2. 技术约束

- Python 3.x
- MuJoCo Python 官方接口
- 机械臂模型：Franka Panda
- 使用 Git 管理代码历史
- 所有代码应尽量模块化，禁止把场景、控制、任务流程全部堆在一个脚本中
- 优先使用 MuJoCo 原生 XML/MJCF 建模与碰撞检测能力
- Task1 首先保证稳定完成，不要求复杂强化学习或端到端 VLA

---

## 3. Task1 场景定义

### 3.1 场景组成

场景至少包括：

1. 左侧 Franka Panda 机械臂 `left_arm`
2. 右侧 Franka Panda 机械臂 `right_arm`
3. 一根长杆 `rod`
4. 工作平面 `table`
5. 长杆初始放置区域 `start_region`
6. 长杆目标放置区域 `target_region`
7. 可选的可视化标记，例如目标区域半透明方框或地面标记

### 3.2 初始状态

程序启动后应满足：

- 长杆水平放置在工作平面上并保持静止
- 两台机械臂处于安全初始姿态
- 两侧夹爪张开
- 两台机械臂均未接触长杆
- 机械臂与长杆、桌面、另一台机械臂之间不存在初始穿模
- 目标区域位于长杆初始区域的对面，并有明确可视化标记

建议长杆沿世界坐标系 X 或 Y 方向水平放置，使左右机械臂分别抓取长杆两端附近。

---

## 4. Task1 行为流程

Task1 必须使用明确的阶段状态机执行，不允许通过一段不可解释的连续轨迹直接完成。

推荐状态：

```text
INIT
  -> PRE_GRASP
  -> APPROACH
  -> GRASP
  -> CHECK_GRASP
  -> LIFT
  -> TRANSPORT
  -> PLACE
  -> RELEASE
  -> RETREAT
  -> SUCCESS
```

发生异常时进入：

```text
FAILURE_STOP
```

### 4.1 INIT

- 初始化仿真
- 设置双臂初始关节状态
- 设置长杆初始位姿
- 夹爪保持张开
- 检查场景中无非法碰撞

### 4.2 PRE_GRASP

- 计算左右机械臂各自的预抓取位姿
- 左臂目标为长杆左端附近
- 右臂目标为长杆右端附近
- 夹爪方向应能够稳定夹持长杆
- 两个末端执行器与长杆保持一定安全距离

### 4.3 APPROACH

- 双臂从预抓取位姿同步接近长杆
- 接近过程中持续进行碰撞检测
- 禁止双机械臂本体互相碰撞
- 禁止机械臂以明显异常方式碰撞桌面

### 4.4 GRASP

- 两侧夹爪同时闭合
- 左右夹爪分别夹持长杆两端附近
- 不允许只依赖“瞬移”或直接修改长杆位姿伪造抓取结果

### 4.5 CHECK_GRASP

必须验证两侧抓取状态。

至少检查：

- 左侧是否成功接触/夹持长杆
- 右侧是否成功接触/夹持长杆
- 长杆是否仍处于合理姿态

如果任意一侧抓取失败：

1. 立即停止另一侧继续搬运动作
2. 双臂进入安全保持状态
3. 状态机进入 `FAILURE_STOP`
4. 输出失败原因

本阶段只要求实现“停止”，不强制完成完整自动重规划；代码中需预留 `replan()` 或等价接口，供后续实现失败重规划。

### 4.6 LIFT

- 确认双侧抓取成功后开始抬升
- 两侧末端应同步向上运动
- 长杆离开桌面一定安全距离
- 抬升过程中监测长杆姿态

### 4.7 TRANSPORT

- 双臂协同将长杆搬运至目标区域上方
- 左右机械臂保持相对协调运动
- 尽量保持长杆接近水平
- 运动过程中持续碰撞检查

### 4.8 PLACE

- 双臂同步降低长杆
- 将长杆放置到目标区域内
- 长杆应稳定接触工作平面

### 4.9 RELEASE

- 确认长杆已稳定放置后，两侧夹爪同时张开
- 释放过程中不得明显推动或弹飞长杆

### 4.10 RETREAT

- 两台机械臂末端向安全方向撤离
- 避免撤离过程中碰撞长杆或另一台机械臂

### 4.11 SUCCESS

满足验收条件后输出任务成功信息，并记录本次运行指标。

---

## 5. 双臂角色定义

虽然 Task1 中两臂基本对称，但程序仍需显式定义角色，方便后续课程任务扩展。

推荐：

```python
roles = {
    "left_arm": "rod_left_grasper",
    "right_arm": "rod_right_grasper",
}
```

任务配置中至少表示：

- 操作对象：`rod`
- 左臂角色
- 右臂角色
- 起始区域
- 目标区域
- 动作先后关系

推荐抽象任务描述：

```python
TaskCommand(
    object="rod",
    left_role="grasp_left_end",
    right_role="grasp_right_end",
    target="target_region",
    sequence=["approach", "grasp", "lift", "transport", "place", "release"],
)
```

当前可直接使用规则或硬编码生成该结构，不要求自然语言模型。

---

## 6. 控制要求

### 6.1 基本控制

可以选择以下任一种方式实现：

- 关节空间位置控制
- 基于逆运动学的末端位姿控制
- Operational Space Control

为了降低 Demo 难度，推荐优先采用：

> 末端目标轨迹 + 逆运动学/位置控制 + MuJoCo actuator

禁止直接修改机械臂末端 body 的世界坐标完成运动。

### 6.2 同步控制

双臂搬运阶段需要在同一个控制循环中更新左右机械臂目标。

定义左右末端相对其期望轨迹的同步误差，可使用简化形式：

```text
E_sync = || Δp_left - Δp_right ||
```

其中 `Δp_left` 和 `Δp_right` 表示当前控制周期两侧末端相对于各自参考轨迹的位移误差。

也可以记录两侧末端到各自期望位置的误差差值。

程序运行结束后至少输出：

- 最大同步误差
- 平均同步误差

---

## 7. 长杆姿态误差

搬运过程中记录长杆姿态。

Task1 主要要求长杆尽量保持水平，因此至少统计：

- 长杆 Roll/Pitch 偏差
- 或长杆当前姿态相对于参考姿态的旋转角误差

建议记录：

```text
rod_position_error
rod_orientation_error
```

任务结束时计算长杆最终位姿相对于目标位姿的误差：

```text
position_error = ||p_rod - p_target||
orientation_error = rotation_distance(R_rod, R_target)
```

---

## 8. 碰撞检测

Task1 必须包含碰撞检查。

至少检测：

1. 左臂与右臂之间的碰撞
2. 左臂自身不合理自碰撞
3. 右臂自身不合理自碰撞
4. 机械臂与桌面的非预期碰撞
5. 搬运过程中除夹爪允许区域外的机械臂与长杆异常碰撞

允许的正常接触：

- 左夹爪与长杆
- 右夹爪与长杆
- 长杆放置阶段与桌面

发现危险碰撞时：

```text
立即停止双臂运动 -> FAILURE_STOP
```

碰撞检测逻辑应独立封装，例如：

```python
CollisionChecker
```

不得只依赖人工观察判断碰撞。

---

## 9. 抓取失败处理

必须实现最小失败联锁机制。

示例：

```python
if not left_grasp_ok or not right_grasp_ok:
    controller.stop_both_arms()
    state = FAILURE_STOP
```

要求：

- 一侧失败时另一侧不得继续执行 `LIFT` 或 `TRANSPORT`
- 日志明确指出是哪一侧失败
- 预留重新规划接口

推荐：

```python
def replan(failure_reason):
    raise NotImplementedError
```

后续再扩展真正的重新抓取流程。

---

## 10. 推荐项目结构

```text
project/
├── AGENTS.md
├── spec.md
├── README.md
├── requirements.txt
├── assets/
│   ├── franka/
│   └── scene/
├── models/
│   ├── dual_panda.xml
│   └── task1_scene.xml
├── src/
│   ├── main.py
│   ├── config.py
│   ├── scene.py
│   ├── task.py
│   ├── state_machine.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── arm_controller.py
│   │   └── dual_arm_controller.py
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── trajectory.py
│   │   └── ik.py
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── collision.py
│   │   └── grasp_monitor.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── visualization/
│       ├── __init__.py
│       └── viewer.py
├── scripts/
│   └── run_task1.py
├── tests/
│   ├── test_scene.py
│   ├── test_collision.py
│   └── test_task1.py
└── outputs/
    ├── logs/
    └── screenshots/
```

允许根据实际实现适度精简，但应保持场景、控制、安全检查、指标统计相互独立。

---

## 11. 日志与指标

每次运行至少输出：

```text
Task: task1_rod_transport
Result: SUCCESS / FAILURE
Failure reason: ...
Max synchronization error: ... m
Mean synchronization error: ... m
Final rod position error: ... m
Final rod orientation error: ... deg
Collision detected: True / False
Left grasp success: True / False
Right grasp success: True / False
```

建议保存到：

```text
outputs/logs/task1_xxx.json
```

---

## 12. 验收标准

Task1 只有在以下条件全部满足时才视为完成：

### 必须满足

- [ ] MuJoCo 场景可正常加载
- [ ] 场景中存在两台 Franka Panda
- [ ] 长杆初始稳定放置在桌面
- [ ] 双臂初始不抓取长杆
- [ ] 有明确目标区域
- [ ] 双臂能够同时接近长杆
- [ ] 左右夹爪能够分别抓住长杆两端
- [ ] 任一侧抓取失败时另一侧停止
- [ ] 双臂能够共同抬起长杆
- [ ] 双臂能够将长杆搬运到目标区域
- [ ] 能够放下并释放长杆
- [ ] 任务全过程执行碰撞检查
- [ ] 不发生双臂之间的危险碰撞
- [ ] 能输出同步误差
- [ ] 能输出长杆最终位姿误差
- [ ] 成功和失败状态均有明确日志

### 建议目标

为了方便课程展示，可将以下数值作为初期调试目标，而非硬性课程指标：

```text
最终位置误差 < 0.10 m
最终姿态误差 < 10 deg
双臂最大同步误差 < 0.05 m
```

若实际控制精度允许，可继续收紧阈值。

---

## 13. 可视化验证

开发过程中如有可用视觉能力，应启动 MuJoCo Viewer 进行验证。

建议在以下状态保存截图：

1. `INIT`：长杆静置、双臂未抓取
2. `GRASP`：双臂同时抓住长杆
3. `TRANSPORT`：长杆被抬起并搬运
4. `PLACE`：长杆位于目标区域
5. `SUCCESS`：夹爪释放并撤离

截图保存到：

```text
outputs/screenshots/
```

视觉检查只作为辅助，不能替代程序中的碰撞检测和数值指标。

---

## 14. 当前阶段不做的内容

以下内容不是 Task1 最小 Demo 的阻塞项：

- 真正的 VLM/VLA 模型推理
- 复杂自然语言理解
- 多相机视觉定位
- 强化学习控制
- 动态避障
- 完整自动失败重规划
- Task2 左手固定右手装配
- Task3 双手抱箱

但代码设计不得妨碍后续增加这些功能。

---

## 15. Task1 完成定义（Definition of Done）

能够通过一条命令启动 Demo，例如：

```bash
python scripts/run_task1.py
```

随后在 MuJoCo Viewer 中完整观察到：

```text
初始静止
-> 双臂接近
-> 同时抓取
-> 同步抬升
-> 搬运至目标区域
-> 同步放下
-> 释放
-> 双臂撤离
```

且程序最终输出 `SUCCESS`、同步误差、长杆位姿误差和碰撞状态，即认为 Task1 最小版本完成。
