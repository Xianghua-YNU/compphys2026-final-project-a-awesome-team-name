# 3_Data/ - 模拟数据目录

**目的**: 存放模拟过程中产生的数据或用于分析的外部公开数据。

## 目录结构

```
3_Data/
├── raw_data/              # 原始输出数据
│   ├── 粒子抽样结果.npz    # 粒子抽样参数（碰撞参数等）
│   ├── 经典散射数据.npz    # 经典散射模拟结果
│   ├── 量子散射数据.npz    # 量子散射模拟结果
│   └── 波包演化数据.npz    # 波包演化过程数据
└── processed_data/        # 处理后的汇总数据
    └── 汇总数据表.md       # 模拟结果数据汇总（含误差分析）
```

## 数据说明

### raw_data/

#### 粒子抽样结果.npz
- 内容：30个粒子的碰撞参数(b)、波包索引、波包碰撞参数
- 格式：NumPy .npz 文件
- 键：`b_values`, `wavepacket_indices`, `b_wavepacket`

#### 经典散射数据.npz
- 内容：经典散射模拟结果（碰撞参数与散射角的对应关系）
- 格式：NumPy .npz 文件
- 键：`b_values`, `theta`

#### 量子散射数据.npz
- 内容：量子散射模拟结果（碰撞参数与散射角的对应关系）
- 格式：NumPy .npz 文件
- 键：`b_values`, `theta`

#### 波包演化数据.npz
- 内容：5个不同碰撞参数波包的演化过程快照
- 格式：NumPy .npz 文件

### processed_data/

#### 汇总数据表.md
- 内容：散射模拟数据汇总表，包含理论值、经典散射结果、量子散射结果及误差分析
- 格式：Markdown表格

## 数据来源

所有数据均由 [2_Code/经典散射和量子散射/main.py](file:///C:/Users/zweige/Desktop/compphys2026-final-project-a-awesome-team-name/2_Code/经典散射和量子散射/main.py) 生成：
- 经典散射：使用欧拉方法数值积分相对论运动方程
- 量子散射：使用分裂步傅里叶方法演化高斯波包

## 重新生成数据

```bash
cd 2_Code/经典散射和量子散射
python main.py
```

运行后数据会自动保存到对应目录。
