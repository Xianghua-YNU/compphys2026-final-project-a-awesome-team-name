"""
主程序：协调经典和量子散射模拟

模拟流程：
    1. 设置物理参数和网格参数
    2. 验证Nyquist条件（确保网格能分辨波包）
    3. 均匀抽样30个粒子的碰撞参数（b: 10~125 fm）
    4. 经典散射计算（数值积分相对论运动方程）
    5. 量子散射计算（波包演化，分裂步傅里叶方法）
    6. 生成可视化图表和数据汇总

物理背景：
    本程序模拟卢瑟福散射实验：α粒子轰击金箔。
    通过对比经典和量子模拟结果，分析量子效应的影响。
    
    经典散射：基于牛顿力学，粒子具有确定的轨迹
    量子散射：基于薛定谔方程，粒子用波包描述，具有波动性
"""
import numpy as np
import time
from 物理参数 import Ek_alpha, k, m_alpha, hbar_c
from 经典散射 import batch_classical_scatter
from 量子散射 import batch_quantum_scatter
from 绘图 import plot_theta_b_relation, plot_summary_table

def main():
    print("=" * 60)
    print("散射模拟主程序【30个粒子，b:10~125fm】")
    print("=" * 60)
    
    # =============================================
    # 参数设置
    # =============================================
    n_particles = 30    # 粒子数量
    b_min = 10          # 最小碰撞参数 (fm)
    b_max = 125         # 最大碰撞参数 (fm)
    
    # 网格参数配置
    # 空间范围和分辨率需满足：
    # 1. 足够大以包含波包演化全过程
    # 2. 足够细以满足Nyquist条件 (Δx ≤ λ/2)
    grid_params = {
        'nx': 1024,     # x方向网格点数
        'ny': 512,      # y方向网格点数
        'x_min': -250,  # x方向最小值 (fm)
        'x_max': 250,   # x方向最大值 (fm)
        'y_min': -200,  # y方向最小值 (fm)
        'y_max': 200,   # y方向最大值 (fm)
        'sigma_x': 20.0, # x方向波包宽度 (fm)
        'sigma_y': 10.0, # y方向波包宽度 (fm)
        'x0': -200,     # 波包初始x位置 (fm)
        'dt': 1.0,      # 时间步长 (fm/c)
        'n_steps': 8000 # 演化总步数
    }
    
    # =============================================
    # Nyquist条件校验
    # =============================================
    # Nyquist条件：采样间距必须小于最小波长的一半
    # λ = 2π/k, Δx ≤ λ/2 = π/k
    # 即：k_max = π/Δx ≥ k
    dx = (grid_params['x_max'] - grid_params['x_min']) / (grid_params['nx'] - 1)
    Etot = Ek_alpha + m_alpha
    p0 = np.sqrt(Etot**2 - m_alpha**2)  # 动量 (MeV/c)
    k0x = p0 / hbar_c                    # 波矢 (1/fm)
    k_max = np.pi / dx                   # 网格最大可分辨波矢
    
    print(f"\n网格Nyquist条件校验:")
    print(f"  dx = {dx:.2f} fm")
    print(f"  k0x = {k0x:.4f} /fm (入射波矢)")
    print(f"  k_max = π/dx = {k_max:.2f} /fm (网格最大可分辨波矢)")
    print(f"  Nyquist条件满足: {k0x < k_max}")
    
    # =============================================
    # 步骤1: 抽样粒子碰撞参数
    # =============================================
    # 均匀分布在 [b_min, b_max] 区间
    # 碰撞参数b是粒子入射方向与靶核中心的垂直距离
    # 小b对应大散射角，大b对应小散射角
    print(f"\n步骤1: 抽样{n_particles}个粒子")
    b_values = np.linspace(b_min, b_max, n_particles)
    print(f"  b范围: [{b_min}, {b_max}] fm")
    
    # 选择5个代表性碰撞参数用于波包演化可视化
    wavepacket_indices = np.linspace(0, n_particles-1, 5, dtype=int)
    b_wavepacket = b_values[wavepacket_indices]
    
    # 保存抽样数据
    np.savez('粒子抽样结果.npz',
             b_values=b_values,
             wavepacket_indices=wavepacket_indices,
             b_wavepacket=b_wavepacket)
    print(f"  抽样数据已保存至：粒子抽样结果.npz")
    
    # =============================================
    # 步骤2: 经典散射计算
    # =============================================
    # 使用欧拉方法数值积分相对论运动方程
    # 每个粒子独立计算，跟踪其在库仑场中的轨迹
    print(f"\n步骤2: 经典散射计算（{n_particles}个粒子，数值模拟）")
    print(f"  演化步数：{grid_params['n_steps']}")
    start_time = time.time()
    theta_classical = batch_classical_scatter(b_values, Ek_alpha, grid_params)
    elapsed = time.time() - start_time
    valid_c = ~np.isnan(theta_classical)
    print(f"  计算完成，耗时: {elapsed:.2f}秒")
    print(f"  有效粒子数量: {np.sum(valid_c)}/{n_particles}")
    print(f"  经典平均散射角: {np.mean(theta_classical[valid_c]):.2f}°")
    
    # 保存经典散射结果
    np.savez('经典散射数据.npz',
             b_values=b_values,
             theta=theta_classical)
    print(f"  经典结果已保存至：经典散射数据.npz")
    
    # =============================================
    # 步骤3: 量子散射计算
    # =============================================
    # 使用分裂步傅里叶方法演化高斯波包
    # 波包在库仑势场中散射，通过动量空间分析确定散射角
    print(f"\n步骤3: 量子散射计算（{n_particles}个粒子）")
    print(f"  空间网格：{grid_params['nx']}×{grid_params['ny']}")
    print(f"  总演化步数：{grid_params['n_steps']}")
    start_time = time.time()
    theta_quantum = batch_quantum_scatter(b_values, Ek_alpha, grid_params)
    elapsed = time.time() - start_time
    valid_q = ~np.isnan(theta_quantum)
    print(f"  计算完成，耗时: {elapsed:.2f}秒")
    print(f"  有效粒子数量: {np.sum(valid_q)}/{n_particles}")
    print(f"  量子平均散射角: {np.mean(theta_quantum[valid_q]):.2f}°")
    
    # 保存量子散射结果
    np.savez('量子散射数据.npz',
             b_values=b_values,
             theta=theta_quantum)
    print(f"  量子结果已保存至：量子散射数据.npz")
    
    # =============================================
    # 步骤4: 可视化和数据汇总
    # =============================================
    print(f"\n步骤4: 绘制图二和生成数据汇总")
    
    # 绘制散射角与碰撞参数关系图
    plot_theta_b_relation(b_values, theta_classical, b_values, theta_quantum)
    print(f"  图二已保存：图2_散射角与碰撞参数.png")
    
    # 生成数据汇总表
    plot_summary_table(b_values, theta_classical, b_values, theta_quantum)
    print(f"  数据汇总已保存：汇总数据表.md")
    
    print("\n全部仿真流程执行完毕！")
    print("如需生成波包演化图，请运行：python 波包热力图.py")

if __name__ == "__main__":
    main()