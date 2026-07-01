"""
绘图模块：包含所有可视化功能

物理意义：
    通过图表展示散射模拟结果，直观对比经典和量子散射的差异。
    
    图1：波包演化过程，展示量子粒子的波动特性
    图2：散射角与碰撞参数关系，对比经典、量子和理论结果
    
    卢瑟福散射理论公式：
        θ = 2 * arctan(k / (2 * Ek * b))
        其中 k = Z1*Z2*α*ℏc
"""
import numpy as np
import matplotlib.pyplot as plt
from 物理参数 import Ek_alpha, k

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_single_wavepacket_evolution(snapshots, x, y, b_value, times, filename='波包演化图.png'):
    """
    绘制单个波包的演化图（多帧）
    
    物理意义：
        展示高斯波包在库仑势场中的演化过程，
        可以观察到波包的扩散、散射和干涉现象。
    
    参数：
        snapshots: 帧列表，每帧是概率密度 |ψ|²
        x, y: 空间坐标
        b_value: 碰撞参数
        times: 时间数组
        filename: 输出文件名
    """
    n_frames = len(snapshots)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 4))
    
    if n_frames == 1:
        axes = [axes]
    
    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]
    
    # 统一颜色标尺，便于对比不同时间帧
    vmax = max(s.max() for s in snapshots)
    
    for j in range(n_frames):
        ax = axes[j]
        # 绘制概率密度热力图
        im = ax.imshow(snapshots[j].T, extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap='hot', vmin=0, vmax=vmax)
        
        ax.set_title(f't={times[j]:.0f} fm/c')
        if j == 0:
            ax.set_ylabel(f'b={b_value} fm')
        ax.set_xlabel('x (fm)')
        
        # 标记靶核位置（原点）
        ax.scatter([0], [0], c='blue', s=30, marker='x')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"波包演化图已保存: {filename}")


def plot_wavepacket_evolution(snapshots, x, y, b_values, times, filename='图1_量子散射波包.png'):
    """
    绘制多个波包的演化图（对比不同碰撞参数）
    
    物理意义：
        对比不同碰撞参数下波包的散射行为，
        小b时波包更接近靶核，散射更明显；
        大b时波包几乎不受影响。
    
    参数：
        snapshots: 列表，每个元素是一个波包的帧列表
        b_values: 碰撞参数数组
        x, y: 空间坐标
        times: 时间数组
        filename: 输出文件名
    """
    n_wavepackets = len(snapshots)
    n_frames = len(snapshots[0])
    
    fig, axes = plt.subplots(n_wavepackets, n_frames, figsize=(8*n_frames, 6*n_wavepackets))
    
    if n_wavepackets == 1:
        axes = axes.reshape(1, -1)
    
    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]
    
    for i in range(n_wavepackets):
        vmax = max(s.max() for s in snapshots[i])
        for j in range(n_frames):
            ax = axes[i, j]
            im = ax.imshow(snapshots[i][j].T, extent=[x_min, x_max, y_min, y_max],
                          origin='lower', cmap='hot', vmin=0, vmax=vmax)
            
            if i == 0:
                ax.set_title(f't={times[j]:.0f}')
            if j == 0:
                ax.set_ylabel(f'b={b_values[i]}')
            if i == n_wavepackets-1:
                ax.set_xlabel('x (fm)')
            
            ax.scatter([0], [0], c='blue', s=30, marker='x')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"波包演化图已保存: {filename}")

def plot_theta_b_relation(b_values_classical, theta_classical, 
                          b_values_quantum, theta_quantum,
                          filename='图2_散射角与碰撞参数.png'):
    """
    绘制散射角与碰撞参数关系图
    
    物理意义：
        卢瑟福散射的核心关系：散射角θ与碰撞参数b成反比。
        经典散射应与理论曲线完全一致（忽略量子效应）。
        量子散射在小b（大角度）时会出现量子效应。
    
    参数：
        b_values_classical: 经典散射的碰撞参数
        theta_classical: 经典散射角
        b_values_quantum: 量子散射的碰撞参数
        theta_quantum: 量子散射角
        filename: 输出文件名
    """
    # 理论曲线：卢瑟福散射公式
    # θ = 2 * arctan(k / (2 * Ek * b))
    # 其中 k = Z1*Z2*α*ℏc
    b_theory = np.linspace(10, 200, 100)
    theta_theory = np.degrees(2 * np.arctan(k / (2 * Ek_alpha * b_theory)))
    
    plt.figure(figsize=(10, 6))
    
    # 理论曲线（红色实线）
    plt.plot(b_theory, theta_theory, 'r-', linewidth=2, label='卢瑟福理论')
    
    # 经典散射点（蓝色圆点）
    valid_c = ~np.isnan(theta_classical)
    if np.any(valid_c):
        plt.plot(b_values_classical[valid_c], theta_classical[valid_c], 
                'bo', markersize=3, alpha=0.5, label='经典散射')
    
    # 量子散射点（绿色三角）
    valid_q = ~np.isnan(theta_quantum)
    if np.any(valid_q):
        plt.plot(b_values_quantum[valid_q], theta_quantum[valid_q], 
                'g^', markersize=3, alpha=0.5, label='量子散射')
    
    plt.xlabel('碰撞参数 b (fm)')
    plt.ylabel('散射角 θ (度)')
    plt.title('散射角与碰撞参数关系')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(0, 150)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"散射角关系图已保存: {filename}")

def plot_summary_table(b_values_classical, theta_classical,
                      b_values_quantum, theta_quantum,
                      filename='汇总数据表.md'):
    """
    生成数据汇总表（Markdown格式）
    
    物理意义：
        量化对比经典和量子散射与理论值的偏差，
        计算相对误差，分析量子效应的影响程度。
    
    参数：
        b_values_classical: 经典散射的碰撞参数
        theta_classical: 经典散射角
        b_values_quantum: 量子散射的碰撞参数
        theta_quantum: 量子散射角
        filename: 输出文件名
    """
    # 计算理论散射角
    theta_theory = np.degrees(2 * np.arctan(k / (2 * Ek_alpha * b_values_classical)))
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 散射模拟数据汇总\n\n")
        f.write("| b (fm) | 理论θ (度) | 经典θ (度) | 量子θ (度) | 经典误差 (%) | 量子误差 (%) |\n")
        f.write("|--------|-----------|-----------|-----------|-------------|-------------|\n")
        
        for i in range(len(b_values_classical)):
            b = b_values_classical[i]
            th_t = theta_theory[i]
            th_c = f'{theta_classical[i]:.2f}' if not np.isnan(theta_classical[i]) else '-'
            th_q = f'{theta_quantum[i]:.2f}' if not np.isnan(theta_quantum[i]) else '-'
            
            # 计算相对误差
            # 误差 = |模拟值 - 理论值| / 理论值 * 100%
            err_c = f'{abs(theta_classical[i] - th_t) / th_t * 100:.2f}' if not np.isnan(theta_classical[i]) else '-'
            err_q = f'{abs(theta_quantum[i] - th_t) / th_t * 100:.2f}' if not np.isnan(theta_quantum[i]) else '-'
            
            f.write(f'| {b:.2f} | {th_t:.2f} | {th_c} | {th_q} | {err_c} | {err_q} |\n')
    
    print(f"数据汇总表已保存: {filename}")