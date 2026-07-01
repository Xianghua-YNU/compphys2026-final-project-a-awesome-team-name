"""
重新运行5个波包粒子的量子散射模拟
每个波包单独运行，保存为独立图片
"""
import numpy as np
import time
import gc
from 量子散射 import init_wavepacket, evolve_and_get_snapshots
from 绘图 import plot_single_wavepacket_evolution
from 物理参数 import m_alpha, Ek_alpha, hbar_c

print("重跑5个波包粒子模拟...")

sample_data = np.load('粒子抽样结果.npz')
b_values = sample_data['b_values']
wavepacket_indices = sample_data['wavepacket_indices']
b_wavepacket = b_values[wavepacket_indices]
print(f"波包粒子b值: {b_wavepacket}")

grid_params = {
    'nx': 1024,
    'ny': 1024,
    'x_min': -500,
    'x_max': 500,
    'y_min': -500,
    'y_max': 500,
    'sigma_x': 20.0,
    'sigma_y': 10.0,
    'x0': -200,
    'dt': 1.0,
    'n_steps': 8000
}

nx = grid_params['nx']
ny = grid_params['ny']
x_min = grid_params['x_min']
x_max = grid_params['x_max']
y_min = grid_params['y_min']
y_max = grid_params['y_max']
sigma_x = grid_params['sigma_x']
sigma_y = grid_params['sigma_y']
x0 = grid_params['x0']
dt = grid_params['dt']
n_steps = grid_params['n_steps']

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)

Etot = Ek_alpha + m_alpha
p0 = np.sqrt(Etot**2 - m_alpha**2)
k0x = p0 / hbar_c

frames = 5
steps_per_frame = n_steps // frames
times = np.arange(0, n_steps + 1, steps_per_frame) * dt

all_snapshots = []
for idx, b in enumerate(b_wavepacket):
    print(f"\n========== 波包 {idx+1}/{len(b_wavepacket)} ==========")
    print(f"碰撞参数 b = {b:.2f} fm")
    start_time = time.time()
    
    try:
        psi = init_wavepacket(x, y, x0, b, k0x, 0, sigma_x, sigma_y)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
        if norm > 0:
            psi /= norm
        print("波包初始化完成")
        
        _, frames_list = evolve_and_get_snapshots(psi, x, y, dx, dy, dt, n_steps, frames)
        all_snapshots.append(frames_list)
        print(f"波包演化完成，共{len(frames_list)}帧")
        
        filename = f"波包演化图_b{b:.0f}.png"
        plot_single_wavepacket_evolution(frames_list, x, y, b, times, filename)
        
        elapsed = time.time() - start_time
        print(f"波包 {idx+1} 完成，耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        print(f"波包 {idx+1} 出错: {e}")
        import traceback
        traceback.print_exc()
    
    gc.collect()

np.savez('波包演化数据.npz', 
         x=x, y=y, snapshots=all_snapshots, times=times, b_wavepacket=b_wavepacket)
print("\n波包演化数据已保存")

print("\n========== 全部完成 ==========")
print(f"生成的图片文件:")
for b in b_wavepacket:
    print(f"  波包演化图_b{b:.0f}.png")