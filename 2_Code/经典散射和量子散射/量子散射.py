"""
量子散射核心计算模块

物理原理：
    量子散射的描述基于含时薛定谔方程。将α粒子用高斯波包表示，
    在金核的库仑势场中演化，通过分析散射后的波包动量分布来确定散射角。

核心方程：
    含时薛定谔方程: iℏ ∂ψ/∂t = (-ℏ²/(2m) ∇² + V(r)) ψ
    在自然单位制下 (ℏ=1): i ∂ψ/∂t = (-1/(2m) ∇² + V(r)) ψ

数值方法：
    分裂步傅里叶方法 (Split-Step Fourier Method):
    1. 在实空间应用势场演化 exp(-iVΔt/2)
    2. 傅里叶变换到动量空间
    3. 在动量空间应用动能演化 exp(-iKΔt)
    4. 傅里叶变换回实空间
    5. 在实空间应用势场演化 exp(-iVΔt/2)
    
    这种方法的优势在于动能算符在动量空间是对角的，
    可以高效计算。
"""
import numpy as np
from 物理参数 import m_alpha, hbar_c, k

def init_wavepacket(x, y, x0, y0, k0x, k0y, sigma_x, sigma_y):
    """
    初始化高斯波包
    
    物理意义：
        高斯波包描述了一个局域化的量子粒子，具有确定的平均动量。
        波包的空间宽度和动量宽度满足不确定关系: Δx * Δp ≥ ℏ/2
    
    波包形式：
        ψ(x,y) = exp(-(x-x0)²/(2σ_x²) - (y-y0)²/(2σ_y²)) 
                  * exp(i(k0x*(x-x0) + k0y*(y-y0)))
    
    参数：
        x, y: 空间坐标网格
        x0, y0: 波包中心位置
        k0x, k0y: 平均波矢 (1/fm)
        sigma_x, sigma_y: 波包宽度 (fm)
    
    返回：
        psi: 归一化前的波函数 (复数数组)
    """
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # 高斯包络：描述粒子的空间分布
    # 指数衰减保证波包局域化，宽度由sigma_x/y控制
    psi = np.exp(-(xx-x0)**2/(2*sigma_x**2) - (yy-y0)**2/(2*sigma_y**2))
    psi = psi.astype(np.complex128)
    
    # 平面波相位：描述粒子的运动方向和动量
    # k0x, k0y 对应平均动量 p0x = ℏk0x, p0y = ℏk0y
    psi *= np.exp(1j * (k0x*(xx-x0) + k0y*(yy-y0)))
    
    return psi

def V_coulomb(x, y):
    """
    库仑势场计算
    
    物理意义：
        金核对α粒子的库仑排斥势 V(r) = Z1*Z2*α*ℏc / r
    
    参数：
        x, y: 空间坐标
    
    返回：
        V: 库仑势 (MeV)
    """
    r = np.sqrt(x**2 + y**2)
    r = np.maximum(r, 0.1)  # 防止r=0导致势场无穷大
    return k / r

def evolve_wavepacket(psi, x, y, dx, dy, dt, n_steps):
    """
    分裂步傅里叶方法演化波包
    
    物理原理：
        含时薛定谔方程的形式解: ψ(t+Δt) = exp(-iHΔt) ψ(t)
        其中 H = K + V 是哈密顿量（动能+势能）
        
        分裂步近似利用Baker-Campbell-Hausdorff公式：
        exp(-i(K+V)Δt) ≈ exp(-iVΔt/2) * exp(-iKΔt) * exp(-iVΔt/2)
        
        这样可以将动能和势能的演化分离，分别在动量空间和实空间高效计算。
    
    参数：
        psi: 初始波函数 (复数数组)
        x, y: 空间坐标
        dx, dy: 空间步长 (fm)
        dt: 时间步长 (fm/c)
        n_steps: 演化步数
    
    返回：
        psi: 演化后的波函数
    """
    nx, ny = len(x), len(y)
    
    # =============================================
    # 动量空间网格计算
    # =============================================
    # 使用FFT频率公式：k = 2π * fftfreq(n, dx)
    # kx, ky 是动量空间的波矢坐标 (1/fm)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing='ij')
    k2 = kx2 + ky2  # 波矢平方 (1/fm²)
    
    # =============================================
    # 预计算演化算符
    # =============================================
    # 库仑势 V(r) (MeV)，转换为自然单位制 (除以ℏc)
    V = V_coulomb(x[:, np.newaxis], y[np.newaxis, :]) / hbar_c
    
    # 势场演化算符: exp(-iVΔt/2)
    # 在实空间应用，对每个空间点独立作用
    exp_V = np.exp(-1j * V * dt / 2)
    
    # 动能演化算符: exp(-iKΔt) = exp(-iℏ²k²/(2m) Δt)
    # 在动量空间应用，K = ℏ²k²/(2m)
    exp_K = np.exp(-1j * hbar_c * k2 / (2 * m_alpha) * dt)
    
    # =============================================
    # 时间演化循环
    # =============================================
    for step in range(n_steps):
        # 第一步：实空间应用势场演化 (前半步)
        psi = exp_V * psi
        
        # 第二步：傅里叶变换到动量空间
        psi_k = np.fft.fft2(psi)
        
        # 第三步：动量空间应用动能演化
        psi_k = exp_K * psi_k
        
        # 第四步：傅里叶变换回实空间
        psi = np.fft.ifft2(psi_k)
        
        # 第五步：实空间应用势场演化 (后半步)
        psi = exp_V * psi
    
    return psi

def evolve_and_get_snapshots(psi, x, y, dx, dy, dt, n_steps, n_frames):
    """
    演化波包并返回多个时间帧（用于可视化）
    
    参数：
        psi: 初始波函数
        x, y: 空间坐标
        dx, dy: 空间步长
        dt: 时间步长
        n_steps: 总演化步数
        n_frames: 需要保存的帧数
    
    返回：
        psi: 最终波函数
        snapshots: 各时间帧的概率密度 |ψ|²
    """
    snapshots = []
    steps_per_frame = n_steps // n_frames
    
    for i in range(n_frames):
        psi = evolve_wavepacket(psi, x, y, dx, dy, dt, steps_per_frame)
        # 保存概率密度 |ψ|²
        snapshots.append((np.abs(psi)**2).copy())
    
    return psi, snapshots

def single_quantum_scatter(b, Ek, grid_params=None):
    """
    单个波包的量子散射模拟，返回散射角
    
    物理过程：
        1. 初始化高斯波包，中心位于碰撞参数b处
        2. 在库仑势场中演化波包
        3. 通过动量空间分析确定散射后的动量方向
        4. 根据动量方向计算散射角
    
    参数：
        b: 碰撞参数 (fm) - 波包中心的初始y坐标
        Ek: α粒子动能 (MeV)
        grid_params: 可选字典，包含网格参数
    
    返回：
        theta: 散射角 (度)
    
    网格参数说明：
        nx, ny: 网格点数 (需满足Nyquist条件)
        x_min, x_max, y_min, y_max: 空间范围
        sigma_x, sigma_y: 波包宽度 (fm)
        x0: 初始x位置 (波包起始位置)
        dt: 时间步长 (fm/c)
        n_steps: 演化步数
    """
    if grid_params is None:
        grid_params = {
            'nx': 512,     # x方向网格点数
            'ny': 256,     # y方向网格点数
            'x_min': -200, # x方向最小值
            'x_max': 200,  # x方向最大值
            'y_min': -150, # y方向最小值
            'y_max': 150,  # y方向最大值
            'sigma_x': 20.0,  # x方向波包宽度 (fm)
            'sigma_y': 15.0,  # y方向波包宽度 (fm)
            'x0': -120,    # 波包初始x位置
            'dt': 1.0,     # 时间步长 (fm/c)
            'n_steps': 8000 # 演化总步数
        }
    
    # =============================================
    # 提取网格参数
    # =============================================
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
    
    # =============================================
    # 构建空间网格
    # =============================================
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    dx = (x_max - x_min) / (nx - 1)  # x方向步长 (fm)
    dy = (y_max - y_min) / (ny - 1)  # y方向步长 (fm)
    
    # =============================================
    # 计算初始波矢
    # =============================================
    # E = Ek + mc² = √(p²c² + m²c⁴)
    # k = p/ℏ = p/(ℏc) * c = p/hbar_c
    Etot = Ek + m_alpha
    p0 = np.sqrt(Etot**2 - m_alpha**2)
    k0x = p0 / hbar_c  # x方向波矢 (1/fm)
    k0y = 0            # y方向波矢 = 0
    
    # =============================================
    # 初始化波包
    # =============================================
    psi = init_wavepacket(x, y, x0, b, k0x, k0y, sigma_x, sigma_y)
    
    # 归一化：确保总概率为1
    # ∫|ψ|² dxdydz = 1
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    if norm > 0:
        psi /= norm
    
    # =============================================
    # 波包演化
    # =============================================
    psi = evolve_wavepacket(psi, x, y, dx, dy, dt, n_steps)
    
    # =============================================
    # 动量空间分析计算散射角
    # =============================================
    # 傅里叶变换到动量空间
    psi_k = np.fft.fft2(psi)
    prob_k = np.abs(psi_k)**2  # 动量空间概率密度
    
    # 动量空间坐标
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    
    # 计算平均动量
    total_prob = np.sum(prob_k)
    if total_prob < 1e-10:
        return np.nan
    
    # <kx> = ∫kx * |ψ(k)|² dk / ∫|ψ(k)|² dk
    kx_avg = np.sum(kx_grid * prob_k) / total_prob
    ky_avg = np.sum(ky_grid * prob_k) / total_prob
    
    # 根据平均动量方向计算散射角
    # θ = arctan(|ky_avg| / kx_avg)
    if kx_avg > 0:
        theta = np.degrees(np.arctan2(np.abs(ky_avg), kx_avg))
    else:
        theta = 180.0 - np.degrees(np.arctan2(np.abs(ky_avg), np.abs(kx_avg)))
    
    return theta

def batch_quantum_scatter(b_values, Ek, grid_params=None):
    """
    批量量子散射计算
    
    参数：
        b_values: 碰撞参数数组 (fm)
        Ek: α粒子动能 (MeV)
        grid_params: 网格参数
    
    返回：
        thetas: 散射角数组 (度)
    """
    thetas = []
    for i, b in enumerate(b_values):
        print(f"  粒子{i+1}/{len(b_values)}: b={b:.1f} fm")
        theta = single_quantum_scatter(b, Ek, grid_params)
        thetas.append(theta)
        print(f"    完成: θ={theta:.2f}°" if not np.isnan(theta) else "    完成: θ=NaN")
    return np.array(thetas)