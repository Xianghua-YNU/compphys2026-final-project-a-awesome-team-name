import numpy as np
from 物理参数 import *

def monte_carlo_sample():
    np.random.seed(42)
    target_num = N_particle
    Ek_result = np.empty(target_num)
    y_result = np.empty(target_num)
    py_result = np.empty(target_num)
    idx = 0
    round_cnt = 0

    while idx < target_num:
        round_cnt += 1
        batch = target_num * 2
        # 能量均匀抽样
        Ek_batch = np.random.uniform(E_k_min, E_k_max, size=batch)
        # 横向位置直接限定在 [-b_max, b_max] 均匀抽样，绝对不会越界
        y_batch = np.random.uniform(-b_max, b_max, size=batch)
        b_batch = np.abs(y_batch)
        # 仅过滤小于最小碰撞参数的无效值
        valid_mask = b_batch >= b_min
        Ek_valid = Ek_batch[valid_mask]
        y_valid = y_batch[valid_mask]
        valid_count = len(Ek_valid)

        if valid_count == 0:
            print(f"第{round_cnt}轮：本轮无有效粒子，继续采样")
            continue

        # 向量化批量计算动量
        Etot = Ek_valid + m_e
        p0 = np.sqrt(Etot ** 2 - m_e ** 2)
        sigma_p = epsilon_p * p0
        py_valid = np.random.normal(loc=0, scale=sigma_p, size=valid_count)

        take_num = min(valid_count, target_num - idx)
        Ek_result[idx:idx+take_num] = Ek_valid[:take_num]
        y_result[idx:idx+take_num] = y_valid[:take_num]
        py_result[idx:idx+take_num] = py_valid[:take_num]
        idx += take_num
        print(f"第{round_cnt}轮：已采集 {idx}/{target_num} 个粒子")

    np.savez("粒子抽样结果.npz", Ek=Ek_result, y=y_result, py=py_result)
    print(f"✅    {target_num}个粒子抽样全部完成！")
    return Ek_result, y_result, py_result

if __name__ == "__main__":
    monte_carlo_sample()