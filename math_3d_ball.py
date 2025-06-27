""""自由场3维脉动球声势解析解"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt

# 物理参数设置
rho0 = 1.2        # 空气密度 [kg/m³]
c = 340           # 声速 [m/s]
a = 0.1           # 球半径 [m]
f = 1000          # 频率 [Hz]
v0 = 1e-3         # 球表面法向速度幅值 [m/s]
omega = 2 * np.pi * f  # 角频率
k = omega / c      # 波数 [rad/m]

# 定义球汉克尔函数（第二类，阶数n）
def hankel2(n, x):
    return spherical_jn(n, x) - 1j * spherical_yn(n, x)

# 创建xy平面网格 (z=0)
x = np.linspace(-0.5, 0.5, 300)
y = np.linspace(-0.5, 0.5, 300)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)  # 到原点的距离

# 初始化声压矩阵
p = np.zeros_like(r, dtype=complex)

# 计算声压（仅球外部区域）
for i in range(len(x)):
    for j in range(len(y)):
        if r[i, j] > a:  # 球外区域
            # 使用解析解公式: p = [j * ρ0 * c * v0] * [h0(kr) / h1(ka)]
            p[i, j] = 1j * rho0 * c * v0 * hankel2(0, k * r[i, j]) / hankel2(1, k * a)

# 绘图 (声压幅值)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, np.abs(p), cmap='viridis', shading='auto')
plt.colorbar(label='声压幅值 [Pa]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('z=0平面声压幅值分布 (频率={}Hz)'.format(f))
plt.gca().set_aspect('equal')
plt.grid(alpha=0.3)
plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
plt.show()