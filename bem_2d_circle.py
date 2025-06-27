import numpy as np
import matplotlib.pyplot as plt

def green_function(x, y, x0, y0):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    return (1 / (2 * np.pi)) * np.log(r)

def dGdn(x, y, x0, y0, nx, ny):
    dx = x - x0
    dy = y - y0
    r2 = dx**2 + dy**2
    return (dx * nx + dy * ny) / (2 * np.pi * r2)

N = 20  # 单元数
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
nodes = np.array([(np.cos(t), np.sin(t)) for t in theta])  # 节点坐标

# 初始化系数矩阵 H 和 G
H = np.zeros((N, N))
G = np.zeros((N, N))

# 组装矩阵 H 和 G
for i in range(N):  # 源点 i
    xi, yi = nodes[i]
    ni_x = np.cos(theta[i])  # 单位圆边界法向量（径向）
    ni_y = np.sin(theta[i])
    
    for j in range(N):  # 单元 j
        xj, yj = nodes[j]
        xj_next = nodes[(j+1)%N][0]
        yj_next = nodes[(j+1)%N][1]
        
        # 常数单元：单元中心作为积分点
        xc = (xj + xj_next) / 2
        yc = (yj + yj_next) / 2
        
        # 计算单元长度和法向
        length = np.sqrt((xj_next - xj)**2 + (yj_next - yj)**2)
        normal_x = (yj_next - yj) / length  # 法向量（逆时针方向）
        normal_y = -(xj_next - xj) / length
        
        # 判断是否为奇异积分（源点和积分点相同）
        if i == j:
            # 奇异积分：H_ii 的对角项（解析积分）
            H[i, j] = -0.5  # 常数单元下几何系数 c=0.5
            G[i, j] = (1 / (2 * np.pi)) * length  # 非奇异项
        else:
            # 非奇异积分：数值积分（单点高斯积分）
            g = green_function(xc, yc, xi, yi)
            dg = dGdn(xc, yc, xi, yi, normal_x, normal_y)
            H[i, j] += dg * length
            G[i, j] += g * length

# 应用边界条件（Dirichlet: u = sin(theta)）
u_bc = np.sin(theta)  # 边界上的已知值
q_bc = np.linalg.solve(G, H @ u_bc)  # 求解通量 q

# 可视化边界上的 q 解
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8, 4))
plt.plot(theta, q_bc, 'o-', label='BEM 解')
plt.xlabel('角度 θ')
plt.ylabel('通量 q(θ)')
plt.legend()
plt.grid(True)
plt.title('边界通量 q(θ)')
plt.show()

# 计算域内点的解（验证解析解）
def bem_solution(x, y):
    u = 0
    for j in range(N):
        xj, yj = nodes[j]
        xj_next = nodes[(j+1)%N][0]
        yj_next = nodes[(j+1)%N][1]
        xc = (xj + xj_next) / 2
        yc = (yj + yj_next) / 2
        length = np.sqrt((xj_next - xj)**2 + (yj_next - yj)**2)
        normal_x = (yj_next - yj) / length
        normal_y = -(xj_next - xj) / length
        
        g = green_function(x, y, xc, yc)
        dg = dGdn(xc, yc, x, y, normal_x, normal_y)
        u += u_bc[j] * dg * length - q_bc[j] * g * length
    return u / (1.0)  # 因为 c=1 在域内

# 绘制域内解（极坐标网格）
r = np.linspace(0, 1, 20)
t = np.linspace(0, 2*np.pi, 40)
R, T = np.meshgrid(r, t)
X = R * np.cos(T)
Y = R * np.sin(T)
U = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        U[i, j] = bem_solution(X[i,j], Y[i,j])

# 绘制解析解对比
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, U, levels=20, cmap='viridis')
plt.colorbar()
plt.title('BEM 数值解')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, R*np.sin(T), levels=20, cmap='viridis')
plt.colorbar()
plt.title('解析解 u = r sinθ')
plt.axis('equal')
plt.show()