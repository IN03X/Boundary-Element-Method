import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.special import spherical_jn, spherical_yn, lpmv
import time
from scipy.linalg import lu_factor, lu_solve

# 参数设置
c = 340.0            # 声速 (m/s)
rho = 1.225          # 空气密度 (kg/m³)
frequency = 1000.0   # 频率 (Hz)
k = 2 * np.pi * frequency / c  # 波数
r_sphere = 1.0       # 球体半径 (m)
n_points = 400      # 球面离散点数（控制精度）

# 创建球面网格
def create_sphere_mesh(radius, num_points):
    """在球面上生成均匀分布的点"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # 黄金角
    
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y从1到-1
        radius_xy = np.sqrt(1 - y * y)  # 球面投影到xy平面的半径
        
        theta = phi * i  # 黄金角旋转
        
        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy
        
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points)

# 创建三角形曲面网格
def create_triangle_mesh(points):
    """使用Delaunay三角剖分创建球面三角形网格"""
    # 投影到球面并进行三角剖分
    tri = Delaunay(points[:, [0, 1]])  # 使用x,y坐标进行2D三角剖分
    
    # 获取三角形面片
    triangles = tri.simplices
    
    # 计算每个三角形的面积和法向量
    areas = np.zeros(len(triangles))
    normals = np.zeros((len(triangles), 3))
    centroids = np.zeros((len(triangles), 3))
    
    for i, tri_indices in enumerate(triangles):
        p1, p2, p3 = points[tri_indices]
        
        # 计算三角形面积 (Heron公式)
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        areas[i] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        
        # 计算法向量 (叉积)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normals[i] = normal / np.linalg.norm(normal)  # 单位法向量
        
        # 计算质心
        centroids[i] = (p1 + p2 + p3) / 3.0
    
    return triangles, centroids, normals, areas

# 生成球面网格
points = create_sphere_mesh(r_sphere, n_points)
triangles, centroids, normals, areas = create_triangle_mesh(points)

# 自由场格林函数及其法向导数
def green_function(r, k):
    """三维自由声场格林函数"""
    if np.linalg.norm(r) < 1e-10:  # 避免奇点问题
        return 0.0
    return np.exp(1j * k * np.linalg.norm(r)) / (4 * np.pi * np.linalg.norm(r))

def grad_green_function(r, k):
    """格林函数的梯度"""
    if np.linalg.norm(r) < 1e-10:  # 避免奇点问题
        return np.array([0.0, 0.0, 0.0])
    r_norm = np.linalg.norm(r)
    g = np.exp(1j * k * r_norm) / (4 * np.pi * r_norm)
    return g * (1j * k - 1/r_norm) * r / r_norm

# 平面波入射场
def plane_wave(point, k, direction=np.array([0, 0, 1])):
    """沿direction方向传播的平面波"""
    return np.exp(1j * k * np.dot(direction, point))

def grad_plane_wave(point, k, direction=np.array([0, 0, 1])):
    """平面波的梯度"""
    return 1j * k * direction * np.exp(1j * k * np.dot(direction, point))

# 构建边界元矩阵
def build_boundary_matrices(centroids, normals, areas, k):
    n = len(centroids)
    A = np.zeros((n, n), dtype=complex)  # 系数矩阵
    M = np.zeros((n, n), dtype=complex)  # 质量矩阵（近场修正）
    
    # 对角元素（自作用项）
    for i in range(n):
        # 使用曲率修正项
        curvature_factor = 0.5  # 简化的常数因子，实际应为立体角
        A[i, i] = curvature_factor
        
    # 非对角元素
    print("构建边界元矩阵...")
    start_time = time.time()
    
    for i in range(n):
        if i % 50 == 0:
            print(f"正在处理第 {i+1}/{n} 个单元...")
        for j in range(n):
            if i == j:
                continue
                
            r = centroids[i] - centroids[j]
            g = green_function(r, k)
            dg_dn = np.dot(grad_green_function(r, k), normals[j])
            
            # Helmholtz积分方程
            A[i, j] = dg_dn * areas[j]
            M[i, j] = g * areas[j]
    
    print(f"矩阵构建完成，耗时 {time.time()-start_time:.2f} 秒")
    return A, M

# 构建右端项（边界条件）
def build_rhs(centroids, normals, areas, k):
    """构建平面波入射的右端项"""
    n = len(centroids)
    b = np.zeros(n, dtype=complex)
    
    # 刚性边界条件：总法向速度为零
    for i in range(n):
        # 入射场的法向导数
        incident_field_grad = grad_plane_wave(centroids[i], k)
        d_incident_dn = np.dot(incident_field_grad, normals[i])
        
        # 边界条件
        b[i] = -d_incident_dn * areas[i]
    
    return b

# 求解边界积分方程
def solve_bem(A, b):
    """求解线性系统"""
    print("求解线性系统...")
    start_time = time.time()
    
    # 使用LU分解求解
    lu =lu_factor(A)
    surface_pressure = lu_solve(lu, b)
    
    print(f"求解完成，耗时 {time.time()-start_time:.2f} 秒")
    return surface_pressure

# 计算近场声压
def compute_near_field(centroids, surface_pressure, k, points):
    """计算空间中任意点的总声压"""
    print(f"计算 {len(points)} 个点的声压...")
    n_tri = len(centroids)
    pressure = np.zeros(len(points), dtype=complex)
    
    for idx, pt in enumerate(points):
        # 入射场
        p_inc = plane_wave(pt, k)
        
        # 散射场
        p_scat = 0 + 0j
        
        for i in range(n_tri):
            r = pt - centroids[i]
            g = green_function(r, k)
            
            # 使用每个单元的声压值
            p_scat += surface_pressure[i] * g * areas[i]
        
        pressure[idx] = p_inc + p_scat
    
    return pressure

# 可视化函数
def plot_results(x, y, z, pressure):
    """可视化xy平面内的声压分布"""
    fig = plt.figure(figsize=(15, 10))
    
    # 声压幅值
    ax1 = fig.add_subplot(221)
    pressure_magnitude = np.abs(pressure).reshape(x.shape)
    im1 = ax1.imshow(pressure_magnitude, extent=[x.min(), x.max(), y.min(), y.max()], 
                    cmap='viridis', origin='lower')
    ax1.set_title('声压幅值 (Pa)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    fig.colorbar(im1, ax=ax1)
    
    # 声压实部
    ax2 = fig.add_subplot(222)
    pressure_real = np.real(pressure).reshape(x.shape)
    im2 = ax2.imshow(pressure_real, extent=[x.min(), x.max(), y.min(), y.max()], 
                    cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    ax2.set_title('声压实部 (Pa)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    fig.colorbar(im2, ax=ax2)
    
    # 声压实部3D视图
    ax3 = fig.add_subplot(223, projection='3d')
    surf = ax3.plot_surface(x, y, pressure_real, cmap='RdBu_r',
                           rstride=1, cstride=1, alpha=0.9)
    ax3.set_title('声压实部3D视图')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('声压 (Pa)')
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
    
    # 球体轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = r_sphere * np.cos(theta)
    y_circle = r_sphere * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'k-', linewidth=2)
    ax2.plot(x_circle, y_circle, 'k-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('bem_acoustic_field.png')
    plt.show()

# 主程序
def main():
    # 1. 创建球面网格
    print(f"在球面上创建 {n_points} 个点的网格...")
    triangles, centroids, normals, areas = create_triangle_mesh(points)
    print(f"网格创建完成，共有 {len(triangles)} 个三角单元")
    
    # 2. 构建边界元矩阵
    A, M = build_boundary_matrices(centroids, normals, areas, k)
    
    # 3. 构建右端项
    b = build_rhs(centroids, normals, areas, k)
    
    # 4. 求解表面声压
    surface_pressure = solve_bem(A, b)
    
    # 5. 计算近场声压分布 (xy平面，z=0)
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # 创建要计算的点集 (展平)
    eval_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # 计算声压
    pressure = compute_near_field(centroids, surface_pressure, k, eval_points)
    
    # 6. 可视化结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plot_results(X, Y, Z, pressure)

if __name__ == "__main__":
    main()