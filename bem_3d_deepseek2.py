import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from stl import mesh as stl_mesh  

# 1. 几何模型定义与面元划分
class SurfaceMesh:
    def __init__(self):
        self.vertices = None      # 顶点坐标 (Nv, 3)
        self.faces = None         # 面元顶点索引 (Nf, 3)
        self.centroids = None     # 面元质心坐标 (Nf, 3)
        self.areas = None         # 面元面积 (Nf,)
        self.normals = None       # 面元单位法向量 (Nf, 3)
        self.N = None             # 面元数量
        self.radius = None        # 球体半径（如果适用）

    def generate_sphere_stl(self, radius=1.0, resolution=12, filename="sphere.stl"):
        """生成球体STL文件（使用numpy-stl）"""
        # 生成顶点
        num_phi = resolution
        num_theta = 2 * resolution
        
        vertices = []
        # 北极点
        vertices.append([0, 0, radius])
        # 中间层顶点
        for i in range(1, num_phi):
            phi = np.pi * i / num_phi
            for j in range(num_theta):
                theta = 2 * np.pi * j / num_theta
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                vertices.append([x, y, z])
        # 南极点
        vertices.append([0, 0, -radius])
        
        # 创建面元数据
        faces = []
        # 北极区
        for j in range(num_theta):
            v0 = 0
            v1 = 1 + j
            v2 = 1 + (j + 1) % num_theta
            faces.append([v0, v1, v2])
        # 中间区
        for i in range(1, num_phi - 1):
            for j in range(num_theta):
                start = 1 + (i - 1) * num_theta
                next_start = 1 + i * num_theta
                v0 = start + j
                v1 = start + (j + 1) % num_theta
                v2 = next_start + j
                v3 = next_start + (j + 1) % num_theta
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])
        # 南极区
        south_pole = len(vertices) - 1
        last_ring_start = 1 + (num_phi - 2) * num_theta
        for j in range(num_theta):
            v0 = last_ring_start + j
            v1 = last_ring_start + (j + 1) % num_theta
            v2 = south_pole
            faces.append([v0, v1, v2])
        
        # 创建STL网格
        stl_mesh = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[face[j]]
        
        # 保存STL文件
        stl_mesh.save(filename)
        print(f"已生成球面STL文件: {filename} (半径={radius:.1f}m, 分辨率={resolution})")
        self.radius = radius
        return filename

    def load_from_stl(self, filename):
        """从STL文件加载网格"""
        # 修正：使用stl_mesh模块而不是自身实例的mesh属性
        stl_mesh_obj = stl_mesh.Mesh.from_file(filename)
        self.N = len(stl_mesh_obj.vectors)
        
        # 提取顶点和面元
        # STL网格中的每个三角形都是独立存储的，所以我们需要重建顶点列表
        # 注意：STL文件中的顶点是按面存储的，会有重复顶点
        all_vertices = stl_mesh_obj.vectors.reshape(-1, 3)
        
        # 创建唯一顶点列表
        self.vertices = np.unique(all_vertices, axis=0)
        
        # 创建面元索引
        self.faces = []
        vertex_map = {}
        for i, vertex in enumerate(self.vertices):
            vertex_map[tuple(vertex)] = i
            
        for vectors in stl_mesh_obj.vectors:
            face = []
            for vector in vectors:
                # 查找当前顶点在唯一列表中的索引
                idx = vertex_map[tuple(vector)]
                face.append(idx)
            self.faces.append(face)
        
        self.faces = np.array(self.faces)
        
        # 计算面元属性
        self._compute_face_properties()
        print(f"已加载STL文件: {filename} ({self.N}个面元)")

    def _compute_face_properties(self):
        """计算每个面元的质心、面积和法向量"""
        self.centroids = np.zeros((self.N, 3))
        self.areas = np.zeros(self.N)
        self.normals = np.zeros((self.N, 3))
        
        for i in range(self.N):
            # 获取面元的三个顶点
            v0, v1, v2 = self.vertices[self.faces[i]]
            
            # 计算质心
            centroid = (v0 + v1 + v2) / 3.0
            self.centroids[i] = centroid
            
            # 计算面法向量
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            # 计算面积
            area = 0.5 * np.linalg.norm(normal)
            self.areas[i] = area
            
            # 归一化法向量
            self.normals[i] = normal / (2 * area)

    def visualize(self):
        """可视化网格模型"""
        if self.vertices is None or self.faces is None:
            print("错误: 尚未加载网格数据")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建面元的集合
        face_vertices = [self.vertices[face] for face in self.faces]
        surface = Poly3DCollection(face_vertices, alpha=0.5, linewidths=0.5, edgecolor='k')
        
        # 设置面元颜色（基于法向量方向）
        colors = np.abs((self.normals + 1.0) / 2.0)
        surface.set_facecolor(colors)
        
        # 添加面元到绘图
        ax.add_collection3d(surface)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if self.radius is not None:
            ax.set_xlim([-self.radius, self.radius])
            ax.set_ylim([-self.radius, self.radius])
            ax.set_zlim([-self.radius, self.radius])
            ax.set_title(f'球体网格可视化 (半径={self.radius:.1f}m, {self.N}个面元)')
        else:
            min_coords = np.min(self.vertices, axis=0)
            max_coords = np.max(self.vertices, axis=0)
            ax.set_xlim([min_coords[0], max_coords[0]])
            ax.set_ylim([min_coords[1], max_coords[1]])
            ax.set_zlim([min_coords[2], max_coords[2]])
            ax.set_title(f'网格模型可视化 ({self.N}个面元)')
        
        plt.tight_layout()
        plt.show()

# 2. 核函数计算
def green_function(k, r, r0):
    """三维自由声场基本解"""
    R = distance.euclidean(r, r0)
    #return np.exp(-1j * k * R) / (4 * np.pi * R)
    return np.exp(1j * k * R) / (4 * np.pi * R)

def green_function_derivative(k, r, r0, normal):
    """计算格林函数的法向导数"""
    R_vec = r - r0
    R = np.linalg.norm(R_vec)
    #return (-1j * k * R - 1) * np.exp(-1j * k * R) / (4 * np.pi * R**3) * np.dot(R_vec, normal)
    return (1j * k * R - 1) * np.exp(1j * k * R) / (4 * np.pi * R**3) * np.dot(R_vec, normal)

# 3. 奇异积分处理
def singular_integration(k, centroid, normal, area):
    """常数元奇异积分处理 (示例: 球面近似法)"""
    # G_ii (CPV积分)
    G_ii = -area / (4 * np.pi)  # 静态近似
    
    # H_ii (Hadamard有限部分积分)
    H_ii = -0.5  # 光滑曲面常数值
    
    # Helmholtz方程修正
    H_ii -= 1j * k * area / (4 * np.pi)  
    return G_ii, H_ii

# 4. BEM核心计算类
class HelmholtzBEM:
    def __init__(self, mesh, k):
        self.mesh = mesh
        self.k = k  # 波数
        self.N = len(mesh.faces)
        self.G = np.zeros((self.N, self.N), dtype=np.complex128)
        self.H = np.zeros((self.N, self.N), dtype=np.complex128)
    
    def assemble_matrices(self):
        """组装G和H矩阵"""
        for i in range(self.N):
            r_i = self.mesh.centroids[i]
            n_i = self.mesh.normals[i]
            
            for j in range(self.N):
                r_j = self.mesh.centroids[j]
                
                if i == j:
                    # 奇异积分处理
                    self.G[i, j], self.H[i, j] = singular_integration(
                        self.k, r_j, self.mesh.normals[j], self.mesh.areas[j]
                    )
                else:
                    # 非奇异积分
                    G_val = green_function(self.k, r_i, r_j)
                    H_val = green_function_derivative(self.k, r_i, r_j, self.mesh.normals[j])
                    
                    # 常数元近似
                    self.G[i, j] = G_val * self.mesh.areas[j]
                    self.H[i, j] = H_val * self.mesh.areas[j]
    
    def apply_boundary_conditions(self, bc_types, bc_values):
        """
        处理边界条件
        bc_types: 边界类型列表 (0: Dirichlet, 1: Neumann, 2: Robin)
        bc_values: 对应的边界值
        """
        A = np.zeros((self.N, self.N), dtype=np.complex128)
        b = np.zeros(self.N, dtype=np.complex128)
        
        # 对角增强 (E矩阵) - 公式中 H = H0 + E
        E = np.eye(self.N) * 0.5
        
        for i in range(self.N):
            if bc_types[i] == 0:  # Dirichlet (给定Φ)
                A[i] = self.H[i] + E[i]
                b[i] = np.sum(self.G[i] * bc_values[i])  # v未知
            elif bc_types[i] == 1:  # Neumann (给定v)
                A[i] = self.G[i]
                b[i] = -np.sum((self.H[i] + E[i]) * bc_values[i])  # Φ未知
            elif bc_types[i] == 2:  # Robin (混合)
                a, b_robin = bc_values[i]  # aΦ + bv = 0
                A[i] = a * (self.H[i] + E[i]) + b_robin * self.G[i]
                b[i] = 0
        
        return A, b
    
    def solve_system(self, A, b):
        """求解线性系统"""
        return np.linalg.solve(A, b)
    
    def compute_potential(self, r_target, phi, v):
        """计算任意目标点声压势"""
        phi_target = 0j
        for j in range(self.N):
            r_j = self.mesh.centroids[j]
            G_integral = green_function(self.k, r_target, r_j) * self.mesh.areas[j]
            H_integral = green_function_derivative(self.k, r_target, r_j, self.mesh.normals[j]) * self.mesh.areas[j]
            
            phi_target += G_integral * v[j] - H_integral * phi[j]
        
        return phi_target
    def visualize_pressure_field(self, phi, v, plane='xy', z=0.0, x_range=(-2, 2), y_range=(-2, 2), resolution=100):
        """
        计算并可视化声压场在某个平面上的分布
        
        参数:
        plane: 可视化平面 ('xy', 'xz' 或 'yz')
        z: 平面在垂直于平面的轴上的位置
        x_range, y_range: 平面范围
        resolution: 网格分辨率
        """
        print(f"正在计算平面 {plane} 上的声压场分布...")
        
        # 创建网格点
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X, dtype=np.complex128)
        
        # 计算每个网格点处的声压
        for i in range(resolution):
            for j in range(resolution):
                if plane == 'xy':
                    point = [X[i, j], Y[i, j], z]
                elif plane == 'xz':
                    point = [X[i, j], z, Y[i, j]]
                elif plane == 'yz':
                    point = [z, X[i, j], Y[i, j]]
                
                # 计算该点的声压
                Z[i, j] = self.compute_potential(point, phi, v)
        
        # 计算声压幅度 (dB)
        magnitude = np.abs(Z)
        pressure_dB = 20 * np.log10(magnitude + 1e-12)  # 避免log(0)，加小偏移量
        
        # 设置绘图
        plt.figure(figsize=(12, 10))
        
        # 绘制声压云图
        plt.imshow(pressure_dB, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                  origin='lower', cmap='viridis')
        plt.colorbar(label='声压级 (dB)')
        
        # 添加标题和标签
        plt.title(f'声压场分布 ({plane}平面，位置z={z:.1f})')
        if plane == 'xy':
            plt.xlabel('X')
            plt.ylabel('Y')
        elif plane == 'xz':
            plt.xlabel('X')
            plt.ylabel('Z')
        elif plane == 'yz':
            plt.xlabel('Y')
            plt.ylabel('Z')
        
        # 叠加显示球体的截面
        self._plot_sphere_cross_section(plane, z, x_range, y_range)
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z
    
    def _plot_sphere_cross_section(self, plane, z, x_range, y_range):
        """绘制球体的截面轮廓"""
        # 创建圆的角度参数
        theta = np.linspace(0, 2 * np.pi, 100)
        
        if plane == 'xy':
            # 球体在XY平面的截面
            center_x = 0
            center_y = 0
            radius = np.sqrt(1 - z**2) if abs(z) <= 1 else 0
            if radius > 0:
                circle_x = center_x + radius * np.cos(theta)
                circle_y = center_y + radius * np.sin(theta)
                plt.plot(circle_x, circle_y, 'r--', linewidth=1.5, label='球体截面')
        
        elif plane == 'xz':
            # 球体在XZ平面的截面
            center_x = 0
            center_z = 0
            radius = np.sqrt(1 - z**2) if abs(z) <= 1 else 0
            if radius > 0:
                circle_x = center_x + radius * np.cos(theta)
                circle_z = center_z + radius * np.sin(theta)
                plt.plot(circle_x, circle_z, 'r--', linewidth=1.5, label='球体截面')
        
        elif plane == 'yz':
            # 球体在YZ平面的截面
            center_y = 0
            center_z = 0
            radius = np.sqrt(1 - z**2) if abs(z) <= 1 else 0
            if radius > 0:
                circle_y = center_y + radius * np.cos(theta)
                circle_z = center_z + radius * np.sin(theta)
                plt.plot(circle_y, circle_z, 'r--', linewidth=1.5, label='球体截面')
        
        plt.legend()
        plt.xlim(x_range)
        plt.ylim(y_range)

# 5. 主程序流程
if __name__ == "__main__":
    # 参数设置
    frequency = 500  # Hz
    c0 = 343  # 声速 (m/s)
    k = 2 * np.pi * frequency / c0  # 波数


    # 1. 创建几何模型 (示例球体)
    mesh = SurfaceMesh()
    sphere_file = "sphere.stl"
    if not os.path.exists(sphere_file):
        print("正在生成球体STL文件...")
        mesh.generate_sphere_stl(radius=1.0, resolution=15, filename=sphere_file)
    
    #加载STL文件
    mesh.load_from_stl(sphere_file)
    mesh.visualize()

    # 2. 组装BEM矩阵
    bem = HelmholtzBEM(mesh, k)
    bem.assemble_matrices()
    
    # 3. 设置边界条件 (示例)
    bc_types = np.zeros(mesh.N)
    bc_values = np.zeros(mesh.N, dtype=np.complex128)
    
    # 上半球: Dirichlet边界 (Φ=1)
    upper = np.where(mesh.centroids[:, 2] > 0)[0]
    bc_types[upper] = 1
    bc_values[upper] = 0.5
    
    # 下半球: Neumann边界 (v=0.5)
    lower = np.where(mesh.centroids[:, 2] <= 0)[0]
    bc_types[lower] = 1
    bc_values[lower] = 0.5
    
    # 4. 构建并求解方程组
    A, b = bem.apply_boundary_conditions(bc_types, bc_values)
    x = bem.solve_system(A, b)
    
    # 5. 分离解变量 (根据边界条件类型)
    phi = np.zeros(mesh.N, dtype=np.complex128)
    v = np.zeros(mesh.N, dtype=np.complex128)
    
    for i in range(mesh.N):
        if bc_types[i] == 0:
            phi[i] = bc_values[i]
            v[i] = x[i]
        else:
            v[i] = bc_values[i]
            phi[i] = x[i]
    
    # 6. 计算场点声势
    target_point = np.array([0, 0, 2.0])  # 球外点
    phi_target = bem.compute_potential(target_point, phi, v)
    print(f"Target potential at {target_point}: {phi_target:.4f}")

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # XY平面（横截面）
    # bem.visualize_pressure_field(phi, v, plane='xy', z=0.5, 
    #                             x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
    #                             resolution=40)
    
    # XZ平面（子午面）
    bem.visualize_pressure_field(phi, v, plane='xz', z=0.0, 
                                x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
                                resolution=40)
    
    # YZ平面（另一个方向剖面）
    bem.visualize_pressure_field(phi, v, plane='yz', z=0.0, 
                                x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
                                resolution=40)