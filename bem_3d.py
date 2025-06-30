import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from stl import mesh as stl_mesh  
import time
from tqdm import tqdm
# 1. 几何模型定义与面元划分
class SurfaceMesh:
    def __init__(self):
        """
        初始化SurfaceMesh类

        Attributes:
            vertices: 顶点坐标 (Nv, 3)
            faces: 面元顶点索引 (Nf, 3)
            centroids: 面元质心坐标 (Nf, 3)
            areas: 面元面积 (Nf,)
            normals: 面元单位法向量 (Nf, 3)
            N: 面元数量
            radius: 球体半径（如果适用）
        """
        self.vertices = None      # 顶点坐标 (Nv, 3)
        self.faces = None         # 面元顶点索引 (Nf, 3)
        self.centroids = None     # 面元质心坐标 (Nf, 3)
        self.areas = None         # 面元面积 (Nf,)
        self.normals = None       # 面元单位法向量 (Nf, 3)
        self.N = None             # 面元数量
        self.radius = None        # 球体半径（如果适用）

    def generate_sphere_stl(self, radius=1.0, resolution=12, filename="sphere.stl"):
        """
        生成示例球体的STL文件
        Args:
            radius (float): 球体的半径，单位为米。
            resolution (int): 球体的分辨率，用于控制球体表面的精细程度。
        Returns:
            str: 生成的STL文件的文件名。
        """
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
        stl_mesh_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh_obj.vectors[i][j] = vertices[face[j]]

        # 保存STL文件
        stl_mesh_obj.save(filename)
        print(f"已生成球面STL文件: {filename} (半径={radius:.1f}m, 分辨率={resolution})")
        self.radius = radius
        return filename

    def generate_big_small_sphere_stl(self, radius=1.0, resolution=12, small_radius = 0.4, small_resolution = 12,small_center = [2.0, 0.0, 0.0], filename="sphere_big_small.stl"):

        """
        生成示例球体的STL文件，包含主球体和额外的小球
        Args:
            radius (float): 主球体的半径，单位为米。
            resolution (int): 主球体的分辨率，用于控制球体表面的精细程度。
        Returns:
            str: 生成的STL文件的文件名。
        """
        # 生成主球体顶点
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

        # 创建主球体面元数据
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
        
        # ======= 添加外部小球 =======
        
        # 记录当前顶点数量
        base_vertex_index = len(vertices)
        
        # 生成小球顶点 (复用相同逻辑)
        # 北极点
        vertices.append([small_center[0], small_center[1], small_center[2] + small_radius])
        # 中间层顶点
        num_theta_small = 2 * small_resolution
        for i in range(1, small_resolution):
            phi = np.pi * i / small_resolution
            for j in range(num_theta_small):
                theta = 2 * np.pi * j / num_theta_small
                x = small_center[0] + small_radius * np.sin(phi) * np.cos(theta)
                y = small_center[1] + small_radius * np.sin(phi) * np.sin(theta)
                z = small_center[2] + small_radius * np.cos(phi)
                vertices.append([x, y, z])
        # 南极点
        vertices.append([small_center[0], small_center[1], small_center[2] - small_radius])
        
        # 创建小球面元数据
        # 北极区
        for j in range(num_theta_small):
            v0 = base_vertex_index
            v1 = base_vertex_index + 1 + j
            v2 = base_vertex_index + 1 + (j + 1) % num_theta_small
            faces.append([v0, v1, v2])
        
        # 中间区
        for i in range(1, small_resolution - 1):
            for j in range(num_theta_small):
                start = base_vertex_index + 1 + (i - 1) * num_theta_small
                next_start = base_vertex_index + 1 + i * num_theta_small
                v0 = start + j
                v1 = start + (j + 1) % num_theta_small
                v2 = next_start + j
                v3 = next_start + (j + 1) % num_theta_small
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])
        
        # 南极区
        south_pole_small = len(vertices) - 1
        last_ring_start = base_vertex_index + 1 + (small_resolution - 2) * num_theta_small
        for j in range(num_theta_small):
            v0 = last_ring_start + j
            v1 = last_ring_start + (j + 1) % num_theta_small
            v2 = south_pole_small
            faces.append([v0, v1, v2])
        # ======= 外部小球添加完成 =======

        # 创建STL网格
        stl_mesh_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh_obj.vectors[i][j] = vertices[face[j]]

        # 保存STL文件
        stl_mesh_obj.save(filename)
        print(f"已生成包含主球体和小球的STL文件: {filename}")
        print(f"主球体: 半径={radius:.1f}m, 分辨率={resolution}")
        print(f"外部小球: 位置={small_center}, 半径={small_radius:.1f}m, 分辨率={small_resolution}")
        self.radius = radius
        return filename
    
    def load_from_stl(self, filename):
        """
        从任意STL文件加载网格模型
        Args:
            filename (str): STL文件的路径。
        Returns:
            None
        """
        # 从STL文件加载网格数据
        stl_mesh_obj = stl_mesh.Mesh.from_file(filename)
        self.N = len(stl_mesh_obj.vectors)
        
        # 提取顶点和面元
        all_vertices = stl_mesh_obj.vectors.reshape(-1, 3)
        self.vertices = np.unique(all_vertices, axis=0)
        
        # 创建面元索引
        self.faces = []
        vertex_map = {}
        for i, vertex in enumerate(self.vertices):
            vertex_map[tuple(vertex)] = i
            
        for vectors in stl_mesh_obj.vectors:
            face = []
            for vector in vectors:
                idx = vertex_map[tuple(vector)]
                face.append(idx)
            self.faces.append(face)
        
        self.faces = np.array(self.faces)
        
        # 计算面元属性
        self._compute_face_properties()
        print(f"已加载STL文件: {filename} ({self.N}个面元)")

    def _compute_face_properties(self):
        """
        计算每个面元的质心、面积和法向量
        Returns:
            None
        """
        self.centroids = np.zeros((self.N, 3))
        self.areas = np.zeros(self.N)
        self.normals = np.zeros((self.N, 3))
        
        for i in range(self.N):
            v0, v1, v2 = self.vertices[self.faces[i]]
            
            # 计算质心
            centroid = (v0 + v1 + v2) / 3.0
            self.centroids[i] = centroid
            
            # 计算法向量
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            # 计算面积
            area = 0.5 * np.linalg.norm(normal)
            self.areas[i] = area
            
            # 归一化法向量
            self.normals[i] = normal / np.linalg.norm(normal)

    def visualize(self):
        """
        可视化3D网格模型
        Returns:
            None
        """
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
    """三维自由声场基本解

    Args:
        k (float): 波数。
        r (np.ndarray): 计算点坐标，形状为 (3,)。
        r0 (np.ndarray): 源点坐标，形状为 (3,)。

    Returns:
        complex: 格林函数的值。
    """
    R = distance.euclidean(r, r0)
    return np.exp(-1j * k * R) / (4 * np.pi * R)
    #return np.exp(1j * k * R) / (4 * np.pi * R)
    # 两种基本解的定义，都可取用，不同处在于最后输出的声势虚部取反，声压需取模因此数值相同，振速虚部取反，取模后数值相同，因此两种定义无数值区别

def green_function_derivative(k, r, r0, normal):
    """计算格林函数的法向导数

    Args:
        k (float): 波数。
        r (np.ndarray): 计算点坐标，形状为 (3,)。
        r0 (np.ndarray): 源点坐标，形状为 (3,)。
        normal (np.ndarray): 面元的单位法向量，形状为 (3,)。

    Returns:
        complex: 格林函数的法向导数。
    """
    R_vec = r - r0
    R = np.linalg.norm(R_vec)
    return (-1j * k * R - 1) * np.exp(-1j * k * R) / (4 * np.pi * R**3) * np.dot(R_vec, normal)
    #return (1j * k * R - 1) * np.exp(1j * k * R) / (4 * np.pi * R**3) * np.dot(R_vec, normal)

# 3. 奇异积分处理
def singular_integration(k, centroid, normal, area):
    """
    常数元奇异积分处理 (示例: 球面近似法)
    Args:
        k (float): 波数。
        centroid (np.ndarray): 面元的质心坐标，形状为 (3,)。
        normal (np.ndarray): 面元的单位法向量，形状为 (3,)。
        area (float): 面元的面积。
    Returns:
        tuple: 包含格林函数和哈达玛有限部分积分的元组 (G_ii, H_ii)。
            G_ii (float): 格林函数的CPV积分值。
            H_ii (float): 哈达玛有限部分积分值。
    """
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
        """
        初始化HelmholtzBEM类
        Attributes:
            mesh (SurfaceMesh): 表示网格模型的SurfaceMesh对象。
            k (float): 波数。
            N (int): 面元数量。
            G (np.ndarray): G矩阵，形状为 (N, N)。
            H (np.ndarray): H矩阵，形状为 (N, N)。
        """
        self.mesh = mesh
        self.k = k  # 波数
        self.N = len(mesh.faces)
        self.G = np.zeros((self.N, self.N), dtype=np.complex128)
        self.H = np.zeros((self.N, self.N), dtype=np.complex128)
        self.radius = mesh.radius

    def assemble_matrices(self):
        """
        计算G和H矩阵
        Returns:
            None
        """
        for i in tqdm(range(self.N), desc="已计算的面元数"):
            r_i = self.mesh.centroids[i]
            n_i = self.mesh.normals[i]
            
            for j in range(self.N):
                r_j = self.mesh.centroids[j]
                
                if i == j:
                    # 奇异积分处理
                    self.G[i, j], self.H[i, j] = singular_integration(
                        self.k, r_j, n_i, self.mesh.areas[j]
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
        Args:
            bc_types (list): 边界类型列表，0: Dirichlet, 1: Neumann, 2: Robin, 就是三类常见边界条件
            bc_values (list): 对应的边界值，对于Dirichlet和Neumann，边界值为单个值；对于Robin，边界值为 (a, b) 的元组。
        Returns:
            tuple: 包含组装后的矩阵A和向量b的元组 (A, b)。
                A (np.ndarray): 组装后的矩阵A，形状为 (N, N)。
                b (np.ndarray): 组装后的向量b，形状为 (N,)。
        """
        A = np.zeros((self.N, self.N), dtype=np.complex128)
        b = np.zeros(self.N, dtype=np.complex128)
        
        # 对角增强 (E矩阵) - 公式中 H = H0 + E
        E = np.eye(self.N) * 0.5 
        
        for i in range(self.N):
            # (H + E) Φ = G v
            if bc_types[i] == 0:  # Dirichlet (给定Φ), 求v
                A[i,:] = self.G[i,:]
                b[i] = np.sum((self.H[i,:] + E[i,:]) * bc_values[i]) # 
            elif bc_types[i] == 1:  # Neumann (给定v), 求Φ
                A[i,:] = self.H[i,:] + E[i,:]
                b[i] = np.sum(self.G[i,:] * bc_values[i])
            elif bc_types[i] == 2:  # Robin (aΦ + bv = 0), 求Φ
                # 0 = aGΦ + bGv = aGΦ + b(H + E)Φ
                a, b_robin = bc_values[i] 
                A[i,:] = a * self.G[i,:] + b_robin * (self.H[i,:] + E[i,:])
                b[i] = 0
        
        return A, b

    def solve_system(self, A, b):
        """
        线性方程组求解
        Args:
            A (np.ndarray): 系统矩阵A，形状为 (N, N)。
            b (np.ndarray): 系统向量b，形状为 (N,)。
        Returns:
            np.ndarray: 解向量，形状为 (N,)。
        """
        return np.linalg.solve(A, b)

    def compute_potential(self, r_target, phi, v):
        """
        计算任意目标点声势
        Args:
            r_target (np.ndarray): 目标点坐标，形状为 (3,)。
            phi (np.ndarray): 面元的声势值，形状为 (N,)。
            v (np.ndarray): 面元的声压值，形状为 (N,)。
        Returns:
            complex: 目标点的声势。
        """
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
        Args:
            phi (np.ndarray): 面元的声势值，形状为 (N,)。
            v (np.ndarray): 面元的声压值，形状为 (N,)。
            plane (str): 可视化平面 ('xy', 'xz' 或 'yz')。
            z (float): 平面在垂直于平面的轴上的位置。
            x_range (tuple): 平面在X轴上的范围。
            y_range (tuple): 平面在Y轴上的范围。
            resolution (int): 网格分辨率。
        Returns:
            tuple: 包含网格点X, Y和声压场Z的元组 (X, Y, Z)。
                X (np.ndarray): 网格点在X轴上的坐标。
                Y (np.ndarray): 网格点在Y轴上的坐标。
                Z (np.ndarray): 网格点处的声压场值。
        """
        print(f"正在计算平面 {plane} 上的声压场分布...")
        # 创建网格点
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X, dtype=np.complex128)
        
        # 计算每个网格点处的声压
        for i in tqdm(range(resolution), desc="声场生成进度"):
            for j in range(resolution):
                if plane == 'xy':
                    point = [X[i, j], Y[i, j], z]
                elif plane == 'xz':
                    point = [X[i, j], z, Y[i, j]]
                elif plane == 'yz':
                    point = [z, X[i, j], Y[i, j]]
                
                # 计算该点的声势
                Z[i, j] = self.compute_potential(point, phi, v)
        
        # 计算声压幅度 (dB)
        rho0 = 1.2
        c0 =343
        magnitude = np.abs(self.k*rho0*c0*Z)
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
        if self.radius is not None:
            self._plot_sphere_cross_section(plane, z, x_range, y_range)
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z, pressure_dB

    def _plot_sphere_cross_section(self, plane, z, x_range, y_range):
        """
        Mesh为球体时运行,绘制球体的截面虚线轮廓,
        Args:
            plane (str): 可视化平面 ('xy', 'xz' 或 'yz')。
            z (float): 平面在垂直于平面的轴上的位置。
            x_range (tuple): 平面在X轴上的范围。
            y_range (tuple): 平面在Y轴上的范围。
        Returns:
            None
        """
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

# 5. 总封装函数
class Mesh2Field:
    def __init__(self, frequency=60, mesh_file='cuboid.stl', bc_types=None, bc_values=None, ):
        """
        重要参数介绍

        :param mesh_file: 3D模型的存储路径
        :param mesh: 3D模型, 归属于SurfaceMesh(), 请关注相关的函数
        :param bc_types(list): 各个mesh所属的边界类型, 0: Dirichlet, 1: Neumann, 2: Robin, 就是三类常见边界条件
        :param bc_values(list): 各个mesh所属的边界类型对应的边界值，对于Dirichlet和Neumann，边界值为单个值；对于Robin，边界值为 (a, b) 的元组。
        :param bem: BEM模拟要用到的函数, 归属于HelmholtzBEM(self.mesh, self.k)
        """

        # 0. 参数设置
        self.frequency = frequency  # Hz
        self.c0 = 343  # 声速 (m/s)
        self.k = 2 * np.pi * frequency / self.c0  # 波数
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        self.mesh = SurfaceMesh()# 创建几何模型
        self.mesh_file = mesh_file
        self.bc_types = bc_types#设置边界
        self.bc_values = bc_values
        self.bc_phi = None
        self.bc_v = None
        # 1. 导入模型
        self.mesh.load_from_stl(self.mesh_file)
        self.mesh.visualize()  # 可视化
        self.bem = HelmholtzBEM(self.mesh, self.k)
        

    def calc_bc_phiv(self):
        """
        计算边界上的声势和振速

        :return: 边界上的声势和振速储存在 self.bc_phi 和 self.bc_v 之中, 下面程序会自己调用, 无需关注
        """
        # 2. 计算HG矩阵
        HG_start_time = time.time()
        self.bem.assemble_matrices()
        HG_end_time = time.time()
        print(f"计算HG矩阵运行时间: {HG_end_time - HG_start_time:.3f} 秒")
        # 3. 解线性方程组
        equartion_start_time = time.time()
        A, b = self.bem.apply_boundary_conditions(self.bc_types, self.bc_values)
        x = self.bem.solve_system(A, b)
        equartion_end_time = time.time()
        print(f"求解线性方程组运行时间: {equartion_end_time - equartion_start_time:.3f} 秒")
        # 4. 分解得到边界上的声势和振速
        phi = np.zeros(self.mesh.N, dtype=np.complex128)
        v = np.zeros(self.mesh.N, dtype=np.complex128)
        for i in range(self.mesh.N):
            if self.bc_types[i] == 0:
                phi[i] = self.bc_values[i]
                v[i] = x[i]
            elif self.bc_types[i] == 1:
                v[i] = self.bc_values[i]
                phi[i] = x[i]
            elif self.bc_types[i] == 2:
                phi[i] = x[i]
                a, b_robin = self.bc_values[i] # Robin (aΦ + bv = 0), 求Φ
                v[i] = -a/b_robin * phi[i]
        self.bc_phi = phi
        self.bc_v = v

    def visualize_pressure_field(self, resolution=40, plane='xy',z=0.0, x_range=(-2.5, 2.5), y_range=(-2.5, 2.5)):
        """
        计算并可视化声压场

        :param resolution: 压力场的分辨率，默认为40（即40x40）
        :param plane: 选定计算的平面朝向，'xy', 'xz', 'yz'之一, 默认为'xy'
        :param z: 选定要计算的平面的高度，z表示要计算的平面与所选平面的法向距离
        :param x_range: x轴的范围，元组类型，默认为(-2.5, 2.5), 单位为米
        :param y_range: y轴的范围，元组类型，默认为(-2.5, 2.5), 单位为米
        :return: X, Y, Z 都为40x40的矩阵; X,Y包含这些点的X,Y位置信息; Z为这些点上的声势(复数，线性); pressure_dB为这些点上的声压级(dB)
        """
        # XY平面
        X,Y,Z,pressure_dB=self.bem.visualize_pressure_field(self.bc_phi, self.bc_v, plane=plane, z=z, 
                                        x_range=x_range, y_range=y_range,
                                        resolution=resolution)
        return X,Y,Z,pressure_dB

    def calc_point_potential(self,target_point):
        """
        计算特定点的声势, 要求此点在声源外部

        :param target_point: 要求点的空间坐标(3x1), 单位为米,
        :return: phi_target为此点上的声势(复数，线性)
        """
        phi_target = self.bem.compute_potential(target_point, self.bc_phi, self.bc_v)
        print(f"Target potential at {target_point}: {phi_target:.4f}")
        return phi_target

# 下面的主程序是使用任意的stl作为输入的, 可以使用这个程序生成需要的stl
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    mesh = SurfaceMesh()
    #1.1球体模型
    radius = 1.0
    mesh.radius = radius
    resolution = 15
    small_radius = 0.2
    small_resolution = 12
    small_center = [2.0, 0.0, 0.0]
    sphere_file = f"sphere_radius_{radius}_{small_radius}_resolution_{resolution}_{small_resolution}_position{small_center}.stl"
    if not os.path.exists(sphere_file):
        print("正在生成球体STL文件...")
        #mesh.generate_big_small_sphere_stl(radius=radius, resolution=resolution, filename=sphere_file)
        mesh.generate_big_small_sphere_stl(radius=radius, resolution=resolution, small_radius=small_radius, small_resolution=small_resolution,small_center=small_center, filename=sphere_file)
    else:
        print("已生成球体STL文件,直接加载")
    #加载球体STL文件
    mesh.load_from_stl(sphere_file)
    mesh.visualize()

# 主程序
if __name__ == "__main__":
    #下面这一大段是在生成边界条件,这是由于stl不包含材质信息,在NN训练时最好考虑使用包含边界条件的数据
    mesh = SurfaceMesh()
    #mesh_file = "sphere_radius_1.0_resolution_15.stl"
    mesh_file = "sphere_radius_1.0_0.2_resolution_15_12_position[2.0, 0.0, 0.0].stl"
    mesh.load_from_stl(mesh_file)
    bc_types = np.zeros(mesh.N)
    bc_values = np.zeros(mesh.N, dtype=np.complex128)
    # 上半球: Neumann边界 (v=0.5)
    upper = np.where(mesh.centroids[:, 2] > 0)[0]
    bc_types[upper] = 1
    bc_values[upper] = 0.5
    # 下半球: Neumann边界 (v=0.5)
    lower = np.where(mesh.centroids[:, 2] <= 0)[0]
    bc_types[lower] = 1
    bc_values[lower] = 0.5
    # mesh_number = 400
    # bc_types[mesh_number] = 1
    # bc_values[mesh_number] = 20

    #下面是示例
    #基本输入为: stl模型, 边界条件, 频率(注意一次只能模拟单频率)
    #生成声场自定义参数输入为: 声场所在平面, 声场范围, 声场网格分辨率
    #输出为：声场声压， 声场声势， 声场各点位置
    #mesh_file = "sphere_radius_1.0_resolution_15.stl"
    mesh_file = "sphere_radius_1.0_0.2_resolution_15_12_position[2.0, 0.0, 0.0].stl"
    mesh2field_test = Mesh2Field(frequency=100,mesh_file=mesh_file,bc_types=bc_types,bc_values=bc_values)
    #计算边界上的声势和振速
    mesh2field_test.calc_bc_phiv()
    #计算特定点的声势
    target_point = np.array([0, 0, 2.0])  
    mesh2field_test.calc_point_potential(target_point=target_point)
    #计算并可视化声压场
    X,Y,potential,pressure_dB=mesh2field_test.visualize_pressure_field(resolution=100, plane='xy',z=0.0, x_range=(-2.5, 2.5), y_range=(-2.5, 2.5)) 
