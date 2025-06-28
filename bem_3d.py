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

    from tqdm import tqdm

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
        if self.radius is not None:
            self._plot_sphere_cross_section(plane, z, x_range, y_range)
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z

    
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


# 5. 主程序流程
if __name__ == "__main__":
    # 参数设置
    frequency = 100  # Hz
    c0 = 343  # 声速 (m/s)
    k = 2 * np.pi * frequency / c0  # 波数
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    mesh = SurfaceMesh()
    
    # 1. 创建几何模型
    #1.1球体模型
    radius = 1.0
    mesh.radius = radius
    resolution = 30
    sphere_file = f"sphere_radius_{radius}_resolution_{resolution}.stl"
    if not os.path.exists(sphere_file):
        print("正在生成球体STL文件...")
        mesh.generate_sphere_stl(radius=radius, resolution=resolution, filename=sphere_file)
    else:
        print("已生成球体STL文件,直接加载")
    #加载球体STL文件
    mesh.load_from_stl(sphere_file)
    #mesh.visualize()

    #1.2长方体模型
    # # 加载长方体STL文件
    # cuboid_file = "cuboid.stl"
    # mesh.load_from_stl(cuboid_file)
    # mesh.visualize()  # 可视化长方体网格
    
    # 2. 组装BEM矩阵
    HG_start_time = time.time()
    bem = HelmholtzBEM(mesh, k)
    bem.assemble_matrices()
    HG_end_time = time.time()
    print(f"计算HG矩阵运行时间: {HG_end_time - HG_start_time:.3f} 秒")
    
    # 3. 设置边界条件 (脉动球模型)
    bc_start_time = time.time()
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
    bc_end_time = time.time()
    print(f"边界条件应用时间: {bc_end_time - bc_start_time:.3f} 秒")
    
    # 4. 构建并求解方程组
    equartion_start_time = time.time()
    A, b = bem.apply_boundary_conditions(bc_types, bc_values)
    x = bem.solve_system(A, b)
    equartion_end_time = time.time()
    print(f"求解线性方程组运行时间: {equartion_end_time - equartion_start_time:.3f} 秒")
    
    # 5. 分离解变量 (根据边界条件类型)
    phi = np.zeros(mesh.N, dtype=np.complex128)
    v = np.zeros(mesh.N, dtype=np.complex128)
    
    for i in range(mesh.N):
        if bc_types[i] == 0:
            phi[i] = bc_values[i]
            v[i] = x[i]
        elif bc_types[i] == 1:
            v[i] = bc_values[i]
            phi[i] = x[i]
        elif bc_types[i] == 2:
            phi[i] = x[i]

    
    # 6. 计算指定位置处场点声势
    # target_point = np.array([0, 0, 2.0])  # 球外点
    # phi_target = bem.compute_potential(target_point, phi, v)
    # print(f"Target potential at {target_point}: {phi_target:.4f}")

    # 7. 可视化声场
    # XY平面（横截面）
    # bem.visualize_pressure_field(phi, v, plane='xy', z=0.5, 
    #                             x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
    #                             resolution=40)
    
    # XZ平面（子午面）
    bem.visualize_pressure_field(phi, v, plane='xz', z=0.0, 
                                x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
                                resolution=80)

    # YZ平面
    bem.visualize_pressure_field(phi, v, plane='yz', z=0.0, 
                                x_range=(-2.5, 2.5), y_range=(-2.5, 2.5),
                                resolution=40)