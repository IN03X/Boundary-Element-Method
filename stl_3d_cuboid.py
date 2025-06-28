"""生成三维的长方体st，用于检查BEM"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

def create_cuboid_mesh(length=2.0, width=1.0, height=0.5, resolution=1, filename="cuboid.stl"):
    """
    生成长方体网格并保存为STL文件
    
    参数:
        length: 长度 (x方向)
        width: 宽度 (y方向)
        height: 高度 (z方向)
        resolution: 每个面的分割数
        filename: 导出的STL文件名
    """
    # 定义长方体的8个顶点
    vertices = np.array([
        [0, 0, 0],           # 0: 左下后
        [length, 0, 0],       # 1: 右下后
        [length, width, 0],   # 2: 右下前
        [0, width, 0],       # 3: 左下前
        [0, 0, height],      # 4: 左后上
        [length, 0, height],  # 5: 右后上
        [length, width, height], # 6: 右前上
        [0, width, height]    # 7: 左前上
    ])
    
    # 定义每个面的顶点索引
    # 每个面由2个三角形组成
    faces = []
    
    # 分割每个面
    for i in range(resolution):
        for j in range(resolution):
            # 前表面 (z=width)
            s = i / resolution
            t = j / resolution
            next_s = (i+1) / resolution
            next_t = (j+1) / resolution
            
            # 前表面 (z=width)
            v0 = [s*length, width, t*height]
            v1 = [next_s*length, width, t*height]
            v2 = [next_s*length, width, next_t*height]
            v3 = [s*length, width, next_t*height]
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            # 后表面 (z=0)
            v0 = [s*length, 0, t*height]
            v1 = [next_s*length, 0, t*height]
            v2 = [next_s*length, 0, next_t*height]
            v3 = [s*length, 0, next_t*height]
            faces.append([v0, v2, v1])
            faces.append([v0, v3, v2])
            
            # 上表面 (y=height)
            v0 = [s*length, t*width, height]
            v1 = [next_s*length, t*width, height]
            v2 = [next_s*length, next_t*width, height]
            v3 = [s*length, next_t*width, height]
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            # 下表面 (y=0)
            v0 = [s*length, t*width, 0]
            v1 = [next_s*length, t*width, 0]
            v2 = [next_s*length, next_t*width, 0]
            v3 = [s*length, next_t*width, 0]
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            # 右侧面 (x=length)
            v0 = [length, t*width, s*height]
            v1 = [length, next_t*width, s*height]
            v2 = [length, next_t*width, next_s*height]
            v3 = [length, t*width, next_s*height]
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
            
            # 左侧面 (x=0)
            v0 = [0, t*width, s*height]
            v1 = [0, next_t*width, s*height]
            v2 = [0, next_t*width, next_s*height]
            v3 = [0, t*width, next_s*height]
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    # 创建STL网格对象
    cuboid = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    
    # 将三角形添加到网格
    for i, f in enumerate(faces):
        for j in range(3):
            cuboid.vectors[i][j] = f[j]
    
    # 保存STL文件
    cuboid.save(filename)
    print(f"已生成长方体STL文件: {filename}")
    print(f"尺寸: 长={length}m, 宽={width}m, 高={height}m")
    
    # 可视化长方体
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 添加网格
    mesh_collection = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolor='k')
    mesh_collection.set_facecolor('cyan')
    ax.add_collection3d(mesh_collection)
    
    # 添加顶点
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=50, c='red')
    
    # 标记顶点
    for i, vertex in enumerate(vertices):
        ax.text(vertex[0], vertex[1], vertex[2], f'v{i}', size=12, zorder=1, color='k')
    
    # 设置坐标轴
    ax.set_xlabel('X (长度)')
    ax.set_ylabel('Y (宽度)')
    ax.set_zlabel('Z (高度)')
    ax.set_xlim([0, length])
    ax.set_ylim([0, width])
    ax.set_zlim([0, height])
    ax.set_title(f'长方体网格可视化 (面元数量: {len(faces)})')
    
    # 添加参考线
    # 底部矩形
    ax.plot([0, length, length, 0, 0], 
            [0, 0, width, width, 0], 
            [0, 0, 0, 0, 0], 'k-', linewidth=2)
    
    # 顶部矩形
    ax.plot([0, length, length, 0, 0], 
            [0, 0, width, width, 0], 
            [height, height, height, height, height], 'k-', linewidth=2)
    
    # 垂直线
    ax.plot([0, 0], [0, 0], [0, height], 'k-', linewidth=2)
    ax.plot([length, length], [0, 0], [0, height], 'k-', linewidth=2)
    ax.plot([length, length], [width, width], [0, height], 'k-', linewidth=2)
    ax.plot([0, 0], [width, width], [0, height], 'k-', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    return filename

if __name__ == "__main__":
    # 生成长方体网格并导出STL
    # 默认：长2m, 宽1m, 高0.5m, 每个面分割数=1
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    create_cuboid_mesh(length=1, width=0.8, height=0.5, resolution=10)